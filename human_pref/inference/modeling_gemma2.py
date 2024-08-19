# modified from transformers==4.43.3
# coding=utf-8
# Copyright 2024 Google Inc. HuggingFace Inc. team. All rights reserved.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Optional

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config


from xformers.ops import fmha
from .ops.rms_norm import rms_norm
from .ops.gelu_and_mul import gelu_and_mul_fwd
from .ops.fused_rotary_emb import fused_rotary_emb
from .ops.flash_attention_nopad import context_attention_fwd


logger = logging.get_logger(__name__)


class Gemma2RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        with torch.cuda.device(x.device):
            return rms_norm(x.contiguous(), self.weight + 1, self.eps)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Gemma2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.gate_up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=False
        )
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        # return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        gate_up = self.gate_up_proj(x)
        with torch.cuda.device(x.device):
            gate_out = gelu_and_mul_fwd(gate_up)
        return self.down_proj(gate_out)


class Gemma2Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: Gemma2Config, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True
        self.scaling = config.query_pre_attn_scalar**-0.5

        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias
        )
        # self.sliding_window = config.sliding_window if not bool(layer_idx % 2) else None
        # disable sliding window in case T4 sdpa implementation does not support it
        # our max input length is slightly longer than 4096
        self.sliding_window = False

    def forward(self, hidden_states, seq_info, inv_freq):
        q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)

        with torch.cuda.device(hidden_states.device):
            query_states, key_states = fused_rotary_emb(
                query_states[None],
                key_states[None],
                seq_info["position_ids"],
                inv_freq=inv_freq,
                scaling_factor=1.0,
                out_q=query_states[None],
                out_k=key_states[None],
            )

        if "attn_bias" in seq_info:
            # NOTE: This implementation does not do logit softcapping
            q = query_states.reshape(
                1, q_len, self.num_heads // self.num_key_value_groups, -1, self.head_dim
            )
            k = key_states.reshape(
                1, q_len, self.num_key_value_heads, -1, self.head_dim
            )
            v = value_states.reshape(
                1, q_len, self.num_key_value_heads, -1, self.head_dim
            )
            with torch.cuda.device(hidden_states.device):
                attn_output = fmha.memory_efficient_attention(
                    q,
                    k.expand_as(q),
                    v.expand_as(q),
                    seq_info["attn_bias"],
                )
            attn_output = attn_output.reshape(q_len, -1)
        else:
            query_states = query_states[0]
            key_states = key_states[0]

            cu_seqlens = seq_info["cu_seqlens"]
            max_seq_len = seq_info["max_seq_len"]
            with torch.cuda.device(hidden_states.device):
                context_attention_fwd(
                    query_states,
                    key_states,
                    value_states,
                    query_states,  # write to query_states
                    cu_seqlens[:-1],
                    cu_seqlens[1:] - cu_seqlens[:-1],
                    max_seq_len,
                    logit_softcapping=50.0,  # hard-coded gemma-2 value
                )
            attn_output = query_states.reshape(q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        # assert config._attn_implementation == "flash_attention_2"
        self.self_attn = Gemma2Attention(config=config, layer_idx=layer_idx)

        self.mlp = Gemma2MLP(config)
        self.input_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.is_sliding = not bool(layer_idx % 2)
        self.pre_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma2RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.sliding_window = config.sliding_window

    def forward(self, hidden_states, seq_info, inv_freq):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states, seq_info, inv_freq)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class Gemma2PreTrainedModel(PreTrainedModel):
    config_class = Gemma2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Gemma2DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = False
    _supports_quantized_cache = False
    _supports_static_cache = True
    _is_stateful = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class Gemma2Model(Gemma2PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Gemma2DecoderLayer`]

    Args:
        config: Gemma2Config
    """

    def __init__(self, config: Gemma2Config):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Gemma2DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Gemma2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        # dim = config.head_dim
        # inv_freq = 1.0 / (
        #     config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        # )
        # self.register_buffer("inv_freq", inv_freq)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, seq_info, inv_freq):
        inputs_embeds = self.embed_tokens(input_ids)
        # embed positions
        hidden_states = inputs_embeds

        # normalized
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(
            self.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer

        for decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(hidden_states, seq_info, inv_freq)

        hidden_states = self.norm(hidden_states)

        return hidden_states


class Gemma2ForSequenceClassification(Gemma2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 3
        self.model = Gemma2Model(config)
        self.score = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, self.num_labels),
        )

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def forward(self, input_ids, seq_info, inv_freq):
        assert input_ids.size(0) == 1
        hidden_states = self.model(input_ids.squeeze(0), seq_info, inv_freq)

        last_token_inds = seq_info["cu_seqlens"][1:] - 1
        hidden_states = hidden_states[last_token_inds]

        logits = self.score(hidden_states)
        logits = logits.float()
        return logits

    def forward_part1(self, input_ids, seq_info, inv_freq):
        input_ids = input_ids.squeeze(0)
        model = self.model
        inputs_embeds = model.embed_tokens(input_ids)
        # embed positions
        hidden_states = inputs_embeds

        # normalized
        # Gemma2 downcasts the below to float16, causing sqrt(3072)=55.4256 to become 55.5
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = torch.tensor(
            model.config.hidden_size**0.5, dtype=hidden_states.dtype
        )
        hidden_states = hidden_states * normalizer

        n = len(model.layers)
        for decoder_layer in model.layers[: n // 2]:
            hidden_states = decoder_layer(hidden_states, seq_info, inv_freq)

        return hidden_states

    def forward_part2(self, hidden_states, seq_info, inv_freq):
        model = self.model
        for decoder_layer in model.layers[len(model.layers) // 2 :]:
            hidden_states = decoder_layer(hidden_states, seq_info, inv_freq)

        hidden_states = model.norm(hidden_states)

        last_token_inds = seq_info["cu_seqlens"][1:] - 1
        hidden_states = hidden_states[last_token_inds]

        logits = self.score(hidden_states)
        logits = logits.float()
        return logits
