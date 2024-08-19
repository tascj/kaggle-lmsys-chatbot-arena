# modified from transformers==4.43.3

# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
from transformers.models.llama.configuration_llama import LlamaConfig

from xformers.ops import fmha
from .ops.rms_norm import rms_norm
from .ops.silu_and_mul import silu_and_mul_fwd
from .ops.fused_rotary_emb import fused_rotary_emb
from .ops.flash_attention_nopad import context_attention_fwd

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlamaConfig"


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        with torch.cuda.device(hidden_states.device):
            return rms_norm(
                hidden_states.contiguous(), self.weight, self.variance_epsilon
            )

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        # self.gate_proj = nn.Linear(
        #     self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        # )
        # self.up_proj = nn.Linear(
        #     self.hidden_size, self.intermediate_size, bias=config.mlp_bias
        # )
        self.gate_up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size * 2, bias=config.mlp_bias
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=config.mlp_bias
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        with torch.cuda.device(x.device):
            gate_out = silu_and_mul_fwd(gate_up)
        down_proj = self.down_proj(gate_out)
        return down_proj


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, layer_idx: Optional[int] = None):
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
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
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
            self.hidden_size, self.hidden_size, bias=config.attention_bias
        )

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
                    logit_softcapping=0.0,
                )
            attn_output = query_states.reshape(q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(self, hidden_states, seq_info, inv_freq):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(hidden_states, seq_info, inv_freq)
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True

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


class LlamaModel(LlamaPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize RoPE
        # dim = config.hidden_size // config.num_attention_heads
        # inv_freq = 1.0 / (
        #     config.rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        # )
        # seq = torch.arange(config.max_position_embeddings, dtype=inv_freq.dtype)
        # freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        # emb = torch.cat((freqs, freqs), dim=-1)
        # emb = emb.reshape(emb.size(0), 1, 1, emb.size(1))
        # self.register_buffer("rotary_emb", emb)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids, seq_info, inv_freq):
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(hidden_states, seq_info, inv_freq)

        hidden_states = self.norm(hidden_states)
        return hidden_states


class LlamaForSequenceClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = LlamaModel(config)
        self.score = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Dropout(0.1),
            nn.GELU(),
            nn.Linear(config.hidden_size // 2, 3),
        )

        # Initialize weights and apply final processing
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

        hidden_states = inputs_embeds
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
