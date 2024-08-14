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
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)
from transformers.models.gemma2.configuration_gemma2 import Gemma2Config


import transformer_engine.pytorch as te
from flash_attn import flash_attn_varlen_func


logger = logging.get_logger(__name__)


class Gemma2RMSNorm(te.RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-6, use_fp32=True):
        super().__init__(hidden_size=dim, eps=eps, zero_centered_gamma=True)
        self.use_fp32 = use_fp32

    def forward(self, x):
        if self.use_fp32:
            return super().forward(x.float()).type_as(x)
        return super().forward(x)


class GeGLUFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, fc1_out) -> torch.Tensor:
        gelu_out = te.cpp_extensions.geglu(
            fc1_out,
            None,
            te.cpp_extensions.FP8FwdTensors.GEMM2_INPUT,
            te.constants.TE_DType[fc1_out.dtype],
        )
        ctx.save_for_backward(fc1_out)

        return gelu_out

    @staticmethod
    def backward(ctx, fc2_dgrad: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        (fc1_out,) = ctx.saved_tensors
        fc2_dgrad = te.cpp_extensions.dgeglu(
            fc2_dgrad, fc1_out, te.constants.TE_DType[fc2_dgrad.dtype]
        )
        return fc2_dgrad


class Gemma2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_activation]

    def forward(self, x):
        # fc1_out = torch.cat((self.gate_proj(x), self.up_proj(x)), dim=-1)
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


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
        # disable sliding window in case T4 sdpa implementation does not support it
        # our max input length is slightly longer than 4096
        if self.config.sliding_window == -1:
            self.sliding_window = None
        else:
            self.sliding_window = config.sliding_window if not bool(layer_idx % 2) else None

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_emb,
        is_last_decoder_layer=False,
    ):
        q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(q_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(q_len, self.num_key_value_heads, self.head_dim)

        # cos, sin = self.rotary_emb(value_states, position_ids)
        # query_states, key_states = apply_rotary_pos_emb(
        #     query_states, key_states, cos, sin
        # )
        query_states = te.attention.FusedRoPEFunc.apply(
            query_states, rotary_emb, "thd", cu_seqlens
        )
        key_states = te.attention.FusedRoPEFunc.apply(
            key_states, rotary_emb, "thd", cu_seqlens
        )

        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).amax()
        if is_last_decoder_layer:
            # pool on last layer Q
            last_token_inds = cu_seqlens[1:] - 1
            query_states = query_states[last_token_inds]
            cu_seqlens_q = torch.arange(
                query_states.size(0) + 1, device=query_states.device, dtype=torch.int32
            )
            max_seqlen_q = 1
            q_len = query_states.size(0)
            causal = False
        else:
            cu_seqlens_q = cu_seqlens
            max_seqlen_q = max_seqlen
            causal = True

        dropout_rate = self.attention_dropout if self.training else 0.0
        if self.sliding_window:
            window_size = (self.sliding_window, self.sliding_window)
        else:
            window_size = (-1, -1)
        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout_rate,
            softmax_scale=self.scaling,
            causal=causal,
            window_size=window_size,
            softcap=self.config.attn_logit_softcapping,
        )
        attn_output = attn_output.reshape(q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output


class Gemma2DecoderLayer(nn.Module):
    def __init__(self, config: Gemma2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size

        assert config._attn_implementation == "flash_attention_2"
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

    def forward(
        self,
        hidden_states,
        cu_seqlens,
        rotary_emb,
        is_last_decoder_layer=False,
    ):
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states,
            cu_seqlens,
            rotary_emb,
            is_last_decoder_layer=is_last_decoder_layer,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        if is_last_decoder_layer:
            last_token_inds = cu_seqlens[1:] - 1
            residual = residual[last_token_inds]
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


GEMMA2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Gemma2Config`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Gemma2 Model outputting raw hidden-states without any specific head on top.",
    GEMMA2_START_DOCSTRING,
)
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


_CONFIG_FOR_DOC = "Gemma2Config"


GEMMA2_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).

            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.

            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.

            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.

            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.

            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
            Indices depicting the position of the input sequence tokens in the sequence. Contrarily to `position_ids`,
            this tensor is not affected by padding. It is used to update the cache in the correct position and to infer
            the complete sequence length.
"""


@add_start_docstrings(
    "The bare Gemma2 Model outputting raw hidden-states without any specific head on top.",
    GEMMA2_START_DOCSTRING,
)
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

        dim = config.head_dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        seq = torch.arange(config.max_position_embeddings, dtype=inv_freq.dtype)
        freqs = torch.einsum("i , j -> i j", seq, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        emb = emb.reshape(emb.size(0), 1, 1, emb.size(1))
        self.register_buffer("rotary_emb", emb)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(GEMMA2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        cu_seqlens,
    ):
        assert input_ids.size(0) == 1
        input_ids = input_ids.squeeze(0)
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

        for idx, decoder_layer in enumerate(self.layers):
            is_last_decoder_layer = idx == len(self.layers) - 1
            layer_outputs = decoder_layer(
                hidden_states,
                cu_seqlens,
                self.rotary_emb,
                is_last_decoder_layer=is_last_decoder_layer,
            )

            hidden_states = layer_outputs

        hidden_states = self.norm(hidden_states)

        return hidden_states

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if past_key_values is not None:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if attention_mask is not None
                else input_tensor.shape[1]
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError(
                    "Custom 4D attention mask should be passed in inverted form with max==0`"
                )
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(
                target_length, device=device
            ) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(
                input_tensor.shape[0], 1, -1, -1
            )
            if attention_mask is not None:
                causal_mask = (
                    causal_mask.clone()
                )  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = (
                    causal_mask[:, :, :, :mask_length]
                    + attention_mask[:, None, None, :]
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[
                    :, :, :, :mask_length
                ].masked_fill(padding_mask, min_dtype)
        return causal_mask


@add_start_docstrings(
    """
    The Gemma2 Model transformer with a sequence classification head on top (linear layer).

    [`Gemma2ForSequenceClassification`] uses the last token in order to do the classification, as other causal models
    (e.g. GPT-2) do.

    Since it does classification on the last token, it requires to know the position of the last token. If a
    `pad_token_id` is defined in the configuration, it finds the last token that is not a padding token in each row. If
    no `pad_token_id` is defined, it simply takes the last value in each row of the batch. Since it cannot guess the
    padding tokens when `inputs_embeds` are passed instead of `input_ids`, it does the same (take the last value in
    each row of the batch).
    """,
    GEMMA2_START_DOCSTRING,
)
class Gemma2ForSequenceClassification(Gemma2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma2Model(config)
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

    @add_start_docstrings_to_model_forward(GEMMA2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        cu_seqlens,
    ):
        transformer_outputs = self.model(
            input_ids,
            cu_seqlens,
        )
        hidden_states = transformer_outputs
        # last_token_inds = cu_seqlens[1:] - 1
        # hidden_states = hidden_states[0][last_token_inds]
        # hidden_states = hidden_states[0]
        logits = self.score(hidden_states)
        logits = logits.float()
        return logits
