# -*- coding: utf-8 -*-
# @Time    : 2023/6/5 18:55
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : modeling_bloom.py
# @Software: CleanTransformer
# @Description: modeling_bloom

import math
import torch
from torch.nn import CrossEntropyLoss
from CleanTransformer.transformer import LayerNorm
from CleanTransformer.generation.generation_util import GenerationMixin


class BloomConfig():
    def __init__(
            self,
            vocab_size=250880,
            hidden_size=64,
            n_layer=2,
            num_attention_heads=8,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            use_cache=True,
            bos_token_id=1,
            eos_token_id=2,
            apply_residual_connection_post_layernorm=False,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            pretraining_tp=1,  # TP rank used when training with megatron
            slow_but_exact=False,
            **kwargs,
    ):
        self.vocab_size = vocab_size
        # Backward compatibility with n_embed kwarg
        n_embed = kwargs.pop("n_embed", None)
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = self.num_attention_heads = num_attention_heads
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout

        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.slow_but_exact = slow_but_exact

        self.num_hidden_layers = self.n_layer


class BloomAttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact

        self.hidden_size = config.hidden_size
        self.num_heads = config.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.hidden_dropout = config.hidden_dropout

        self.inv_norm_factor = 1.0 / math.sqrt(self.head_dim)
        self.beta = 1.0

        self.query_key_value = torch.nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.attention_dropout = torch.nn.Dropout(config.attention_dropout)

    def forward(self, hidden_states, residual, alibi, k_v_past=None,
                attention_mask=None,
                head_mask=None):
        hidden_states = self.query_key_value(hidden_states)
        bsz, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(bsz, seq_len, self.num_heads, 3, self.head_dim)
        q, k, v = hidden_states[..., 0, :], hidden_states[..., 1, :], hidden_states[..., 2, :]

        q = q.transpose(1, 2)  # .reshape(bsz * self.num_heads, seq_len, self.head_dim)
        k = k.transpose(1, 2)  # .reshape(bsz * self.num_heads, self.head_dim, seq_len)
        v = v.transpose(1, 2)  # .reshape(bsz * self.num_heads, seq_len, self.head_dim)

        if k_v_past is not None:
            past_k, past_v = k_v_past
            k = torch.concat((past_k, k), dim=-2)
            v = torch.concat((past_v, v), dim=-2)
        k_v_past = (k, v)

        q = q.reshape(bsz * self.num_heads, seq_len, self.head_dim)
        extend_seq_len = v.shape[-2]
        k = k.transpose(2, 3).reshape(bsz * self.num_heads, self.head_dim, extend_seq_len)
        v = v.reshape(bsz * self.num_heads, extend_seq_len, self.head_dim)

        weight = alibi.baddbmm(
            batch1=q,
            batch2=k,
            beta=self.beta,
            alpha=self.inv_norm_factor,
        )

        if weight.dtype == torch.float16:
            weight = weight.to(torch.float)
        weight = torch.masked_fill(weight.view(bsz, self.num_heads, seq_len, -1), attention_mask,
                                   torch.finfo(weight.dtype).min)
        weight = torch.softmax(weight, dim=-1)
        weight = self.attention_dropout(weight)
        if head_mask:
            weight = weight * head_mask
        v = torch.matmul(weight.view(bsz * self.num_heads, seq_len, -1), v)

        v = v.view(bsz, self.num_heads, seq_len, self.head_dim).transpose(1, 2).contiguous().view(bsz, seq_len, self.num_heads * self.head_dim)

        if self.pretraining_tp > 1 and self.slow_but_exact:
            raise Exception("pretraining_tp and slow_but_exact not supported yet")
        else:
            v = self.dense(v)
        v = residual + torch.nn.functional.dropout(v, p=self.hidden_dropout, training=self.training)

        return v, k_v_past


class BloomBlock(torch.nn.Module):
    def __init__(self, config):
        super(BloomBlock, self).__init__()
        hidden_size = config.hidden_size

        self.input_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        self.num_heads = config.n_head
        self.self_attention = BloomAttentionLayer(config)
        self.post_attention_layernorm = LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        self.mlp = BloomMLP(config)

        self.apply_residual_connection_post_layernorm = config.apply_residual_connection_post_layernorm
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states, attention_mask, alibi, head_mask, k_v_past=None):
        layernorm_output = self.input_layernorm(hidden_states)

        residual = layernorm_output if self.apply_residual_connection_post_layernorm else hidden_states

        attention_output, k_v_past = self.self_attention(
            layernorm_output,
            residual,
            attention_mask=attention_mask,
            alibi=alibi,
            head_mask=head_mask,
            k_v_past=k_v_past
        )
        layernorm_output = self.post_attention_layernorm(attention_output)

        residual = layernorm_output if self.apply_residual_connection_post_layernorm else attention_output
        output = self.mlp(layernorm_output, residual)
        return output, k_v_past


class BloomModel(torch.nn.Module):
    def __init__(self, config):
        super(BloomModel, self).__init__()
        self.config = config
        self.num_heads = config.n_head
        self.embed_dim = config.hidden_size

        self.word_embeddings = torch.nn.Embedding(config.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

        self.blocks = torch.nn.ModuleList([BloomBlock(config) for _ in range(config.num_hidden_layers)])

        self.ln_f = LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)

    def _attn_mask(self, attention_mask, input_shape, k_v_past_length=0):
        bsz, seq_len = input_shape
        attention_mask = attention_mask[:, None, None, :].expand(bsz, 1, seq_len, attention_mask.shape[-1]).to(torch.bool)
        attention_mask = ~attention_mask
        if input_shape[1] > 1:
            bias = torch.tril(torch.ones(seq_len, seq_len))[None, None, :, :].expand(bsz, 1, seq_len, seq_len).to(torch.bool)
            bias = ~bias.to(attention_mask.device)
            return attention_mask | bias
        else:
            return attention_mask

    def forward(self, input_ids, attention_mask, head_mask, k_v_pasts=None):
        if k_v_pasts is None:
            k_v_pasts = [None] * self.config.n_layer
        input_embeds = self.word_embeddings(input_ids)
        hidden_states = self.word_embeddings_layernorm(input_embeds)

        alibi = build_alibi_tensor(attention_mask, self.num_heads, dtype=hidden_states.dtype)

        attention_mask = self._attn_mask(attention_mask, input_shape=input_ids.shape,
                                         k_v_past_length=0 if k_v_pasts[0] is None else k_v_pasts[0][0].shape[2])

        for i, block in enumerate(self.blocks):
            hidden_states, k_v_pasts[i] = block(hidden_states,
                                                attention_mask=attention_mask,
                                                alibi=alibi,
                                                head_mask=head_mask if head_mask else None,
                                                k_v_past=k_v_pasts[i])

        return self.ln_f(hidden_states), k_v_pasts


class BloomForCausalLM(torch.nn.Module, GenerationMixin):
    def __init__(self, config: BloomConfig):
        super(BloomForCausalLM, self).__init__()
        self.config = config
        self.bloom = BloomModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def _tie_weight(self):
        self.lm_head.weight = self.bloom.word_embeddings.weight

    def forward(self, input_ids, attention_mask=None, head_mask=None, k_v_pasts=None, labels=None, **kwargs):
        hidden_states, k_v_pasts = self.bloom(input_ids, attention_mask, head_mask, k_v_pasts)
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits, hidden_states)

        if labels is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            bsz, seq_len, vocab_size = shift_logits.shape

            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(bsz * seq_len, vocab_size),
                            shift_labels.view(bsz * seq_len))
            outputs = (loss,) + outputs
        return outputs, k_v_pasts


def _cite(info):
    def decorator(cls):
        return cls

    return decorator


@_cite(info='以下代码来源于transformers 4.26.0')
class BloomMLP(torch.nn.Module):
    def __init__(self, config: BloomConfig):
        super().__init__()
        hidden_size = config.hidden_size

        self.pretraining_tp = config.pretraining_tp
        self.slow_but_exact = config.slow_but_exact
        self.dense_h_to_4h = torch.nn.Linear(hidden_size, 4 * hidden_size)
        self.gelu_impl = BloomGelu()
        self.dense_4h_to_h = torch.nn.Linear(4 * hidden_size, hidden_size)
        self.hidden_dropout = config.hidden_dropout

    def forward(self, hidden_states: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        hidden_states = self.gelu_impl(self.dense_h_to_4h(hidden_states))

        if self.pretraining_tp > 1 and self.slow_but_exact:
            intermediate_output = torch.zeros_like(residual)
            slices = self.dense_4h_to_h.weight.shape[-1] / self.pretraining_tp
            for i in range(self.pretraining_tp):
                intermediate_output = intermediate_output + torch.nn.functional.linear(
                    hidden_states[:, :, int(i * slices): int((i + 1) * slices)],
                    self.dense_4h_to_h.weight[:, int(i * slices): int((i + 1) * slices)],
                )
        else:
            intermediate_output = self.dense_4h_to_h(hidden_states)

        output = residual + torch.nn.functional.dropout(intermediate_output, p=self.hidden_dropout, training=self.training)

        return output


@_cite(info='以下代码来源于transformers 4.26.0')
class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return bloom_gelu_forward(input)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input = ctx.saved_tensors
        tmp = bloom_gelu_back(grad_output, input)
        return tmp


@_cite(info='以下代码来源于transformers 4.26.0')
class BloomGelu(torch.nn.Module):
    """
    BloomBiasGelu wrapper function that make use of the simple function on inference mode to make the model
    torchscriptable and use the autograd function in training mode to get the accurate results of the gradients Partly
    copied from Megatron-DeepSpeed code and adapted for our needs

    See here why autograd functions are not torchscriptable: https://github.com/pytorch/pytorch/issues/22329
    """

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            return GeLUFunction.apply(x)
        else:
            return bloom_gelu_forward(x)


@_cite(info='以下代码来源于transformers 4.26.0')
def build_alibi_tensor(attention_mask, num_heads, dtype):
    batch_size, seq_length = attention_mask.shape

    ############### 给每个head计算不同的slope ###############
    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
    base = torch.tensor(
        2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
    )
    powers = torch.arange(1, 1 + closest_power_of_2, device=attention_mask.device, dtype=torch.int32)
    slopes = torch.pow(base, powers)

    if closest_power_of_2 != num_heads:
        extra_base = torch.tensor(
            2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3))), device=attention_mask.device, dtype=torch.float32
        )
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = torch.arange(1, 1 + 2 * num_remaining_heads, 2, device=attention_mask.device, dtype=torch.int32)
        slopes = torch.cat([slopes, torch.pow(extra_base, extra_powers)], dim=0)

    ############### 将slope和按相对距离变化的预设值相乘 ###############
    arange_tensor = ((attention_mask.cumsum(dim=-1) - 1) * attention_mask)[:, None, :]
    alibi = slopes[..., None] * arange_tensor
    return alibi.reshape(batch_size * num_heads, 1, seq_length).to(dtype)


@_cite(info='以下代码来源于transformers 4.26.0')
def bloom_gelu_forward(x: torch.Tensor) -> torch.Tensor:
    """
    Custom bias GELU function. Adapted from Megatron-DeepSpeed code. Here we use a simple implementation (inference) to
    make the model jitable.

    Args:
        x (`torch.tensor`, *required*):
            input hidden states
    """
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))


@_cite(info='以下代码来源于transformers 4.26.0')
def bloom_gelu_back(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    gradient of tanh approximation of gelu gradient of actual gelu is: 0.5 * (1. + torch.erf(x * 0.70710678)) +
    0.3989423 * x * torch.exp(-0.5 * x * x)

    Args:
        g (`torch.tensor`, *required*):
            gradient output tensor
        x (`torch.tensor`, *required*):
            input tensor
    """
    x = x[0]  # x is a tuple of 1 element, needs to unpack it first
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)
    return ff * g


if __name__ == "__main__":
    g = bloom_gelu_back()
