# -*- coding: utf-8 -*-
# @Time    : 2023/4/26 7:12 PM
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : modeling_gpt.py
# @Software: CleanTransformer
# @Description: modeling_gpt

import math, torch
from CleanTransformer.transformer import LayerNorm
from CleanTransformer.generation.generation_util import GenerationMixin


class GPTConfig():
    def __init__(self, vocab_size=100, n_embd=100, n_positions=100, n_layer=3, n_head=2, n_ctx=2000,
                 embd_pdrop=0.1, attn_pdrop=0.1, resid_pdrop=0.1, layer_norm_epsilon=1e-5,
                 afn='gelu_new',
                 **kwargs):
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.n_positions = n_positions
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.embd_pdrop, self.attn_pdrop, self.resid_pdrop = embd_pdrop, attn_pdrop, resid_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.afn = afn
        for k, v in kwargs.items():
            setattr(self, k, v)


class Conv1D(torch.nn.Module):
    '''
    本质上是一个普通的Linear
    '''

    def __init__(self, out_dim, input_dim):
        super(Conv1D, self).__init__()

        w = torch.empty(input_dim, out_dim)
        torch.nn.init.normal_(w, std=0.02)
        self.weight = torch.nn.Parameter(w)
        self.bias = torch.nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight.transpose(0, 1), self.bias)


class AttentionLayer(torch.nn.Module):
    def __init__(self, config, scale=False):
        super().__init__()
        self.config, self.scale = config, scale
        self.n_state, self.n_head, self.n_ctx = config.n_embd, config.n_head, config.n_ctx
        assert self.n_state % self.n_head == 0

        self.register_buffer("bias", torch.tril(torch.ones(self.n_ctx, self.n_ctx)).view(1, 1, self.n_ctx, self.n_ctx))

        self.c_attn = Conv1D(self.n_state * 3, self.n_state)
        self.c_proj = Conv1D(self.n_state, self.n_state)
        self.attn_dropout = torch.nn.Dropout(config.attn_pdrop)
        self.resid_dropout = torch.nn.Dropout(config.resid_pdrop)

    def _split_m_head(self, x):
        b, s, d = x.shape
        x = x.view(b, s, self.n_head, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, k_v_past=None,
                attention_mask=None, head_mask=None):
        ############### 获取q、k、v ###############
        hidden_states = self.c_attn(hidden_states)
        q, k, v = hidden_states.split(self.n_state, dim=-1)
        q, k, v = self._split_m_head(q), self._split_m_head(k), self._split_m_head(v)

        ############### 用于inference时的k、v计算复用 ###############
        if k_v_past is not None:
            past_k, past_v = k_v_past
            k = torch.concat((past_k, k), dim=-2)
            v = torch.concat((past_v, v), dim=-2)
        k_v_past = (k, v)

        ############### 获取weight ###############
        weight = torch.matmul(q, k.transpose(2, 3))
        if self.scale:
            weight = weight / math.sqrt(v.size(-1))

        # 保留weight下三角，让模型只能看到上文
        b = self.bias[:, :, k.size(-2) - q.size(-2): k.size(-2), : k.size(-2)]
        weight = weight * b + -1e4 * (1 - b)

        if attention_mask is not None:
            weight = weight + attention_mask
        weight = torch.softmax(weight, dim=-1)
        weight = self.attn_dropout(weight)
        if head_mask:
            weight = weight * head_mask

        ############### 计算v ###############
        v = torch.matmul(weight, v)

        ############### 对v进行reshape ###############
        b, h, s, hd = v.shape
        v = v.transpose(1, 2).contiguous().view(b, s, h * hd)

        ############### v外面套一层Linear ###############
        v = self.c_proj(v)
        v = self.resid_dropout(v)

        return v, k_v_past


class NewGELUActivation(torch.nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input):
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


ACT2FN = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU, 'gelu_new': NewGELUActivation}


class TransformerBlock(torch.nn.Module):
    def __init__(self, config, scale=False, version='gpt'):
        super(TransformerBlock, self).__init__()
        n_embd = config.n_embd
        self.version = version
        self.attn = AttentionLayer(config, scale)
        self.norm1 = LayerNorm(n_embd, eps=config.layer_norm_epsilon)
        self.mlp = torch.nn.Sequential(
            Conv1D(4 * n_embd, n_embd),
            ACT2FN[config.afn](),
            Conv1D(n_embd, 4 * n_embd),
            torch.nn.Dropout()
        )
        self.norm2 = LayerNorm(n_embd, eps=config.layer_norm_epsilon)

    def forward(self, x, attn_output=None, attention_mask=None, head_mask=None, k_v_past=None):
        if self.version == 'gpt':  # gpt1
            if attn_output is None:
                attn_output, k_v_past = self.attn(x, attention_mask=attention_mask, head_mask=head_mask, k_v_past=k_v_past)
            norm1_output = self.norm1(x + attn_output)
            mlp_output = self.mlp(norm1_output)
            output = self.norm2(norm1_output + mlp_output)
        else:  # gpt2/3
            if attn_output is None:
                attn_output, k_v_past = self.attn(self.norm1(x), attention_mask=attention_mask, head_mask=head_mask, k_v_past=k_v_past)
            x = x + attn_output
            mlp_output = self.mlp(self.norm2(x))
            output = x + mlp_output
        return output, k_v_past


class GPTModel(torch.nn.Module):
    def __init__(self, config, version='gpt'):
        super(GPTModel, self).__init__()
        self.version = version
        self.tokens_embed = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embed = torch.nn.Embedding(config.n_positions, config.n_embd)
        self.drop = torch.nn.Dropout(config.embd_pdrop)
        self.blocks = torch.nn.ModuleList([TransformerBlock(config, scale=True, version=version) for _ in range(config.n_layer)])

        if version != 'gpt':
            self.ln_f = LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, attention_mask=None, position_ids=None, segment_ids=None, k_v_pasts=None):
        input_embeds = self.tokens_embed(input_ids)

        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids[:, -input_embeds.shape[1]:]

        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=input_embeds.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        if k_v_pasts is None:
            k_v_pasts = [None] * len(self.blocks)

        position_embeds = self.position_embed(position_ids)

        segment_embeds = 0 if segment_ids is None else self.tokens_embed(segment_ids.view(-1, segment_ids.size(-1)))

        hidden_states = self.drop(input_embeds + position_embeds + segment_embeds)
        for i, block in enumerate(self.blocks):
            hidden_states, k_v_pasts[i] = block(hidden_states, attention_mask=attention_mask, k_v_past=k_v_pasts[i])

        if self.version == 'gpt':
            return hidden_states, k_v_pasts
        else:
            return self.ln_f(hidden_states), k_v_pasts


class GPTLMHeadModel(torch.nn.Module, GenerationMixin):
    def __init__(self, config, version='gpt'):
        super(GPTLMHeadModel, self).__init__()
        self.version = version
        self.gpt = GPTModel(config, version=version)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self._tie_weights()

    def _tie_weights(self):
        self.lm_head.weight = self.gpt.tokens_embed.weight

    def forward(self, input_ids, attention_mask=None, segment_ids=None, position_ids=None, k_v_pasts=None):
        hidden_states, k_v_pasts = self.gpt(input_ids, attention_mask, position_ids, segment_ids, k_v_pasts)
        lm_logits = self.lm_head(hidden_states)
        outputs = (lm_logits, hidden_states)
        return outputs, k_v_pasts


def sample_conv1d():
    conv = Conv1D(3 * 4, 4)
    x = torch.randn(2, 3, 4)
    o = conv(x)


def sample_attention():
    config = GPTConfig()
    attention = AttentionLayer(config, scale=True)
    x = torch.randn(2, 3, config.n_embd)
    attention(x)


def sample_block():
    config = GPTConfig()
    block = TransformerBlock(GPTConfig(), scale=True)
    x = torch.randn(2, 3, config.n_embd)
    block(x)


def sample_gpt():
    config = GPTConfig()
    gpt = GPTModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 3))
    segment_ids = torch.randint(0, config.vocab_size, (2, 3))
    position_ids = torch.randint(0, config.n_positions, (2, 3))
    gpt(input_ids, position_ids=position_ids, segment_ids=segment_ids)


def sample_gpt_lmhead():
    config = GPTConfig()
    gpt = GPTLMHeadModel(config)
    input_ids = torch.randint(0, config.vocab_size, (2, 3))
    segment_ids = torch.randint(0, config.vocab_size, (2, 3))
    position_ids = torch.randint(0, config.n_positions, (2, 3))
    gpt(input_ids, position_ids=position_ids, segment_ids=segment_ids)


if __name__ == '__main__':
    sample_conv1d()
    sample_attention()
    sample_block()
    sample_gpt()
    sample_gpt_lmhead()
