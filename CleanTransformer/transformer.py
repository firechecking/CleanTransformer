# -*- coding: utf-8 -*-
# @Time    : 2023/4/23 1:29 PM
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : transformer.py
# @Software: CleanTransformer
# @Description: transformer

import math, torch


class AttentionLayer(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0
        self.dim, self.m_head = config.hidden_size, config.num_attention_heads

        self.q_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.k_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.v_linear = torch.nn.Linear(config.hidden_size, config.hidden_size)

        self.dropout = torch.nn.Dropout(config.attention_probs_dropout_prob)

    def _split_m_head(self, x):
        b, s, d = x.shape
        x = x.view(b, s, self.m_head, -1)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,
                attention_mask=None,
                head_mask=None):
        """
        q.shape = k.shape = v.shape =(bs, seq, dim)
        """
        ############### 获取q、k、v ###############
        q, k, v = self.q_linear(hidden_states), self.k_linear(hidden_states), self.v_linear(hidden_states)
        q, k, v = self._split_m_head(q), self._split_m_head(k), self._split_m_head(v)

        ############### 获取weight ###############
        weight = torch.matmul(q, k.transpose(2, 3))
        weight = weight / math.sqrt(self.dim / self.m_head)
        if attention_mask is not None:
            # attention_mask之所以用+，是因为如果想要v中某一个元素的weight趋近于0，则给对应weight加上一个负的极大值后，再下一步进行softmax后就会变成趋近于0
            weight = weight + attention_mask
        weight = torch.softmax(weight, dim=-1)
        weight = self.dropout(weight)
        if head_mask:
            # head_mask直接作用在n_head上，让某一个head上的weight为0。因为是在softmax之后，所以不用+，而是用*
            weight = weight * head_mask

        ############### 计算v ###############
        v = torch.matmul(weight, v)

        ############### 对v进行reshape并返回 ###############
        b, h, s, hd = v.shape
        v = v.transpose(1, 2).contiguous().view(b, s, h * hd)
        return v


class LayerNorm(torch.nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super(LayerNorm, self).__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)

        self.normalized_shape, self.eps = normalized_shape, eps
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))

    def _mean(self, x):
        _shape = list(x.shape[:-len(self.normalized_shape)]) + [-1]
        _x = x.view(*_shape)
        mean = torch.sum(_x, dim=-1) / _x.shape[-1]
        for i in range(len(x.shape) - len(mean.shape)):
            mean = mean.unsqueeze(-1)
        return mean

    def forward(self, x):
        '''
        参考论文 https://arxiv.org/abs/1607.06450
        参考链接 https://blog.csdn.net/xinjieyuan/article/details/109587913
        pytorch文档 https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        :param x: shape=(bs, seq_len, dim)
        '''
        mean = self._mean(x)
        std = self._mean((x - mean).pow(2) + self.eps).pow(0.5)
        x = (x - mean) / std
        return self.weight * x + self.bias


class TransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.attention = AttentionLayer(config)

        self.ffw = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size * 4),
            torch.nn.ReLU(),
            torch.nn.Linear(config.hidden_size * 4, config.hidden_size)
        )
        self.norm1 = LayerNorm(config.hidden_size, config.layer_norm_epsilong)
        self.norm2 = LayerNorm(config.hidden_size, config.layer_norm_epsilong)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        '''
        :param x: shape=(bs, seq_len, dim)
        '''
        ############### attention + Add + Norm ###############
        att_out = self.attention(x)
        att_out = self.dropout(att_out)
        add_norm_out = self.norm1(x + att_out)

        ############### FFW + Add + Norm ###############
        ffw_out = self.ffw(add_norm_out)
        ffw_out = self.dropout(ffw_out)
        add_norm_out = self.norm2(add_norm_out + ffw_out)

        return add_norm_out


class ExampleConfig():
    def __init__(self):
        self.num_attention_heads = 3
        self.layer_norm_epsilong = 1e-5
        self.resid_pdrop = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.hidden_size = 12
        self.hidden_dropout_prob = 0.1


def layernorm_sample():
    torch.manual_seed(999)
    x = torch.rand((3, 4, 6))
    normalized_shape = [4, 6]
    norm1 = LayerNorm(normalized_shape)
    norm2 = torch.nn.LayerNorm(normalized_shape)
    print(norm1(x))
    print(norm2(x))


def t_TransformerBlock():
    torch.manual_seed(999)
    config = ExampleConfig()
    trans = TransformerBlock(config)
    q = torch.rand((3, 4, config.hidden_size))
    r = trans(q)
    print(q)
    print(r)


if __name__ == "__main__":
    layernorm_sample()
    t_TransformerBlock()
