# -*- coding: utf-8 -*-
# @Time    : 2023/4/24 9:04 PM
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : modeling_bert.py
# @Software: CleanTransformer
# @Description: modeling_bert

import unicodedata
import torch
from collections import OrderedDict

from CleanTransformer.tokenizers import WordPieceTokenizer
from CleanTransformer.transformer import LayerNorm, AttentionLayer


class BertConfig():
    def __init__(self,
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 **kwargs):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        for k, v in kwargs.items():
            setattr(self, k, v)


class BasicTokenizer():
    def __init__(self, do_lower_case=False, never_split=None, tokenize_chinese_chars=True, strip_accents=True):
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split) if never_split else set()
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.strip_accents = strip_accents

    def tokenize(self, text, never_split=None):
        never_split = self.never_split.union(set(never_split)) if never_split else self.never_split
        text = self._clean_text(text)

        if self.tokenize_chinese_chars:
            text = self._tokenize_chinese_chars(text)

        orig_tokens = text.strip().split()
        split_tokens = []
        for token in orig_tokens:
            if token not in never_split:  # bug?: 对于"hello world[SEP]too"这种连在一起的情况，无法分出来
                if self.do_lower_case:
                    token = token.lower()
                if self.strip_accents:
                    token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token, never_split))
        output_tokens = " ".join(split_tokens).strip().split()
        return output_tokens

    def _run_split_on_punc(self, text, never_split):
        """Splits punctuation on a piece of text."""
        if text in never_split:
            return [text]
        output = [[]]
        for char in list(text):
            if self._is_punctuation(char):
                output.append([char])
                output.append([])
            else:
                output[-1].append(char)
        return ["".join(x) for x in output]

    def _is_punctuation(self, char):
        # We treat all non-letter/number ASCII as punctuation.
        cp = ord(char)
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        if (cp >= 0x4E00 and cp <= 0x9FFF) or (cp >= 0x3400 and cp <= 0x4DBF) or \
                (cp >= 0x20000 and cp <= 0x2A6DF) or (cp >= 0x2A700 and cp <= 0x2B73F) or \
                (cp >= 0x2B740 and cp <= 0x2B81F) or (cp >= 0x2B820 and cp <= 0x2CEAF) or \
                (cp >= 0xF900 and cp <= 0xFAFF) or (cp >= 0x2F800 and cp <= 0x2FA1F):
            return True
        return False

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            output.append(" {} ".format(char) if self._is_chinese_char(cp) else char)
        return "".join(output)

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xFFFD or self._is_control(char):
                continue
            output.append(" " if self._is_whitespace(char) else char)
        return "".join(output)

    def _is_control(self, char):
        if char == "\t" or char == "\n" or char == "\r":  # treat them as whitespace
            return False
        cat = unicodedata.category(char)
        if cat.startswith("C"):
            return True
        return False

    def _is_whitespace(self, char):
        # \t, \n, and \r are technically control characters but we treat them as whitespace
        if char == " " or char == "\t" or char == "\n" or char == "\r":
            return True
        cat = unicodedata.category(char)
        if cat == "Zs":
            return True
        return False


class BertTokenizer():
    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, tokenizer_chinese_chars=True):
        self.special_tokens = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
        self.unk, self.sep, self.pad, self.cls, self.mask = self.special_tokens
        self.do_basic_tokenize = do_basic_tokenize
        self.vocab = self._load_vocab(vocab_file)
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case, self.special_tokens, tokenizer_chinese_chars)
        self.wordpiece_tokenizer = WordPieceTokenizer(vocab_size=len(self.vocab), lowercase=do_lower_case,
                                                      basic_tokenizer=lambda x: x.strip().split(),
                                                      unk=self.unk, sep=self.sep, pad=self.pad, cls=self.cls, mask=self.mask)
        self.wordpiece_tokenizer.load(vocab=self.vocab)

    def _load_vocab(self, vocab_file):
        vocab = OrderedDict()
        for idx, token in enumerate(open(vocab_file, 'r').readlines()):
            vocab[token.rstrip('\n')] = idx
        return vocab

    def tokenize(self, text):
        tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text, never_split=self.special_tokens):
                if token in self.special_tokens:
                    tokens.append(token)
                else:
                    tokens.extend(self.wordpiece_tokenizer.tokenize(token, add_pre=None, add_mid="##", add_post=None))
        else:
            tokens = self.wordpiece_tokenizer.tokenize(text, add_pre=None, add_mid="##", add_post=None)
        return tokens

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens, ]
        return [self.vocab.get(token, self.vocab.get(self.unk)) for token in tokens]

    def encode_plus(self, text, text_pair=None, max_len=1024, padding=True, truncation=True, truncation_side='right'):
        '''
        返回input_ids, segment_ids, attention_mask
        padding: True, 将text、text_pair拼接后再pad到max_len长度
        trunction：True, 将text、text_pair拼接后如果超过max_len，按longest_first策略进行trunc
        '''

        ############### tokenizer + tokens_to_ids ###############
        text_ids = self.convert_tokens_to_ids(self.tokenize(text))
        text_pair_ids = self.convert_tokens_to_ids(self.tokenize(text_pair)) if text_pair else []  # 在bert中支持输入2个text

        ############### trunction ###############
        ids_len = len(text_ids) + len(text_pair_ids) + 3 if text_pair_ids else len(text_ids) + 2
        if truncation and ids_len > max_len:
            for _ in range(ids_len - max_len):
                if len(text_ids) > len(text_pair_ids):  # TODO: 其他trunc策略
                    text_ids = text_ids[:-1] if truncation_side == 'right' else text_ids[1:]
                else:
                    text_pair_ids = text_pair_ids[:-1] if truncation_side == 'right' else text_pair_ids[1:]

        ############### 定制bert输入格式 ###############
        # 1个text，组织成[cls] text1 [sep]
        # 2个text，则组织成[cls] text1 [sep] text2 [sep]
        input_ids = self.convert_tokens_to_ids([self.cls]) + text_ids + self.convert_tokens_to_ids([self.sep])
        segment_ids = [0] * len(input_ids)
        attention_mask = [1] * len(input_ids)
        if text_pair_ids:
            input_ids += text_pair_ids + self.convert_tokens_to_ids([self.sep])
            segment_ids += [1] * (len(text_pair_ids) + 1)
            attention_mask += [1] * (len(text_pair_ids) + 1)

        ############### padding ###############
        while padding and len(input_ids) < max_len:
            input_ids += self.convert_tokens_to_ids(self.pad)
            segment_ids += [0]
            attention_mask += [0]
        return {"input_ids": input_ids, "segment_ids": segment_ids, "attention_mask": attention_mask}


ACT2FN = {"gelu": torch.nn.GELU, "relu": torch.nn.ReLU}


class BertTransformerBlock(torch.nn.Module):
    def __init__(self, config):
        super(BertTransformerBlock, self).__init__()
        self.config = config
        self.attention = AttentionLayer(config)

        self.attention_post = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.Dropout(config.hidden_dropout_prob)
        )
        self.norm1 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ffw = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.intermediate_size),
            ACT2FN[config.hidden_act](),
            torch.nn.Linear(config.intermediate_size, config.hidden_size),
        )
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.norm2 = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None):
        '''
        :param hidden_states: shape=(bs, seq_len, dim)
        '''
        ############### attention + linear + layernorm ###############
        attention_outputs = self.attention(hidden_states, attention_mask)
        attention_outputs = self.attention_post(attention_outputs)
        attention_outputs = self.norm1(attention_outputs + hidden_states)

        ############### feedforward + layernorm ###############
        ffw_outputs = self.ffw(attention_outputs)
        drop_outputs = self.dropout(ffw_outputs)
        norm_outputs = self.norm2(drop_outputs + attention_outputs)
        return norm_outputs


class BertModel(torch.nn.Module):
    def __init__(self, config):
        super(BertModel, self).__init__()
        self.config = config

        ############### embeddings ###############
        self.word_embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.segment_embeddings = torch.nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.embedding_post = torch.nn.Sequential(
            LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
            torch.nn.Dropout(config.hidden_dropout_prob)
        )
        ############### transformer blocks ###############
        self.blocks = torch.nn.ModuleList([BertTransformerBlock(config) for _ in range(config.num_hidden_layers)])

        ############### output ###############
        self.pooler = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.Tanh()
        )

    def forward(self,
                input_ids=None,
                attention_mask=None,
                segment_ids=None,
                position_ids=None):
        if position_ids == None:  # 默认按顺序的position_id
            position_ids = torch.arange(input_ids.shape[1], dtype=torch.long)
        # embeddings
        input_embed = self.word_embeddings(input_ids)
        segment_embed = self.segment_embeddings(segment_ids)
        position_embed = self.position_embeddings(position_ids)
        hidden_states = self.embedding_post(input_embed + segment_embed + position_embed)

        if attention_mask is not None:  # attention_mask的计算在softmax之前，所以对于需要mask的位置，赋一个负的极大值
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * -10000.0
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)

        # Bert取[cls] token处理分类任务
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.pooler(first_token_tensor)

        return (hidden_states, pooled_output)


class BertForSequenceClassification(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.bert = BertModel(config)
        self.drop = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, config.num_labels)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                segment_ids=None,
                position_ids=None, ):
        hidden_states, pooled_output = self.bert(input_ids, attention_mask, segment_ids, position_ids)
        pooled_output = self.drop(pooled_output)
        logits = self.classifier(pooled_output)
        # TODO: training loss
        return logits


def sample_BertTokenizer():
    text = "“五一”小长假临近，30岁的武汉市民万昕在文旅博览会上获得了一些制定5天旅游计划的新思路。“‘壮美广西’‘安逸四川’，还有‘有一种叫云南的生活’这些展馆标识都很新颖，令人心向往之。”万昕说，感到身边越来越多的人走出家门去旅游。"
    text = 'Say that thou didst forsake me for some fault, And I will comment upon that offence; Speak of my lameness, and I straight will halt, Against thy reasons making no defence.'

    tokenizer = BertTokenizer(vocab_file='../../checkpoints/bert-base-uncased/vocab.txt')
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    plus = tokenizer.encode_plus(text, text_pair=text, max_len=100, padding=True)
    print("=" * 50, "自定义")
    print(tokens)
    print(ids)
    print(plus)

    from transformers import BertTokenizer as OfficialBertTokenizer
    official_tokenizer = OfficialBertTokenizer(vocab_file='../../checkpoints/bert-base-uncased/vocab.txt')
    o_tokens = official_tokenizer.tokenize(text)
    o_ids = official_tokenizer.convert_tokens_to_ids(o_tokens)
    o_plus = official_tokenizer.encode_plus(text, text_pair=text, max_length=100, padding='max_length', truncation='longest_first')
    print('=' * 50 + 'huggingface')
    print(o_tokens)
    print(o_ids)
    print(o_plus)

    assert tokens == o_tokens
    assert ids == o_ids
    assert plus['input_ids'] == o_plus['input_ids']
    assert plus['segment_ids'] == o_plus['token_type_ids']
    assert plus['attention_mask'] == o_plus['attention_mask']

    print('=' * 50 + 'special tokens')
    tokens = ['[UNK]', '[SEP]', '[PAD]', '[CLS]', '[MASK]']
    print(tokenizer.convert_tokens_to_ids(tokens))
    print(official_tokenizer.convert_tokens_to_ids(tokens))


if __name__ == "__main__":
    sample_BertTokenizer()
