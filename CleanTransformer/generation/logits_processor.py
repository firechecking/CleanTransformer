# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 21:29
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : logits_processor.py
# @Software: CleanTransformer
# @Description: logits_processor

import torch

class NoRepeatNGramLogitsProcessor():
    def __init__(self, ngram_size):
        self.ngram_size = ngram_size

    def __call__(self, input_ids, scores):
        bsz, cur_len = input_ids.shape

        for i in range(bsz):
            gen_tokens = input_ids[i].tolist()

            ############### 统计上文已经出现过的ngram组合 ###############
            generated_ngrams = {}
            for j in range(cur_len - self.ngram_size + 1):
                ngram_tuple = tuple(gen_tokens[j:j + self.ngram_size])
                generated_ngrams[ngram_tuple[:-1]] = generated_ngrams.get(ngram_tuple[:-1], []) + [ngram_tuple[-1]]

            ############### 将已出现组合的token分数设为负无穷 ###############
            ngram_tuple = tuple(gen_tokens[-self.ngram_size + 1:])
            banned_tokens = generated_ngrams.get(ngram_tuple, [])
            scores[i, banned_tokens] = -float("inf")

        return scores


class TemperatureLogitsWrapper():
    def __init__(self, temperature):
        self.temperature = max(temperature, 1e-2)

    def __call__(self, input_ids, scores, *args, **kwargs):
        scores = scores / self.temperature
        return scores


class TopKLogitsWrapper():
    def __init__(self, top_k, filter_value=-float('Inf'), min_tokens_to_keep=1):
        self.top_k = int(max(top_k, min_tokens_to_keep, 1))
        self.filter_value = filter_value

    def __call__(self, input_ids, scores, *args, **kwargs):
        top_k = min(self.top_k, scores.size(-1))

        ############### 选出top_k的分数，并将低于这个分数的值设为负无穷 ###############
        topk_scores, topk_indices = scores.topk(top_k, dim=-1, largest=True, sorted=True)
        indices_to_remove = scores < topk_scores[..., -1, None]
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class TopPLogitsWrapper():
    def __init__(self, top_p, filter_value=-float('Inf'), min_tokens_to_keep=1):
        self.top_p = max(min(top_p, 1.0), 0)
        self.filter_value = filter_value
        self.min_tokens_to_keep = max(1, min_tokens_to_keep)

    def __call__(self, input_ids, scores, *args, **kwargs):
        ############### 排序后，使用softmax计算概率，再使用cumsum计算累积概率 ###############
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        ############### 将累积概率<1-p的值标记为删除 ###############
        sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)

        ############### 至少确保前min_tokens_to_keep个不删除 ###############
        sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0

        ############### 根据移除indices，调整原始score ###############
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


if __name__ == "__main__":
    pass
