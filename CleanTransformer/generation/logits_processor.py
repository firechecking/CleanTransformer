# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 21:29
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : logits_processor.py
# @Software: CleanTransformer
# @Description: logits_processor

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


if __name__ == "__main__":
    pass
