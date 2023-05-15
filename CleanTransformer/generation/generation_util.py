# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 00:47
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : generation_util.py
# @Software: CleanTransformer
# @Description: generation_util

import torch
from CleanTransformer.generation.logits_processor import NoRepeatNGramLogitsProcessor, TemperatureLogitsWrapper, TopKLogitsWrapper, TopPLogitsWrapper


class GenerationMixin():
    def __init__(self):
        # 这部分代码不需要执行，因为初始化是在模型代码中声明
        self.gpt = None
        self.version = None

    def generate(self, input_ids, attention_mask=None, position_ids=None, segment_ids=None, generation_configs={}):
        if position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=torch.float32)
            attention_mask = (1.0 - attention_mask) * torch.finfo(attention_mask.dtype).min

        beam_size = generation_configs.get('beam_size', 1)
        max_gen_len = generation_configs.get('max_gen_len', 100)
        end_ids = generation_configs.get('end_ids', None)
        pad_id = generation_configs.get('pad_id', 0)
        no_repeat_ngram_size = generation_configs.get('no_repeat_ngram_size', 0)
        self.do_sample = generation_configs.get('do_sample', True)
        temperature = generation_configs.get('temperature', 1.0)
        top_k = generation_configs.get('top_k', 10)
        top_p = generation_configs.get('top_p', 0.8)

        if isinstance(end_ids, int): end_ids = [end_ids]
        end_ids_tensor = torch.tensor(list(end_ids)).to(input_ids.device) if end_ids is not None else None

        self.logits_processors = []
        if no_repeat_ngram_size > 1:
            self.logits_processors.append(NoRepeatNGramLogitsProcessor(no_repeat_ngram_size))

        self.logits_wrapper = []
        if self.do_sample and temperature != 1.0:
            self.logits_wrapper.append(TemperatureLogitsWrapper(temperature))
        if self.do_sample and top_k > 0:
            self.logits_wrapper.append(TopKLogitsWrapper(top_k, min_tokens_to_keep=1))
        if self.do_sample and top_p < 1.0:
            self.logits_wrapper.append(TopPLogitsWrapper(top_p, min_tokens_to_keep=1))

        if beam_size == 1:
            return self._greedy_search(input_ids, attention_mask, position_ids, segment_ids,
                                       end_ids_tensor, max_gen_len=max_gen_len, pad_id=pad_id)

    def _gen_next_token(self, x, position_ids, segment_ids, attention_mask, k_v_past):
        ############### 计算embedding ###############
        input_embeds = self.gpt.tokens_embed(x)
        position_embeds = self.gpt.position_embed(position_ids)
        segment_embeds = 0 if segment_ids is None else self.gpt.tokens_embed(segment_ids)

        hidden_states = self.gpt.drop(input_embeds + position_embeds + segment_embeds)

        ############### 循环计算transformer的每一个block（复用k_v_past并更新k_v_past） ###############
        for i, block in enumerate(self.gpt.blocks):
            if self.version == 'gpt':
                attn_output, k_v_past[i] = block.attn(hidden_states, layer_past=k_v_past[i], attention_mask=attention_mask, return_kv=True)
            else:
                attn_output, k_v_past[i] = block.attn(block.norm1(hidden_states), layer_past=k_v_past[i], attention_mask=attention_mask,
                                                      return_kv=True)  # gpt2
            hidden_states = block(x=hidden_states, attn_output=attn_output)

        if self.version == 'gpt2':
            hidden_states = self.gpt.ln_f(hidden_states)

        ############### lm_head和token_embedding使用相同的weight ###############
        output_weight = self.gpt.tokens_embed.weight.detach()
        hidden_states = torch.matmul(hidden_states, output_weight.transpose(0, 1))

        return hidden_states

    def _greedy_search(self, input_ids, attention_mask, position_ids, segment_ids, end_ids_tensor, max_gen_len, pad_id):
        bsz = input_ids.size(0)
        max_len = max_gen_len + input_ids.size(-1)
        k_v_past = [None for _ in self.gpt.blocks]
        step = 0
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        while True:
            ############### 计算下一个token的hidden_states （会复用k_v_past并更新k_v_past） ###############
            hidden_states = self._gen_next_token(input_ids[:, step:], position_ids[:, step:],
                                                 None if segment_ids is None else segment_ids[:, step:],
                                                 attention_mask, k_v_past)
            last_token_hidden_states = hidden_states[:, -1, :]

            ############### Logits Penalty ###############
            if len(self.logits_processors) > 0:
                for _processor in self.logits_processors:
                    last_token_hidden_states = _processor(input_ids, last_token_hidden_states)

            if self.do_sample:
                ############### Logits Sampling ###############
                if len(self.logits_wrapper) > 0:
                    for _wrapper in self.logits_wrapper:
                        last_token_hidden_states = _wrapper(input_ids, last_token_hidden_states)
                probs = torch.nn.functional.softmax(last_token_hidden_states, dim=-1)
                step_output = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                ############### 选出得分最高的token ###############
                step_output = torch.argmax(last_token_hidden_states, dim=-1)

            ############### 判断batch的每个case是否生成结束 ###############
            step_output = step_output * unfinished_sequences + pad_id * (1 - unfinished_sequences)
            if end_ids_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    step_output.tile(end_ids_tensor.shape[0], 1).ne(end_ids_tensor.unsqueeze(1)).prod(dim=0)
                )

            ############### 得到最新结果（将作为下一次的输入） ###############
            input_ids = torch.concat([input_ids, step_output[:, None]], dim=-1)
            position_ids = torch.concat([position_ids, (position_ids.max(dim=-1).values + 1).view(-1, 1)], dim=-1)
            segment_ids = None if segment_ids is None else torch.concat([segment_ids, segment_ids[:, -1:]], dim=-1)
            attention_mask = torch.concat([attention_mask, -attention_mask.new_zeros((*attention_mask.shape[:-1], 1))], dim=-1)

            step = input_ids.shape[1] - 1

            ############### 结束条件判断 ###############
            if unfinished_sequences.max() == 0 or step > max_len:
                break

        return input_ids.view(bsz, 1, -1)


if __name__ == "__main__":
    pass
