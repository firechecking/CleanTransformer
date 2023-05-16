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
        early_stop = generation_configs.get('early_stop', True)

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
        else:
            return self._beam_search(input_ids, attention_mask, position_ids, segment_ids,
                                     end_ids_tensor, max_gen_len=max_gen_len, pad_id=pad_id,
                                     beam_size=beam_size, early_stop=early_stop)

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

            ############### 对于batch中已经结束的case,不管新的token是什么，都换成pad_id ###############
            step_output = step_output * unfinished_sequences + pad_id * (1 - unfinished_sequences)
            ############### 判断batch的每个case是否结束 ###############
            if end_ids_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    step_output.tile(end_ids_tensor.shape[0], 1).ne(end_ids_tensor.unsqueeze(1)).prod(dim=0)
                )

            ############### 得到最新结果（将作为下一次的输入） ###############
            input_ids = torch.concat([input_ids, step_output[:, None]], dim=-1)
            position_ids = torch.concat([position_ids, (position_ids.max(dim=-1).values + 1).view(-1, 1)], dim=-1)
            segment_ids = None if segment_ids is None else torch.concat([segment_ids, segment_ids[:, -1:]], dim=-1)
            attention_mask = torch.concat([attention_mask, -attention_mask.new_zeros((*attention_mask.shape[:-1], 1))], dim=-1)

            ############### 结束条件判断 ###############
            step = input_ids.shape[1] - 1
            if unfinished_sequences.max() == 0 or step > max_len:
                break

        return input_ids.view(bsz, 1, -1)

    def _update_beam_infos(self, beam, generated_beam_infos, input_ids, token_indices, next_tokens, probs, end_ids_tensor,
                           pad_token_id,
                           length_penalty=1.0,
                           early_stop=True):
        bsz = next_tokens.shape[0]
        device = input_ids.device
        ############### 保存next_tokens (非end_id)以及来自于哪个beam ###############
        new_indices = torch.zeros((bsz, beam), dtype=token_indices.dtype, device=device)
        new_tokens = torch.zeros((bsz, beam), dtype=next_tokens.dtype, device=device)
        new_probs = torch.zeros((bsz, beam), dtype=probs.dtype, device=device)

        for batch_i in range(bsz):
            candi_generation = generated_beam_infos[batch_i]['candi_generation']
            ############### 如果当前batch_i生成已结束，token替换为pad ###############
            if generated_beam_infos[batch_i]['is_done']:
                new_tokens[batch_i, :] = pad_token_id
                continue

            valid_beam_i = 0
            for beam_i in range(beam):
                if next_tokens[batch_i, beam_i].item() in end_ids_tensor:
                    ############### 对于每个batch_i，首先产生不少于beam_size个候选（每个候选以end_id结尾） ###############
                    if beam_i >= beam: continue  # 在beam_size之后的end_id分数过低，不要
                    choice_idx = beam * batch_i + token_indices[batch_i, beam_i]
                    score = probs[batch_i, beam_i] / (input_ids.shape[-1] ** length_penalty)  # TODO: 这里文本长度是否需要去掉padding？
                    candi_generation.append({"ids": input_ids[choice_idx],
                                             "score": score})
                    ############### 如果候选大于beam_size，则剔除分数最低的候选 ###############
                    if len(candi_generation) > beam:
                        sorted_scores = sorted([(candi['score'], idx) for idx, candi in enumerate(candi_generation)])
                        del candi_generation[sorted_scores[0][1]]
                        generated_beam_infos[batch_i]['worst_score'] = sorted_scores[1][0]
                    else:
                        generated_beam_infos[batch_i]['worst_score'] = min(score, generated_beam_infos[batch_i]['worst_score'])
                else:
                    ############### 没结束前，要尽量保证有beam_size个next_tokens (非end_id)可用于下次输入 ###############
                    new_indices[batch_i, valid_beam_i] = token_indices[batch_i, beam_i]
                    new_tokens[batch_i, valid_beam_i] = next_tokens[batch_i, beam_i]
                    new_probs[batch_i, valid_beam_i] = probs[batch_i, beam_i]
                    valid_beam_i += 1

                if valid_beam_i >= beam:
                    break

            generated_beam_infos[batch_i]['candi_generation'] = candi_generation

            if len(candi_generation) >= beam:
                ############### 结束条件1: 产生beam_size个候选后，且early_stop，则结束 ###############
                if early_stop:
                    generated_beam_infos[batch_i]['is_done'] = True
                    continue
                ############### 结束条件2: 产生beam_size个候选的最低分数，已经比未来可能产生的最大分数更高，则结束 ###############
                next_highest_prob = probs[batch_i].max().item()
                next_highest_score = next_highest_prob / ((input_ids.shape[-1] + 1) ** length_penalty)
                if generated_beam_infos[batch_i]['worst_score'] > next_highest_score:
                    generated_beam_infos[batch_i]['is_done'] = True

        return generated_beam_infos, new_indices, new_tokens, new_probs

    def _beam_topk(self, x_ids, bsz, beam_size, last_token_hidden_states, probs):
        scores = torch.nn.functional.log_softmax(last_token_hidden_states, dim=-1)
        vocab_size = scores.shape[-1]
        probs = probs.view(-1, 1).expand_as(scores)
        scores = scores + probs
        scores = scores.view(bsz, -1)

        probs, next_tokens = scores.topk(2 * beam_size, dim=1, largest=True, sorted=True)

        ############### 确定next_tokens以及来自于哪个beam ###############
        token_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size
        return token_indices, next_tokens, probs

    def _beam_search(self, input_ids, attention_mask, position_ids, segment_ids, end_ids_tensor, max_gen_len, pad_id, beam_size, early_stop):
        bsz = input_ids.size(0)
        max_len = max_gen_len + input_ids.size(-1)
        k_v_past = [None for _ in self.gpt.blocks]
        step = 0

        ############### 将所有输入扩展成beam_size份 ###############
        input_ids = input_ids.repeat_interleave(beam_size, dim=0)
        position_ids = position_ids.repeat_interleave(beam_size, dim=0)
        attention_mask = attention_mask.repeat_interleave(beam_size, dim=0)
        segment_ids = None if segment_ids is None else segment_ids.repeat_interleave(beam_size, dim=0)

        ############### sentence score初始化 ###############
        probs = torch.zeros((bsz, beam_size), device=input_ids.device)
        probs[:, 1:] = -1e9  # 第一次输入时，每个beam都一样，为防止从每个beam中都选出同一个最大token，第一次只从beam 1中选token

        ############### 记录每个case状态 ###############
        generated_beam_infos = [{'is_done': False, 'worst_score': 1e9, 'candi_generation': []} for _ in range(bsz)]

        while True:
            hidden_states = self._gen_next_token(input_ids[:, step:], position_ids[:, step:],
                                                 None if segment_ids is None else segment_ids[:, step:],
                                                 attention_mask, k_v_past)
            last_token_hidden_states = hidden_states[:, -1, :]

            ############### 获取top_k的next_tokens ###############
            token_indices, step_output, probs = self._beam_topk(input_ids, bsz, beam_size, last_token_hidden_states, probs=probs)

            generated_beam_infos, token_indices, step_output, probs = self._update_beam_infos(beam_size, generated_beam_infos, input_ids,
                                                                                              token_indices, step_output, probs,
                                                                                              end_ids_tensor, pad_token_id=pad_id,
                                                                                              early_stop=early_stop)

            def concat_new(value, name):
                if value is None: return None
                value = value.view(bsz, beam_size, -1)
                value = value.gather(1, token_indices[:, :, None].expand_as(value))
                value = value.view(bsz * beam_size, -1)
                if name == 'token':
                    return torch.concat([value, step_output.view(-1)[:, None]], dim=-1)
                elif name == 'position':
                    return torch.concat([value, value[:, -1:] + 1], dim=-1)
                elif name == 'attention':
                    value = value[:, None, None, :]
                    return torch.concat([value, value[:, :, :, -1:]], dim=-1)
                elif name == 'segment':
                    return torch.concat([value, value[:, -1:]], dim=-1)

            ############### 根据next_tokens对应的token_indices, 构造新的输入 ###############
            input_ids = concat_new(input_ids, name='token')
            position_ids = concat_new(position_ids, name='position')
            attention_mask = concat_new(attention_mask, name='attention')
            segment_ids = concat_new(segment_ids, name='segment')

            ############### 选择了不同的beam，k_v_past也要相应变化 ###############
            for i in range(bsz):
                token_indices[i, :] += i * beam_size  # 这一步的原因是token_indices的shape是（bsz,-1），而k_v_past元素的shape是(bsz*beam_size,-1)
            for i, layer_past in enumerate(k_v_past):
                _states = []
                for state_past in layer_past:
                    _states.append(state_past.index_select(0, token_indices.view(-1)))
                k_v_past[i] = tuple(_states)

            # END判断
            step = input_ids.shape[1] - 1
            if step > max_len:
                break

        return input_ids.view(bsz, beam_size, -1)


if __name__ == "__main__":
    pass
