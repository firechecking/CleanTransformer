# -*- coding: utf-8 -*-
# @Time    : 2023/6/6 19:44
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : inference_bloom.py
# @Software: CleanTransformer
# @Description: inference_bloom


import os, json, torch, argparse
from collections import OrderedDict
from CleanTransformer.models.modeling_bloom import BloomForCausalLM, BloomConfig
from transformers import BloomTokenizerFast


def load_model(config, ckpt_path):
    def map_from_huggingface(state_dict):
        if 'bloom.word_embeddings.weight' in state_dict:
            return state_dict
        prefix = 'transformer.' if 'transformer.word_embeddings.weight' in state_dict else ''

        new_state_dict = OrderedDict()
        new_state_dict['bloom.word_embeddings.weight'] = state_dict[f'{prefix}word_embeddings.weight']
        new_state_dict['bloom.word_embeddings_layernorm.weight'] = state_dict[f'{prefix}word_embeddings_layernorm.weight']
        new_state_dict['bloom.word_embeddings_layernorm.bias'] = state_dict[f'{prefix}word_embeddings_layernorm.bias']
        for i in range(config.n_layer):
            for t in ('weight', 'bias'):
                for name in ('input_layernorm', 'self_attention.query_key_value', 'self_attention.dense',
                             'post_attention_layernorm', 'mlp.dense_h_to_4h', 'mlp.dense_4h_to_h'):
                    new_state_dict[f'bloom.blocks.{i}.{name}.{t}'] = state_dict[f'{prefix}h.{i}.{name}.{t}']

        for t in ('weight', 'bias'):
            new_state_dict[f'bloom.ln_f.{t}'] = state_dict[f'{prefix}ln_f.{t}']
        new_state_dict['lm_head.weight'] = state_dict['lm_head.weight'] if 'lm_head.weight' in state_dict \
            else state_dict[f'{prefix}word_embeddings.weight']
        return new_state_dict

    model = BloomForCausalLM(config)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if not isinstance(state_dict, dict):
        state_dict = state_dict.state_dict()

    new_state_dict = map_from_huggingface(state_dict)
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model._tie_weight()
    return model


def load_config(config_fn):
    _d = json.load(open(config_fn, 'r'))

    synonymes_list = (['n_embed', 'hidden_size'], ['n_head', 'num_attention_heads'])
    for _synonymes in synonymes_list:
        source_k = None
        for k in _synonymes:
            if k in _d:
                source_k = k
                break
        for k in _synonymes:
            _d[k] = _d[source_k]

    return BloomConfig(**_d)


def init_args():
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--model_dir', type=str)
    parser.add_argument('--model_name', default='pytorch_model.bin', type=str)

    args, _unknown = parser.parse_known_args()
    return args


def sample_generate():
    # model_dir = "../checkpoints/bloom-560m/"
    model_dir = "../checkpoints/bloom-396m-zh/"
    config = load_config(os.path.join(model_dir, "config.json"))
    tokenizer = BloomTokenizerFast.from_pretrained(model_dir, padding_side='left')
    texts = ["when we talk about something for the first time,", "New York City plans to"]
    texts = ['以神舟十六号发射为题，写一首诗', '1+1=']

    encoded_input = tokenizer(texts, return_tensors='pt', padding=True)

    model = load_model(config, os.path.join(model_dir, 'pytorch_model.bin'))

    generation_configs = {
        'beam_size': 1,
        'max_gen_len': 100,
        'end_ids': tokenizer.eos_token_id,
        'pad_id': tokenizer.pad_token_id,
        'early_stop': True,
        'no_repeat_ngram_size': 2,
        'do_sample': True,
        'temperature': 0.8,
        'top_k': 10,
        'top_p': 0.8,
    }
    generated_sequence = model.generate(**encoded_input, generation_configs=generation_configs)
    generated_sequence = generated_sequence.numpy().tolist()
    print(generated_sequence)

    for sequences in generated_sequence:
        for i, sequence in enumerate(sequences):
            text = tokenizer.decode(
                sequence,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            print('beam: ', i, text)


if __name__ == "__main__":
    sample_generate()
