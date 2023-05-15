# -*- coding: utf-8 -*-
# @Time    : 2023/5/15 19:55
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : inference_gpt2.py
# @Software: CleanTransformer
# @Description: inference_gpt2

import json, torch
from collections import OrderedDict
from transformers import GPT2Tokenizer
from CleanTransformer.models.modeling_gpt import GPTLMHeadModel, GPTConfig


def load_model(config, ckpt_path):
    def map_huggingface_to_gpt2(state_dict):
        new_state_dict = OrderedDict()
        new_state_dict['gpt.tokens_embed.weight'] = state_dict['wte.weight']
        new_state_dict['gpt.position_embed.weight'] = state_dict['wpe.weight']
        for i in range(config.n_layer):
            new_state_dict[f'gpt.blocks.{i}.attn.bias'] = state_dict[f'h.{i}.attn.bias']
            for t in ('weight', 'bias'):
                new_state_dict[f'gpt.blocks.{i}.attn.c_attn.{t}'] = state_dict[f'h.{i}.attn.c_attn.{t}']
                new_state_dict[f'gpt.blocks.{i}.attn.c_proj.{t}'] = state_dict[f'h.{i}.attn.c_proj.{t}']

                new_state_dict[f'gpt.blocks.{i}.norm1.{t}'] = state_dict[f'h.{i}.ln_1.{t}']

                new_state_dict[f'gpt.blocks.{i}.mlp.0.{t}'] = state_dict[f'h.{i}.mlp.c_fc.{t}']
                new_state_dict[f'gpt.blocks.{i}.mlp.2.{t}'] = state_dict[f'h.{i}.mlp.c_proj.{t}']

                new_state_dict[f'gpt.blocks.{i}.norm2.{t}'] = state_dict[f'h.{i}.ln_2.{t}']

        for t in ('weight', 'bias'):
            new_state_dict[f'gpt.ln_f.{t}'] = state_dict[f'ln_f.{t}']
        new_state_dict['lm_head.weight'] = state_dict['wte.weight']
        return new_state_dict

    model = GPTLMHeadModel(config, version='gpt2')
    state_dict = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = map_huggingface_to_gpt2(state_dict)
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model


def load_config(config_fn):
    _d = json.load(open(config_fn, 'r'))
    return GPTConfig(**_d)


def sample_generate():
    text = "New York City plans to"

    config = load_config("../checkpoints/gpt2/config.json")
    tokenizer = GPT2Tokenizer.from_pretrained('../checkpoints/gpt2')
    model = load_model(config, '../checkpoints/gpt2/pytorch_model.bin')

    eos_token = '<|endoftext|>'
    encoded_input = tokenizer(text, return_tensors='pt')
    print(encoded_input)

    generation_configs = {'beam_size': 1,
                          'max_gen_len': 100,
                          'end_ids': tokenizer.convert_tokens_to_ids(eos_token)}

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
