# -*- coding: utf-8 -*-
# @Time    : 2023/6/8 00:25
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : bloom_interactive.py.py
# @Software: CleanTransformer
# @Description: bloom_interactive.py

import os, sys, argparse
import torch

from transformers import BloomTokenizerFast

from examples.inference_bloom import load_model, load_config


class ConsoleSteamer():
    def __init__(self, tokenizer, input_text_len, stops=None):
        self.tokenizer = tokenizer
        self.origin_text_len = input_text_len
        self.printed_text_len = input_text_len
        self.stops = stops
        self.final_generation = ''

    def __call__(self, token_ids):
        generated_sequence = token_ids.numpy().tolist()
        sequence = generated_sequence[0][0]

        text = self.tokenizer.decode(
            sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        if len(text) > self.printed_text_len:
            sys.stdout.write(text[self.printed_text_len:])
            sys.stdout.flush()
            self.final_generation = text

            if self.stops is not None:
                for stop in self.stops:
                    if text[self.origin_text_len:].find(stop) > -1:
                        print('<|endoftext|>')
                        sys.stdout.flush()
                        return True

            self.printed_text_len = len(text)

        return False


def generate(prompt, model, tokenizer, stops=['\nHuman:', '\nHuman：']):
    generation_configs = {
        'beam_size': 1,
        'max_gen_len': 500,
        'end_ids': tokenizer.eos_token_id,
        'pad_id': tokenizer.pad_token_id,
        'early_stop': True,
        'no_repeat_ngram_size': 2,
        'do_sample': True,
        'temperature': 0.8,
        'top_k': 10,
        'top_p': 0.8,
    }
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    encoded_input = tokenizer(prompt, return_tensors='pt', padding=False, max_length=1024, truncation=True)
    for k, v in encoded_input.items():
        if isinstance(v, torch.Tensor):
            encoded_input[k] = v.to(device)

    steamer = ConsoleSteamer(tokenizer, len(prompt), stops=stops)
    model.generate(**encoded_input, generation_configs=generation_configs,
                   steamers=steamer)
    return steamer.final_generation


def build_prompt(query, history=None):
    prompt = ''
    for ctx in history:
        prompt += '{}: {}\n\n'.format(ctx['role'], ctx['value'])
    prompt += 'Human: {}\n\nAssistant: '.format(query)
    return prompt


def init_args():
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--model_dir', default='checkpoints/bloom-396m-zh/', type=str)
    parser.add_argument('--model_name', default='pytorch_model.bin', type=str)
    parser.add_argument('--tokenizer_dir', default='checkpoints/bloom-396m-zh/', type=str)

    args, _unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = init_args()
    model_dir = args.model_dir
    tokenizer_dir = args.tokenizer_dir if args.tokenizer_dir else model_dir

    config = load_config(os.path.join(model_dir, "config.json"))
    tokenizer = BloomTokenizerFast.from_pretrained(tokenizer_dir, padding_side='left')
    model = load_model(config, os.path.join(model_dir, args.model_name))

    query, history = '', []
    while query.lower() != 'q':
        query = input('\nUser: ').strip()
        if query.lower() == 'q':
            sys.stdout.write('exit\n')
            break
        elif query.lower() == 'new':
            sys.stdout.write('create new session...\n')
            history = []
            continue
        else:
            prompt = build_prompt(query, history)
            sys.stdout.write('Assistant: ')
            generation = generate(prompt, model=model, tokenizer=tokenizer).strip()
            if generation.startswith('Assistant:'):
                generation = generation[10:].strip()

            history.append({'role': 'Human', 'value': query})
            history.append({'role': 'Assistant', 'value': generation})

            sys.stdout.write('\n')
