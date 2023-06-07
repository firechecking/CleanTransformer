# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 23:12
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : ft_bloom.py
# @Software: CleanTransformer
# @Description: ft_bloom

import os, json, argparse
from functools import partial
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BloomTokenizerFast

# from CleanTransformer.optimizer import AdamW
from torch.optim import AdamW
from examples.inference_bloom import load_model, load_config


class BelleDataset(Dataset):
    def __init__(self, path):
        self.data = [json.loads(l.strip()) for l in open(path, 'r')]

    def __getitem__(self, item):
        one_data = self.data[item]
        instruction, output = one_data['instruction'], one_data['output'] if 'output' in one_data else ''
        if not instruction.startswith('\n\nHuman: '):
            instruction = f'\n\nHuman: {instruction}'
        if not instruction.endswith('\n\nAssistant: '):
            instruction = f'{instruction}\n\nAssistant: '
        prompt = '{}{}'.format(instruction, output)
        return {"prompt": prompt}

    def __len__(self):
        return len(self.data)

    @classmethod
    def collate_fn(self, batch, tokenizer, max_length, pad_to_max=False):
        texts = [sample['prompt'] + tokenizer.eos_token for sample in batch]
        encoded_inputs = tokenizer(texts, truncation=True, padding=True, max_length=max_length, return_tensors='pt')
        outputs = {**encoded_inputs}

        while pad_to_max and outputs['input_ids'].shape[-1] < max_length:
            outputs['input_ids'] = torch.concat([outputs['input_ids'],
                                                 torch.full((len(batch), 1), tokenizer.pad_token_id)], dim=-1)
            outputs['attention_mask'] = torch.concat([outputs['attention_mask'],
                                                      torch.zeros(len(batch), 1)], dim=-1)

        outputs['labels'] = outputs['input_ids'].clone()
        outputs['prompts'] = texts

        return outputs


def build_dataloader(tokenizer, data_fn, batch_size, shuffle, max_length=1024):
    dataset = BelleDataset(data_fn)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                            collate_fn=partial(BelleDataset.collate_fn, tokenizer=tokenizer, max_length=max_length))
    return dataloader


def train(model, train_loader, epoches, save_interval=1000, print_interval=10, save_dir='./'):
    ############### 判断gpu or cpu ###############
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=1e-5)
    model.train()
    steps = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ############### train loop ###############
    for epoch in range(epoches):
        for batch in train_loader:
            ############### 模型，数据必须在相同的设备 ###############
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            outputs, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
            loss = outputs[0]

            ############### 反向传播，参数更新 ###############
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ############### 模型保存 ###############
            steps += 1
            if steps % print_interval == 0:
                print('step: {}, loss: {}'.format(steps, loss.cpu().item()))
            if steps % save_interval == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f'model_step_{steps}.pt'))


def init_seed(seed=999):
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_args():
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--model_dir', default='checkpoints/bloom-396m-zh/', type=str)
    parser.add_argument('--data_fn', default='datasets/belle/train_3.5M_CN_processed.jsonl', type=str)
    parser.add_argument('--save_dir', default='checkpoints/bloom-396m-zh-SFT-belle3.5M', type=str)
    parser.add_argument('--batch_size', default=32, type=int)

    args, _unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    seed = 999
    init_seed(seed=seed)
    run_config = init_args()

    model_dir = run_config.model_dir
    data_fn = run_config.data_fn

    config = load_config(os.path.join(model_dir, "config.json"))
    tokenizer = BloomTokenizerFast.from_pretrained(model_dir, padding_side='right')
    model = load_model(config, os.path.join(model_dir, 'pytorch_model.bin'))

    train_loader = build_dataloader(tokenizer, data_fn,
                                    batch_size=run_config.batch_size, shuffle=True)

    save_dir = run_config.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_config = config.__dict__
    save_config.update(run_config.__dict__)
    json.dump(save_config, open(os.path.join(save_dir, 'config.json'), 'w'), ensure_ascii=False, indent=4)

    train(model, train_loader, epoches=10, save_dir=save_dir)
