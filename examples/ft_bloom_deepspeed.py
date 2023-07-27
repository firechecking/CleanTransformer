# -*- coding: utf-8 -*-
# @Time    : 2023/7/24 20:08
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : ft_bloom_deepspeed.py
# @Software: CleanTransformer
# @Description: ft_bloom_deepspeed

import os, json, sys, argparse
from collections import OrderedDict
from functools import partial
import numpy as np

sys.path.append('.')

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader

from transformers import BloomTokenizerFast

# from CleanTransformer.optimizer import AdamW
from torch.optim import AdamW
from examples.inference_bloom import load_config
from CleanTransformer.models.modeling_bloom import BloomForCausalLM, BloomConfig


def load_model(config, ckpt_path, ds_config=None, ds_stage=0):
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

    if ds_config is not None and ds_stage == 3:
        with deepspeed.zero.Init():
            model = BloomForCausalLM(config)
    else:
        model = BloomForCausalLM(config)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    if not isinstance(state_dict, dict):
        state_dict = state_dict.state_dict()

    new_state_dict = map_from_huggingface(state_dict)
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    model._tie_weight()
    return model


def print_rank(value, rank_set=None):
    rank = int(os.environ['RANK'])
    if rank_set == None or rank == rank_set:
        print(f'rank {rank}: {value}')


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
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=partial(BelleDataset.collate_fn, tokenizer=tokenizer, max_length=max_length))
    return dataloader


def train(model, train_loader, epoches, save_interval=1000, print_interval=10, save_dir='./', ds_config=None, ds_stage=0):
    local_rank = int(os.environ["LOCAL_RANK"])
    ############### 判断gpu or cpu ###############
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    ############### 使用deepspeed/DDP ###############
    if ds_config != None:
        print_rank('deepspeed initialize...')
        model_engine, optimizer, _, _ = deepspeed.initialize(config_params=ds_config,
                                                             model=model,
                                                             model_parameters=model.parameters())
    else:
        model = DDP(model, device_ids=[local_rank])
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

            if ds_config != None:
                outputs, _ = model_engine(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs[0]
                model_engine.backward(loss)
                model_engine.step()
            else:
                outputs, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs[0]

                ############### 反向传播，参数更新 ###############
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            ############### 模型保存 ###############
            steps += 1
            if steps == 1:
                print_rank('step{}: input_ids[0:2,10:20]={}'.format(steps, batch['input_ids'][0:2, 10:20]))
                if ds_config == None or ds_stage == 1:
                    if hasattr(model, 'lm_head'):
                        print_rank('step{}: lm_head.weigth.grad[100:110,100:110]={}'.format(steps, model.lm_head.weight.grad[100:110, 100:110]))
                    else:
                        print_rank(
                            'step{}: lm_head.weigth.grad[100:110,100:110]={}'.format(steps, model.module.lm_head.weight.grad[100:110, 100:110]))

            if steps % print_interval == 0 and local_rank == 0:
                print_rank('step: {}, loss: {}'.format(steps, loss.cpu().item()))

            if steps % save_interval == 0 and local_rank == 0:
                torch.save(model.state_dict(), os.path.join(save_dir, f'model_step_{steps}.pt'))


def init_seed(seed=999):
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_args():
    parser = argparse.ArgumentParser(description='Args')
    group = parser.add_argument_group(title='train')
    group.add_argument('--model_dir', default='checkpoints/bloom-396m-zh/', type=str)
    group.add_argument('--data_fn', default='datasets/belle/train_3.5M_CN_processed.jsonl', type=str)
    group.add_argument('--save_dir', default='checkpoints/bloom-396m-zh-SFT-belle3.5M', type=str)
    group.add_argument('--batch_size', default=32, type=int)

    group = parser.add_argument_group(title='distributed')
    group.add_argument('--ds_config', default=None, type=str)
    group.add_argument('--ds_stage', default=0, type=int)

    args, _unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    seed = 999
    init_seed(seed=seed)
    run_config = init_args()

    if run_config.ds_config is not None:
        import deepspeed

        deepspeed.init_distributed()
    else:
        dist.init_process_group("nccl")

    model_dir = run_config.model_dir
    data_fn = run_config.data_fn

    config = load_config(os.path.join(model_dir, "config.json"))
    tokenizer = BloomTokenizerFast.from_pretrained(model_dir, padding_side='right')
    model = load_model(config, os.path.join(model_dir, 'pytorch_model.bin'))

    train_loader = build_dataloader(tokenizer, data_fn,
                                    batch_size=run_config.batch_size, shuffle=True)

    save_dir = run_config.save_dir
    if not os.path.exists(save_dir) and int(os.environ["LOCAL_RANK"]) == 0:
        os.makedirs(save_dir)
    dist.barrier()
    save_config = config.__dict__
    save_config.update(run_config.__dict__)
    json.dump(save_config, open(os.path.join(save_dir, 'config.json'), 'w'), ensure_ascii=False, indent=4)

    train(model, train_loader, epoches=10, save_dir=save_dir,
          ds_config=run_config.ds_config, ds_stage=run_config.ds_stage)

    if run_config.ds_config is None:
        dist.destroy_process_group()
