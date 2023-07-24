# -*- coding: utf-8 -*-
# @Time    : 2023/7/20 20:58
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : ft_bloom_DDP.py
# @Software: CleanTransformer
# @Description: ft_bloom_DDP

import os, json, sys, argparse
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
from examples.inference_bloom import load_model, load_config


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
    datasampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False,
                            sampler=datasampler,
                            collate_fn=partial(BelleDataset.collate_fn, tokenizer=tokenizer, max_length=max_length))
    return dataloader


def train(model, train_loader, epoches, save_interval=1000, print_interval=10, save_dir='./', use_torch_amp=None, apex_level=None):
    if use_torch_amp and apex_level != None:
        raise Exception('torch_amp和apex_level只能使用1个，use_torch_amp={}, apex_level={}'.format(use_torch_amp, apex_level))
    if (apex_level != None) and (apex_level not in ['O0', 'O1', 'O2', 'O3']):
        raise Exception('apex_level只能设置为O0,O1,O2,O3。当前值apex_level={}'.format(apex_level))

    local_rank = int(os.environ["LOCAL_RANK"])
    ############### 判断gpu or cpu ###############
    device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    ############### 使用apex/DDP ###############
    if apex_level != None:
        from apex.parallel import DistributedDataParallel as apexDDP
        from apex import amp
        print_rank("using apex...", rank_set=0)
        optimizer = AdamW(model.parameters(), lr=1e-5)
        model, optimizer = amp.initialize(model, optimizer, opt_level=apex_level)
        model = apexDDP(model, delay_allreduce=True)
    else:
        model = DDP(model, device_ids=[local_rank])
        optimizer = AdamW(model.parameters(), lr=1e-5)

    model.train()
    steps = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    ############### 使用pytorch amp ###############
    if use_torch_amp:
        print_rank("using pytorch amp...", rank_set=0)
        scaler = torch.cuda.amp.GradScaler()

    ############### train loop ###############
    for epoch in range(epoches):
        if train_loader.sampler:
            train_loader.sampler.set_epoch(epoch)
        for batch in train_loader:
            ############### 模型，数据必须在相同的设备 ###############
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            if use_torch_amp:
                with torch.cuda.amp.autocast():
                    outputs, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                    loss = outputs[0]
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs, _ = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
                loss = outputs[0]

                ############### 反向传播，参数更新 ###############
                optimizer.zero_grad()
                loss.backward()
                # if apex_level != None:
                #     with amp.scale_loss(loss, optimizer) as scaled_loss:
                #         scaled_loss.backward()
                # else:
                #     loss.backward()
                optimizer.step()

            ############### 模型保存 ###############
            steps += 1
            if steps == 1:
                print_rank('step{}: input_ids[0:2,10:20]={}'.format(steps, batch['input_ids'][0:2, 10:20]))
                if hasattr(model, 'lm_head'):
                    print_rank('step{}: lm_head.weigth.grad[100:110,100:110]={}'.format(steps, model.lm_head.weight.grad[100:110, 100:110]))
                else:
                    print_rank('step{}: lm_head.weigth.grad[100:110,100:110]={}'.format(steps, model.module.lm_head.weight.grad[100:110, 100:110]))

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
    group.add_argument('--use_torch_amp', action='store_true', default=False)
    group.add_argument('--apex_level', default=None, type=str)

    args, _unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    seed = 999
    init_seed(seed=seed)
    dist.init_process_group("nccl")

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

    train(model, train_loader, epoches=10, save_dir=save_dir,
          use_torch_amp=run_config.use_torch_amp, apex_level=run_config.apex_level)

    dist.destroy_process_group()
