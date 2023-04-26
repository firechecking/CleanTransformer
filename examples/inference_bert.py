# -*- coding: utf-8 -*-
# @Time    : 2023/4/26 6:13 PM
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : inference_bert.py
# @Software: CleanTransformer
# @Description: inference_bert

import json
import torch
from collections import OrderedDict
from CleanTransformer.models.modeling_bert import BertForSequenceClassification, BertTokenizer, BertConfig


def load_model(config, ckpt_path):
    def map_mmtkx_to_bert(state_dict):
        '''
        因为自定义模型和huggingface模型名字又差异，所以做一下映射后才能加载成功
        '''
        new_state_dict = OrderedDict()
        new_state_dict['bert.word_embeddings.weight'] = state_dict['bert.embeddings.word_embeddings.weight']
        new_state_dict['bert.position_embeddings.weight'] = state_dict['bert.embeddings.position_embeddings.weight']
        new_state_dict['bert.segment_embeddings.weight'] = state_dict['bert.embeddings.token_type_embeddings.weight']
        new_state_dict['bert.embedding_post.0.weight'] = state_dict['bert.embeddings.LayerNorm.weight']
        new_state_dict['bert.embedding_post.0.bias'] = state_dict['bert.embeddings.LayerNorm.bias']
        for i in range(12):
            for t in ('weight', 'bias'):
                new_state_dict[f'bert.blocks.{i}.attention.q_linear.{t}'] = state_dict[f'bert.encoder.layer.{i}.attention.self.query.{t}']
                new_state_dict[f'bert.blocks.{i}.attention.k_linear.{t}'] = state_dict[f'bert.encoder.layer.{i}.attention.self.key.{t}']
                new_state_dict[f'bert.blocks.{i}.attention.v_linear.{t}'] = state_dict[f'bert.encoder.layer.{i}.attention.self.value.{t}']

                new_state_dict[f'bert.blocks.{i}.attention_post.0.{t}'] = state_dict[f'bert.encoder.layer.{i}.attention.output.dense.{t}']
                new_state_dict[f'bert.blocks.{i}.norm1.{t}'] = state_dict[f'bert.encoder.layer.{i}.attention.output.LayerNorm.{t}']
                new_state_dict[f'bert.blocks.{i}.ffw.0.{t}'] = state_dict[f'bert.encoder.layer.{i}.intermediate.dense.{t}']
                new_state_dict[f'bert.blocks.{i}.ffw.2.{t}'] = state_dict[f'bert.encoder.layer.{i}.output.dense.{t}']
                new_state_dict[f'bert.blocks.{i}.norm2.{t}'] = state_dict[f'bert.encoder.layer.{i}.output.LayerNorm.{t}']
        for t in ('weight', 'bias'):
            new_state_dict[f'bert.pooler.0.{t}'] = state_dict[f'bert.pooler.dense.{t}']
            new_state_dict[f'classifier.{t}'] = state_dict[f'classifier.{t}']
        return new_state_dict

    ############### 实例化模型 ###############
    model = BertForSequenceClassification(config)

    ############### 加载huggingface checkpoint ###############
    state_dict = torch.load(ckpt_path, map_location='cpu')
    new_state_dict = map_mmtkx_to_bert(state_dict)
    model.load_state_dict(new_state_dict, strict=True)
    return model


def load_config(config_fn):
    _d = json.load(open(config_fn, 'r'))
    return BertConfig(**_d)


if __name__ == "__main__":
    config = load_config('../checkpoints/bert-base-go-emotion/config.json')
    classes = config.id2label
    config.num_labels = len(config.id2label)
    print('classes: {}'.format(config.num_labels))

    ############### 初始化模型，并加载checkpoint ###############
    model = load_model(config, '../checkpoints/bert-base-go-emotion/pytorch_model.bin')

    ############### 构造模型输入 ###############
    tokenizer = BertTokenizer(vocab_file='../checkpoints/bert-base-go-emotion/vocab.txt')
    query = 'I like you. I love you'
    tokens = tokenizer.encode_plus(query, padding=False, truncation=False)

    ############### 模型推理 ###############
    model.eval()
    output_ids = model(input_ids=torch.tensor(tokens['input_ids']).unsqueeze(0),
                       attention_mask=torch.tensor(tokens['attention_mask']).unsqueeze(0),
                       segment_ids=torch.tensor(tokens['segment_ids']).unsqueeze(0))

    ############### 打印推理结果 ###############
    output_probs = torch.softmax(output_ids, dim=-1)
    pred_idx = torch.argmax(output_probs, dim=-1)
    print('max_pred: {}, max_prob: {}'.format(classes[str(pred_idx.item())], output_probs[0, pred_idx].item()))

    print('=' * 10, ' details ', '=' * 10)
    scores = [(i, prob) for i, prob in enumerate(output_probs.tolist()[0])]
    scores = sorted(scores, key=lambda x: -x[1])
    for i, prob in scores:
        print(classes[str(i)], prob)
