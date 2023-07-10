# -*- coding: utf-8 -*-
# @Time    : 2023/5/30 16:04
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : data_preprocess.py
# @Software: CleanTransformer
# @Description: data_preprocess

import argparse, json, tqdm


class Processor():
    def __init__(self, config):
        self.config = config
        print('loading data from {}'.format(self.config.input_fn))
        self.dataset = [json.loads(l) for l in open(self.config.input_fn, 'r')]
        print('loaded {} data'.format(len(self.dataset)))

    def call(self):
        roles = ['Human', 'Assistant']
        with open(self.config.output_fn, 'w') as f:
            for data in tqdm.tqdm(self.dataset):
                instruction, reference = '', ''
                ############### 构造输入 ###############
                for i, term in enumerate(data['conversations'][:-1]):
                    assert term['from'].lower() == roles[i % 2].lower()
                    instruction += '\n\n{}: {}'.format(roles[i % 2], term['value'].strip())
                instruction += '\n\nAssistant: '

                ############### 构造输出 ###############
                reference = data['conversations'][-1]['value'].strip()
                f.write(json.dumps({"instruction": instruction, "output": reference}, ensure_ascii=False))
                f.write('\n')


def init_args():
    parser = argparse.ArgumentParser(description='Args')
    parser.add_argument('--input_fn', type=str, default='train_3.5M_CN.json')
    parser.add_argument('--output_fn', type=str, default='train_3.5M_CN_processed.jsonl')

    args, _unknown = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = init_args()
    processor = Processor(args)
    processor.call()
