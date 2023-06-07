# -*- coding: utf-8 -*-
# @Time    : 2023/6/7 00:07
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : optimizer.py
# @Software: CleanTransformer
# @Description: optimizer

import torch


class SGD():
    def __init__(self, params, lr=0.01, momentum=None, dampening=0, weight_decay=None):
        self.params = params
        self.lr = lr

        self.momentum = momentum
        self.dampening = dampening
        self.momentum_buffer = [None for _ in params]

        self.weight_decay = weight_decay

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = None

    def step(self):
        prev_grad = torch.is_grad_enabled()
        try:
            ############### 关闭梯度传播 ###############
            torch.set_grad_enabled(False)

            ############### 更新权重 ###############
            for i, param in enumerate(self.params):
                ############### weight decay ###############
                if self.weight_decay:
                    param.grad += self.weight_decay * param
                ############### momentum ###############
                if self.momentum:
                    buf = self.momentum_buffer[i]
                    if buf is None:
                        buf = torch.clone(param.grad).detach()
                        self.momentum_buffer[i] = buf
                    else:  # 每个step，梯度都进行累计，通过momentum, dampening进行比例衰减
                        buf.mul_(self.momentum).add_(param.grad, alpha=1 - self.dampening)
                    param.grad = buf
                param -= self.lr * param.grad
        finally:
            torch.set_grad_enabled(prev_grad)


class AdamW():
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.params = params
        self.lr = lr

        self.beta1, self.beta2 = betas
        self.eps = eps
        self.momentum_buffer = [0 for _ in params]
        self.rmsp_buffer = [0 for _ in params]
        self.steps = [1 for _ in params]

        self.weight_decay = weight_decay

    def zero_grad(self):
        for param in self.params:
            if param.grad is not None:
                param.grad = None

    def step(self):
        prev_grad = torch.is_grad_enabled()
        try:
            ############### 关闭梯度传播 ###############
            torch.set_grad_enabled(False)

            ############### 更新权重 ###############
            for i, param in enumerate(self.params):
                ############### weight decay ###############
                if self.weight_decay:
                    param.grad += self.weight_decay * param

                ############### 一阶动量 ###############
                self.momentum_buffer[i] = self.beta1 * self.momentum_buffer[i] + (1 - self.beta1) * param.grad
                ############### 二阶动量 ###############
                self.rmsp_buffer[i] = self.beta2 * self.rmsp_buffer[i] + (1 - self.beta2) * param.grad * param.grad

                ############### 偏差纠正 ###############
                t = self.steps[i]
                momentum_correction = self.momentum_buffer[i] / (1 - self.beta1 ** t)
                rmsp_correction = self.rmsp_buffer[i] / (1 - self.beta2 ** t)
                self.steps[i] += 1

                ############### 权重更新 ###############
                param -= self.lr * momentum_correction / (rmsp_correction.sqrt() + self.eps)
        finally:
            torch.set_grad_enabled(prev_grad)


def sample_optimizer(optim_type):
    from torch.optim import SGD as OfficialSGD
    from torch.optim import AdamW as OfficialAdamW

    torch.manual_seed(999)
    gt_weight, gt_bias = torch.rand(3, 4, requires_grad=False), torch.rand(4, requires_grad=False)
    _weight, _bias = torch.rand(3, 4, requires_grad=True), torch.rand(4, requires_grad=True)
    if optim_type == 'sgd_official':
        optimizer = OfficialSGD([_weight, _bias], lr=0.01, weight_decay=0.01, momentum=0.9)
    elif optim_type == 'sgd_custom':
        optimizer = SGD([_weight, _bias], lr=0.01, weight_decay=0.01, momentum=0.9)
    elif optim_type == 'adam_official':
        optimizer = OfficialAdamW([_weight, _bias], lr=0.01, weight_decay=0.01)
    elif optim_type == 'adam_custom':
        optimizer = AdamW([_weight, _bias], lr=0.01, weight_decay=0.01)
    print('\n', '=' * 50, optim_type)
    for i in range(300):
        inp = torch.rand((2, 3))
        gt = torch.matmul(inp, gt_weight) + gt_bias
        pred = torch.matmul(inp, _weight) + _bias
        loss = torch.sum((gt - pred).view(-1).pow(2))
        loss.backward()
        print('optimizer: {}, step: {}, loss: {}'.format(optim_type, i, loss.data))

        optimizer.step()
        optimizer.zero_grad()


if __name__ == "__main__":
    sample_optimizer(optim_type='sgd_official')
    sample_optimizer(optim_type='sgd_custom')
    sample_optimizer(optim_type='adamw_official')
    sample_optimizer(optim_type='adamw_custom')
