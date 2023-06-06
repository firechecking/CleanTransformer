# -*- coding: utf-8 -*-
# @Time    : 2023/6/6 20:46
# @Author  : Patrick
# @Email   : firechecking@gmail.com
# @File    : loss.py
# @Software: CleanTransformer
# @Description: loss


import torch
from torch.nn import MSELoss as OfficialMSELoss
from torch.nn import CrossEntropyLoss as OfficialCrossEntropyLoss
from torch.nn import NLLLoss as OfficialNLLLoss
from torch.nn import LogSoftmax as OfficialLogSoftmax


class MSELoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(MSELoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        loss = (input - target).pow(2)
        if self.reduction == 'mean':
            return loss.mean()
        return loss.sum()


class CrossEntropyLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        ############### softmax求预测值概率 ###############
        exp_sum = torch.sum(torch.exp(input), dim=-1).unsqueeze(-1)
        logsoftmax = torch.log(torch.exp(input) / exp_sum)

        if len(target.shape) == len(input.shape) - 1:
            ############### one-hot类型，其他类别概率为0，所以只计算真实概率为1的类别 ###############
            loss = -torch.sum(torch.gather(logsoftmax, dim=1, index=target.unsqueeze(1)))
        else:
            # target的最后一维是概率，且和input维度相同
            ############### 非one-hot类型，其他类别概率不为0，需要对应概率相乘 ###############
            logsoftmax = target * logsoftmax
            loss = -torch.sum(logsoftmax)
        if self.reduction == 'mean':
            loss = loss / input.shape[0]
        return loss


class LogSoftmax(torch.nn.Module):
    def __init__(self, dim):
        super(LogSoftmax, self).__init__()
        self.dim = dim

    def forward(self, input):
        _sum = torch.sum(torch.exp(input), dim=self.dim).unsqueeze(self.dim)
        logsoftmax = torch.log(torch.exp(input) / (_sum + 1e-9))
        return logsoftmax


class NLLLoss(torch.nn.Module):
    def __init__(self, reduction='mean'):
        super(NLLLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target):
        ############### 选择one-hot类别对应的预测概率值，并求和 ###############
        r = -torch.sum(torch.gather(input, dim=1, index=target.unsqueeze(1)))
        if self.reduction == 'mean':
            return r / input.shape[0]
        return r


def sample_loss():
    losses = {"mse_official": OfficialMSELoss, "cross_entropy_official": OfficialCrossEntropyLoss, "nll_official": OfficialNLLLoss,
              "mse_custom": MSELoss, "cross_entropy_custom": CrossEntropyLoss, "nll_custom": NLLLoss,
              'prob_cross_entropy_official': OfficialCrossEntropyLoss, 'prob_cross_entropy_custom': CrossEntropyLoss}
    for reduction in ('mean', 'sum'):
        print('=' * 40, reduction)
        for loss_type in ('mse_official', 'mse_custom', 'nll_official', 'nll_custom',
                          'cross_entropy_official', 'cross_entropy_custom', 'prob_cross_entropy_official', 'prob_cross_entropy_custom'):
            torch.manual_seed(999)
            if 'mse' in loss_type or 'prob_cross' in loss_type:
                pred = torch.rand(3, 4)
                gt = torch.rand(3, 4)
            else:
                pred = torch.rand(3, 4)
                gt = torch.randint(0, 4, (3,))
            loss = losses[loss_type](reduction='mean')
            r = loss(pred, gt)
            print('loss_type: {}, loss: {}'.format(loss_type, r))


def sample_logsoftmax():
    torch.manual_seed(999)
    x = torch.rand(3, 4)
    print('type: official_log_softmax, loss: {}'.format(OfficialLogSoftmax(dim=1)(x)))
    print('type: custom_log_softmax, loss: {}'.format(LogSoftmax(dim=1)(x)))


if __name__ == "__main__":
    sample_logsoftmax()
    sample_loss()
