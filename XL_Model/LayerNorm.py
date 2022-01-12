# encoding=utf-8
import torch
from torch import nn
from config import *
import copy


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Classify(nn.Module):
    def __init__(self, hidden_size, classify):
        super(Classify, self).__init__()
        self.classify = nn.Linear(hidden_size, classify)

    def forward(self, dense_x):
        classify_res = self.classify(dense_x[:, 0:1, :].squeeze(1))
        return classify_res


class LayerNorm(nn.Module):
    def __init__(self, eps, hidden_size):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.feature = hidden_size
        self.weight = torch.ones(self.feature)
        self.bias = torch.zeros(self.feature)

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return self.weight * (x-mean) / (std + self.eps) + self.bias


if __name__ == '__main__':
    x = torch.rand(2, 5, args.hidden_size)
    layer_norm = LayerNorm(args.eps, args.hidden_size)
    print(layer_norm(x))











