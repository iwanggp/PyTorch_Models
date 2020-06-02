# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 02:05
# @Author  : gpwang
# @File    : mish.py
# @Software: PyCharm
'''
PyTorch实现的mish激活函数
'''
# import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()
        print("Mish activation loaded......")

    def forward(self, x):
        x = x * (torch.tanh(F.softplus(x)))
        return x


if __name__ == '__main__':
    mish = Mish()
    x = torch.linspace(-10, 10, 1000)
    y = mish(x)
