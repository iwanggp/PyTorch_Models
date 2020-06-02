# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 01:31
# @Author  : gpwang
# @File    : vgg.py
# @Software: PyCharm
import torch
import torch.nn as nn

# __all__=['']
# vgg相关参数的配置，这里BN层几乎是所有CNN的标配了，故这里不再实现无BN层的网络
from mish import Mish

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

act_fn = Mish()


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            act_fn,
            nn.Dropout(),
            nn.Linear(4096, 4096),
            Mish(),
            nn.Dropout(),
            nn.Linear(4096, 2), )

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':  # 如果是M就添加池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           act_fn]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


if __name__ == '__main__':
    model = VGG('VGG16')
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(y.size())
