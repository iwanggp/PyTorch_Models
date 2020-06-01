# -*- coding: utf-8 -*-
# @Time    : 2020/6/2 00:52
# @Author  : gpwang
# @File    : test.py
# @Software: PyCharm
import torchvision.models as models
vgg=models.vgg16()
import torch
x=torch.randn(2,3,640,640)
y=vgg(x)
print(y.size())