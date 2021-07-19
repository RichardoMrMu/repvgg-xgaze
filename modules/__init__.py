# -*- coding: utf-8 -*-
# @Time    : 2020-12-04 10:42
# @Author  : RichardoMu
# @File    : __init__.py.py
# @Software: PyCharm

from .resnet import resnet50
from .resnet import resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,wide_resnet101_2
__all__ = [
    'resnet50',
    'resnext101_32x8d',
    'resnext50_32x4d',
    'resnet152',
    'resnet34',
    'resnet101',
    'resnet18',
    'wide_resnet101_2',
    'wide_resnet50_2'

]