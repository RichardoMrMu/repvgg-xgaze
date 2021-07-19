# -*- coding: utf-8 -*-
# @Time    : 2020-12-04 10:44
# @Author  : RichardoMu
# @File    : model.py
# @Software: PyCharm

import torch.nn as nn

from modules import resnet50,resnet18,resnet34,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d,wide_resnet50_2,wide_resnet101_2

class gaze_network(nn.Module):
    def __init__(self, net_choice='resnet50',use_face=False, num_glimpses=1):
        super(gaze_network, self).__init__()
        if net_choice == 'resnet50':
            self.gaze_network = resnet50(pretrained=True)
        elif net_choice== 'resnet18':
            self.gaze_network = resnet18(pretrained=True)
        elif net_choice == 'resnet34':
            self.gaze_network = resnet34(pretrained=True)
        elif net_choice == 'resnet101':
            self.gaze_network = resnet101(pretrained=True)
        elif net_choice == 'resnet152':
            self.gaze_network = resnet152(pretrained=True)
        elif net_choice == 'resnext50_32x4d':
            self.gaze_network = resnext50_32x4d(pretrained=True)
        elif net_choice == 'resnext101_32x8d':
            self.gaze_network = resnext101_32x8d(pretrained=True)
        elif net_choice == 'wide_resnet50_2':
            self.gaze_network = wide_resnet50_2(pretrained=True)
        elif net_choice == 'wide_resnet101_2':
            self.gaze_network = wide_resnet101_2(pretrained=True)
        print(f"the model chosed is {net_choice}")
        if net_choice == 'resnet34' or net_choice == 'resnet18':
            fc_num = 512
        else:
            fc_num = 2048
        self.gaze_fc = nn.Sequential(
            nn.Linear(fc_num, 2),
        )

    def forward(self, x):
        feature = self.gaze_network(x)
        feature = feature.view(feature.size(0), -1)
        gaze = self.gaze_fc(feature)

        return gaze