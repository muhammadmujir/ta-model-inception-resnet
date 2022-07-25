# -*- coding: utf-8 -*-
"""
Created on Fri May 27 02:11:16 2022

@author: Admin
"""

import torch
from torch import Tensor

class CustomMSELoss(torch.nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', root=False) -> None:
        self.root = root
        super(CustomMSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.root:
            return torch.sqrt(super(CustomMSELoss, self).forward(input, target))
        return super(CustomMSELoss, self).forward(input, target)

class LossPerPatch(torch.nn.L1Loss):
    def __init__(self, kernelSize=5, stride=5, mode='average') -> None:
        super(LossPerPatch, self).__init__(size_average=None, reduce=None, reduction='sum')
        self.kernelSize = kernelSize
        self.mode = mode
        if mode == 'average':
            self.pooling = torch.nn.AvgPool2d(kernel_size=kernelSize, stride=stride)
        else:
            self.pooling = torch.nn.MaxPool2d(kernel_size=kernelSize, stride=stride)
        
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(input.shape) < 3 or len(target.shape) < 3 or len(input.shape) > 4 or len(target.shape) > 4:
            raise ValueError("input and target tensor must be in 3D or 4D")
        if len(input.shape) != len(target.shape):
            raise ValueError("input and target shape must be same")
        # Add Padding At Right of Tensor
        mod = input.shape[len(input.shape)-1] % self.kernelSize
        if mod != 0:
            padding = torch.zeros((input.shape[len(input.shape)-2], self.kernelSize-mod))
            for i in range(len(input.shape)-2):
                padding = padding.unsqueeze(0)
            input = torch.cat((input, padding), len(input.shape)-1)
            target = torch.cat((target, padding), len(input.shape)-1)
        # Add Padding At Bottom of Tensor
        mod = input.shape[len(input.shape)-2] % self.kernelSize
        if mod != 0:
            padding = torch.zeros((self.kernelSize-mod,input.shape[len(input.shape)-1]))
            for i in range(len(input.shape)-2):
                padding = padding.unsqueeze(0)
            input = torch.cat((input, padding), len(input.shape)-2)
            target = torch.cat((target, padding), len(input.shape)-2)
        input = self.pooling(input)
        target = self.pooling(target)
        if self.mode == 'average':
            input = input * (self.kernelSize * self.kernelSize)
            target = target * (self.kernelSize * self.kernelSize)
        return super(LossPerPatch, self).forward(input, target)