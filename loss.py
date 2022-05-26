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