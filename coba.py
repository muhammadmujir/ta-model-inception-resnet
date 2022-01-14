# -*- coding: utf-8 -*-
"""
Created on Thu Nov  4 15:32:29 2021

@author: Admin
"""
import numpy as np
from scipy.spatial import KDTree

x, y = np.mgrid[0:5, 2:8]
tree = KDTree(np.c_[x.ravel(), y.ravel()])

dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=1)
print(dd[1])
print(dd, ii, sep='\n')
print("============================")

dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1])
print(dd, ii, sep='\n')
print("============================")

dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=2)
print(dd, ii, sep='\n')
print("============================")

dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[2])
print(dd, ii, sep='\n')
print("============================")

dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=[1, 2])
print(dd, ii, sep='\n')


#====================================================
#calculate dimension of image
#====================================================
# reference : model.py

def dimenOfConv(width=0, height=0, padding=1, kernel=3, stride=1):
    resultWidth = ((width+padding+padding-kernel)/stride)+1
    resultHeight = ((height+padding+padding-kernel)/stride)+1
    return (resultWidth,resultHeight)

def dimenOfPooling(width=0, height=0, padding=0, kernel=2, stride=2):
    resultWidth = ((width-kernel)/stride)+1
    resultHeight = ((height-kernel)/stride)+1
    return (resultWidth,resultHeight)


width = 1024
height = 746

conv = [True,True,False,True,True,False,True,True,True,False,True,True,True,True,True,True,True,True,True,True]

for i in range(len(conv)):
    if conv[i]==True:
        if (i == 19):
            width,height = dimenOfConv(width=width, height=height, padding=0, kernel=1)
        elif (i > 13):
            width,height = dimenOfConv(width=width, height=height, padding=2, kernel=5)
        else:
            width,height = dimenOfConv(width=width, height=height, padding=1)
    else:
        width,height = dimenOfPooling(width=width, height=height)
    
    print(width,height)

# =======================================================================================
# MSE LOSS
# =======================================================================================

import torch
import torch.nn as nn

loss = nn.MSELoss(size_average=False)
input = torch.randn(3, 5, requires_grad=True)
input = torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]], requires_grad=True)
target = torch.randn(3, 5)
target = torch.tensor([[3.,4.,5.],[6.,7.,8.],[9.,10.,11.]], requires_grad=True)
print("input", input)
print("target", target)
output = loss(input, target)
print("output", output)
output.backward()
print("output after", output)


# =======================================================================================
# INIT ARRAY
# =======================================================================================

import numpy as np
arr = [[1,2,3],[4,5,6], [7,8,9]]
newArr = [len(p) for p in arr]
print(newArr)
print(np.array(newArr))