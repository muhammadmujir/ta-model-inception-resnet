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
# INIT ARRAY & NUMPY
# =======================================================================================

import numpy as np
arr = [[1,2,3],[4,5,6], [7,8,9]]
newArr = [len(p) for p in arr]
newArr2 = [len(p) for p in arr if p[0] > 1]
print(newArr)
print(np.array(newArr))
print(newArr2)
print(np.arange(0,4))
print(np.arange(1))
print(np.arange(0.5,3.5))
print(np.arange(4)/2)
print(2/np.arange(1,4))
print(np.asarray([[],[]]).ndim)
print(np.asarray([[1,2,3],[3,4,5]]).shape)
print(5//2)

# =======================================================================================
# Gussian Filter
# =======================================================================================
from scipy.ndimage import gaussian_filter
# a = np.arange(50, step=2).reshape((5,5))
# gaussian_filter(a, sigma=1)
from scipy import misc
import matplotlib.pyplot as plt
fig = plt.figure()
plt.gray()  # show the filtered result in grayscale
ax1 = fig.add_subplot(121)  # left side
ax2 = fig.add_subplot(122)  # right side
ascent = misc.ascent()
result = gaussian_filter(ascent, sigma=5)
ax1.imshow(ascent)
ax2.imshow(result)
plt.show()

from scipy.ndimage import gaussian_filter1d
gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 1)
gaussian_filter1d([1.0, 2.0, 3.0, 4.0, 5.0], 4)
import matplotlib.pyplot as plt
rng = np.random.default_rng()
x = rng.standard_normal(101).cumsum()
y3 = gaussian_filter1d(x, 3)
y6 = gaussian_filter1d(x, 6)
plt.plot(x, 'k', label='original data')
plt.plot(y3, '--', label='filtered, sigma=3')
plt.plot(y6, ':', label='filtered, sigma=6')
plt.legend()
plt.grid()
plt.show()