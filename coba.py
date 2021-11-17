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