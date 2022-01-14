# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 16:24:04 2022

@author: Admin
"""

import random
import numpy as np

# slicing array
# https://www.hashbangcode.com/article/slicing-arrays-and-strings-using-colon-operator-python
arr = np.array(([1,2,3],[4,5,6],[7,8,9]))
print(arr[1][0])
print("-----------")
print(arr[1])
print("-----------")
print(arr[1:])
print("-----------")
print(arr[0:2][:])
print("-----------")
print(arr[0:2][1:])
print("-----------")
print(arr[0:2:2])

#init array using for
print("=======================")
points = [0,1,2,3,4,5]
arr2 = np.array([p for p in points])
print(arr2)