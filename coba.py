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

loss = nn.L1Loss(size_average=False)
input = torch.randn(3, 5, requires_grad=True)
input = torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
target = torch.randn(3, 5)
target = torch.tensor([[3.,4.,5.],[6.,7.,8.],[9.,10.,11.]])
print("input", input)
print("target", target)
output = loss(input, target)
print("output", output)
output.backward()
print("output after", output)

print(input.sum())
print(target.sum())
print(abs(input.sum() - target.sum()))

target = torch.load("C:\\Users\\Admin\\Desktop\\TA\\Dataset\\Result\\target.pt")
output = torch.load("C:\\Users\\Admin\\Desktop\\TA\\Dataset\\Result\\output.pt")
target = torch.tensor([[0.1402,4.,5.],[6.,7.,8.],[9.,10.,11.]], requires_grad=True)
output = torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
criterion = nn.L1Loss(reduction='sum').cpu().type(torch.FloatTensor)
loss = criterion(output, target)
print(abs(output.data.sum()-target.sum().type(torch.FloatTensor).cpu()))
print("Criterion: ", loss)
loss.backward()
print("Output: ", output.data.sum())
print("Target: ", target.sum().type(torch.FloatTensor).cpu())
# torch.set_printoptions(profile='full')
# print("Output: ", output)

import numpy as np
input = np.array([[0,0,0]])
output = np.array([[0,0,0]])
output = input
print(np.arange(-1.5,4.5))
print(np.exp(-16 / 4 * np.array([2,2,2]) ** 2))
print(-16 / 4 * 2 ** 2)
print(np.array([1,2,3])[::-1])

# Gaussian Filter
import numpy as np
from scipy.ndimage.filters import gaussian_filter
# from custome_filter import gaussian_filter
import scipy
import scipy.spatial
def gaussian_filter_density(gt):
    print (gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    
    pts = np.array(list(zip(np.nonzero(gt)[0], np.nonzero(gt)[1])))
    print("===========Index Of Non Zero Element==============")
    print(pts)
    print("==============================")
    # x = np.array([[3, 0, 0], [0, 4, 0], [5, 6, 0]])
    # array([[3, 0, 0],
    #        [0, 4, 0],
    #        [5, 6, 0]])
    # np.nonzero(x)
    # (array([0, 1, 2, 2]), array([0, 1, 0, 1])) --> index of non zero elemen
    
    
    
    # a = ("John", "Charles", "Mike")
    # b = ("Jenny", "Christy", "Monica")
    # x = zip(a, b)
    # print(list(x))
    # Output:
    # [('John', 'Jenny'), ('Charles', 'Christy'), ('Mike', 'Monica')]
    
    
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    # Query the kd-tree for nearest neighbors
    # k=4 -> for each point of pts, will be searched the 4 nearest neighbors
    distances, locations = tree.query(pts, k=4)
    print("===========Distance KNN==============")
    print(distances)
    print("===========Index Of Neighbor==============")
    print(locations)
    print ('generate density...')
    print("++++++++++++++++++++++++++++++++")
    for i, pt in enumerate(pts):
        
        print("iterasi-",str(i))
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[0],pt[1]] = 1.
        if (i < 2):
            print("===========PT2D==============")
            print(pt2d)
        
        if gt_count > 1:
            # distance ke-0 tidak diikutkan karena menunjuk ke titik itu sendiri
            # (jarak terpendek adalah dirinya sendiri)
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            if (i < 2):
                print("===========SIGMA==============")
                print(sigma)
            
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
            # np.array([3.,2])
            # np.average(np.array([3.,2])) -> Output : 2.5
           
        # np.array([1,2,3,4])+np.array([1,2,3,4])
        # array([2, 4, 6, 8])
        res = gaussian_filter(pt2d, sigma, mode='constant')
        # res = gaussian_filter(pt2d, sigma, nonZeroIndex=list([pt]), mode='constant')
        print("===========Density==============")
        print(res)
        density += res
        with open("C:\\Users\\Admin\\Desktop\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\debug\\density_ori.txt", "a+") as f:
          f.write(str(density)+"\n\n\n")
    
    print("++++++++++++++++++++++++++++++++")
    print ('done.')
    return density

gt = np.array([
    [0,1,0,0,0,0,0], 
    [0,0,0,0,1,0,0], 
    [0,0,1,0,0,0,0], 
    [1,0,0,0,0,0,1]])
print(gaussian_filter_density(gt))
# import inspect
# print("===========SOURCE============")
# print(inspect.getsource(gaussian_filter))

# https://scipy.github.io/devdocs/tutorial/ndimage.html
from scipy.ndimage import correlate1d
a = [0, 0, 0, 1, 0, 0, 0]
correlate1d(a, [1])
correlate1d(a, [1, 1])
correlate1d(a, [1, 1, 1])
correlate1d(a, [1, 1, 1, 1])
correlate1d(a, [1, 1, 1, 1, 1])
correlate1d(a, [1, 1, 1, 1, 1, 1, 1])
correlate1d(a, [0, 0, 0, 1, 0, 0, 0])

a = [0, 0, 1, 1, 1, 0, 0]
correlate1d(a, [-1, 1], origin = -1)  # forward difference
correlate1d(a, [0, -1, 1]) # forward difference

from scipy.ndimage import correlate1d
a = [[0, 0, 0, 1, 0, 0, 0],[0, 0, 0, 2, 0, 0, 0],[0, 0, 0, 1, 0, 0, 0]]
correlate1d(a, [1,1,1,1], axis=0, mode='constant')

# cara kerja correlate1d
# 1. array input akan diappend depan belakang dengan ukuran append sama dengan 
# panjang input array
# contoh :
# input = [1,0,0,1,0,0,1]
# di append menjadi input = [0,0,0,0,0,0,0|input|0,0,0,0,0,0,0]
# 2. input akan dikonvolusi dengan weight dengan stride 1,
# Penempatan hasil konvolusi:
# a. jika panjang array weight ganjil, maka hasil konvolusi akan ditempatkan 
#    di input pada indeks yang berada tepat di tengah weight
# b. jika panjang array weight genap, maka hasil konvolusi akan ditempatkan 
#    di input pada indeks yang berada di posisi int(panjang wieght / 2) + 1
#    dari weight
# c. penempatan hasil konvolusi dapat diubah dengan memasukkan parameter
#    origin yang relative terhadap poisisi tengah dari weight
#    contoh correlate1d(a, [-1, 1, 0], origin = -1) -> penempatan hasil 
#           convolusi diletakkan pada indkes wieght ke -> 
#           center of weight - 1 = 1 - 1 = 0


# =======================================================================================
# UNDERSTANDING Torch.Backward --> process to calculate gradient
# https://towardsdatascience.com/pytorch-autograd-understanding-the-heart-of-pytorchs-magic-2686cd94ec95#:~:text=is%20not%20needed.-,Backward()%20function,grad%20of%20every%20leaf%20node.
# =======================================================================================

import torch

# Creating the graph
x = torch.tensor(1.0, requires_grad = True)
y = torch.tensor(4.0, requires_grad= True)
z = y / x

# Displaying
for i, name in zip([x, y, z], "xyz"):
    print(f"{name}\ndata: {i.data}\nrequires_grad: {i.requires_grad}\n\
grad: {i.grad}\ngrad_fn: {i.grad_fn}\nis_leaf: {i.is_leaf}\n")

print("before backward:")
print(x.grad)
print(y.grad)
print(z.grad)
print("after backward:")
z.backward()
print(x.grad)
print(y.grad)
print(z.grad)


# ===================================================================================
# =========================Convert RGB===============================================
# ===================================================================================

# Old RGB Matrix
# [[[ 1  2  3  4  5]
#   [ 6  7  8  9 10]
#   [ 7  2  8  4  9]
#   [ 3  2  8  5  9]]

#  [[ 5  2  7  4  5]
#   [ 9  7  8  6  1]
#   [ 3  2 10  4  9]
#   [ 6  2  9  3  9]]

#  [[ 5  2  7  4  5]
#   [ 9  7  8  6 10]
#   [ 3  2  5  4  9]
#   [ 1  2  1  4  9]]]

# Valid RGB Matrix
# [[[0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]]

#  [[0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]]

#  [[0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]]

#  [[0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]
#   [0. 0. 0.]]]

import numpy as np
a = np.array([[[[1,2,3,4,5],[6,7,8,9,10],[7,2,8,4,9],[3,2,8,5,9]],[[5,2,7,4,5],[9,7,8,6,1],[3,2,10,4,9],[6,2,9,3,9]],[[5,2,7,4,5],[9,7,8,6,10],[3,2,5,4,9],[1,2,1,4,9]]]])
print(a.shape)
# (1, 3, 4, 5)
# (1, channel, height, width)
# a = a.reshape(a.shape[1],a.shape[2],a.shape[3])
# b = np.zeros((a.shape[1], a.shape[2], a.shape[0]))
matrix = []
for i in range(a.shape[2]):
    row = []
    for j in range(a.shape[3]):
        point = [a[0][0][i][j],a[0][1][i][j],a[0][2][i][j]]
        row.append(point)
    matrix.append(row)

matrix = np.array(matrix)
print(matrix.shape)
print(matrix)
            