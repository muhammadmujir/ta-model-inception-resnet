# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 06:15:17 2021

@author: Admin
"""

# importing libraries

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
# from scipy.ndimage.filters import gaussian_filter
from custome_filter import gaussian_filter
import scipy
import scipy.spatial
import json
from matplotlib import cm as CM
from image import *
import torch
from tqdm import tqdm

# function to create density maps for images
def gaussian_filter_density(img_shape, pts):
    print("image shape", img_shape)
    # density = np.zeros(gt.shape, dtype=np.float32)
    density = np.zeros(img_shape, dtype=np.float32)
    # gt_count = np.count_nonzero(gt)
    gt_count = pts.shape[0]
    if gt_count == 0:
        return density
    
    # gt[i][0] -> Width -> column in matrix
    # gt[i][1] -> Height -> row in matrix
    # pts = np.array(list(zip(np.nonzero(gt)[0], np.nonzero(gt)[1])))
    
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

    print ('generate density...')
    print("++++++++++++++++++++++++++++++++")
    for i, pt in enumerate(pts):
        
        # print("iterasi-",str(i))
        # pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d = np.zeros(img_shape, dtype=np.float32)
        pt2d[pt[0],pt[1]] = 1.
        # print(pt[0], " :: ", pt[1])
        
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            #print(distances)
            
        else:
            # sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
            sigma = np.average(np.array(img_shape))/2./2. #case: 1 point
            # np.array([3.,2])
            # np.average(np.array([3.,2])) -> Output : 2.5
           
        # np.array([1,2,3,4])+np.array([1,2,3,4])
        # array([2, 4, 6, 8])
        # density += gaussian_filter(pt2d, sigma, mode='constant')
        density += gaussian_filter(pt2d, sigma, nonZeroIndex=list([pt]), mode='constant')
    
    print("++++++++++++++++++++++++++++++++")
    print ('done.')
    return density

# Local Machine
# root = 'C:\\Users\\Admin\\Desktop\\Kuliah\\TA\\ShanghaiTech\\'
# part_A_train = os.path.join(root,'part_A\\train_data','images')
# part_A_test = os.path.join(root,'part_A\\test_data','images')
# part_B_train = os.path.join(root,'part_B\\train_data','images')
# part_B_test = os.path.join(root,'part_B\\test_data','images')
# path_sets = [part_A_train,part_A_test,part_B_train,part_B_test]

# Google Collab
# root = 'drive/MyDrive/Projek Akhir/dataset/ShanghaiTech/'
# part_A_train = os.path.join(root,'part_A/train_data_full','images')
# part_A_test = os.path.join(root,'part_A/test_data_full','images')
# part_B_train = os.path.join(root,'part_B/train_data_full','images')
# part_B_test = os.path.join(root,'part_B/test_data_full','images')
# #path_sets = [part_A_train,part_A_test,part_B_train,part_B_test]
# path_sets = [part_B_test]

#local UCF-QNRF
root = 'C:\\Users\\Admin\\Desktop\\TA\\Dataset\\UCF-QNRF_ECCV18\\'
path_train = os.path.join(root,'Train','images')
path_test = os.path.join(root,'Test','images')
#path_sets = [part_A_train,part_A_test,part_B_train,part_B_test]
path_sets = [path_train,path_test]

# path1 = "C:\\Users\\Admin\\Desktop\\Kuliah\\TA\\ShanghaiTech\\part_A\\train_data\\images\\IMG_21.jpg"
# path_sets = [path1]

def create_ground_truth(path_sets):
    img_paths = []
    
    # for path in path_sets:
    #     for img_path in range(830,831):
    #         img_paths.append(path+"\\img_0"+str(img_path)+".jpg")
    
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    
    for img_path in img_paths:
        print (img_path)
        # mat = io.loadmat(img_path.replace('.jpg','_ann.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
        mat = io.loadmat(img_path.replace('.jpg','_ann.mat').replace('images','ground-truth'))
        img = plt.imread(img_path)
        # np.zeros((row, column))
        # k = np.zeros((img.shape[0],img.shape[1]))
        # gt = mat["image_info"][0,0][0,0][0]
        gt = np.array(mat["annPoints"])
        # original position of ground truth annotation is 
        # [[col, row], [col, row], ...]
        # so, we need to flip in order to make ground truth position
        # like an usual array structure -> [[row, col], [row, col], ...]
        gt = np.flip(gt, (1))
        # img.shape[0] -> height/row
        # img.shape[1] -> width/column
        # ground-truth annotation -> [[width atau column, height atau row],[width atau column, height atau row]]
        print(img.shape[0]," :: ",img.shape[1])
        # print("GT ", mat)
        #print("==========================================")
        # pointCount = 0;
        # totalPoint = 0;
        
        invalidPoints = []
        for i in range(0,len(gt)):
            if int(gt[i][0]) > img.shape[0] and int(gt[i][1]) > img.shape[1]:
                
                # k[int(gt[i][0]),int(gt[i][1])] = 1
                invalidPoints.append(i)
                # print(gt[i][0], " :: ", gt[i][1])
                # pointCount = pointCount+1
            # totalPoint = totalPoint+1
        #print("Total Point: ", totalPoint)
        #print("POINT COUNT: ", pointCount)
        #print("==========================================")
        if (len(invalidPoints) > 0):
            print("Invalid")
            gt = np.delete(gt, invalidPoints, axis=0)
        # k = gaussian_filter_density(k)
        with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground-truth'), 'w') as hf:
            gt = gt.astype(int, copy=False)
            hf['density'] = gaussian_filter_density((img.shape[0], img.shape[1]), gt)



# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# print(img_paths[8])
def countCrowd():
    gt_file = h5py.File(img_paths[0].replace('.jpg','.h5').replace('images','ground-truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    print(np.sum(groundtruth))

def isArrayEqual(path_sets):
    img_paths = []
    
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.h5')):
            img_paths.append(img_path)
    
    print(img_paths)
    groundtruth = []
    for i, pth in enumerate(img_paths):
        gt_file = h5py.File(pth,'r')
        groundtruth.append(gt_file['density'])
        
    groundtruth = np.array(groundtruth)
    # print(groundtruth[0][932][913], "::", groundtruth[1][932][913])
    # print(groundtruth[0][932][914], "::", groundtruth[1][932][914])
    groundtruth = groundtruth[0] - groundtruth[1]
    print(np.nonzero(groundtruth))
    print("shape: ",groundtruth.shape)
    return np.nonzero(groundtruth)[0].shape[0] == 0
    
# Compare Array
def compareArray():
    paths = ["C:\\Users\\Admin\\Desktop\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\debug\\"]
    print(isArrayEqual(paths))

# Try Gaussian Filter
def tryGaussianFilter():
    paths = ["C:\\Users\\Admin\\Desktop\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\debug\\images"]
    create_ground_truth(paths)

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++



###################################
# Gaussian Filter #
###################################
# a = np.arange(50, step=2).reshape((5,5))
# print(a)
# gaussian_filter(a, sigma=1)

# input
# [[ 0  2  4  6  8]
#  [10 12 14 16 18]
#  [20 22 24 26 28]
#  [30 32 34 36 38]
#  [40 42 44 46 48]]

#output (after gaussion filter)
# array([[ 4,  6,  8,  9, 11],
#        [10, 12, 14, 15, 17],
#        [20, 22, 24, 25, 27],
#        [29, 31, 33, 34, 36],
#        [35, 37, 39, 40, 42]])




# Analyzing
def analyze():
    import sys
    import numpy
    numpy.set_printoptions(threshold=sys.maxsize)
    
    img_path = "C:\\Users\\Admin\\Desktop\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\images\\"
    # print (img_path)
    # mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground-truth').replace('IMG_','GT_IMG_'))
    mat = io.loadmat(img_path.replace('images','ground-truth')+"img_0004_ann.mat")
    img = plt.imread(img_path+"img_0004.jpg")
    # print("Image shape: "+str(img.shape))
    # k = np.zeros((img.shape[0],img.shape[1]))
    # gt = mat["image_info"][0,0][0,0][0]
    # print(mat['annPoints'])
    np_array = numpy.array(mat['annPoints'])
    print(numpy.amax(np_array, axis=0))
    # save array into file
    # new_array = numpy.array(mat)
    # with open("C:\\Users\\Admin\\Desktop\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\debug\\img_0004_ann.txt", "w+") as f:
    #   data = f.read()
    #   f.write(str(new_array))
      
    # for i in range(0,len(gt)):
    #     if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
    #         k[int(gt[i][1]),int(gt[i][0])]=1
    
    # print(k)
    # k = gaussian_filter_density(k)
    # print("after : ", k)
    # with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground-truth'), 'w') as hf:
    #         hf['density'] = k

# Try KdTree
def tryKdTree():
    import numpy as np
    from scipy.spatial import KDTree
    x, y = np.mgrid[0:5, 2:8]
    print(x)
    print("=================")
    print(y)
    print("=================")
    tree = KDTree(np.c_[x.ravel(), y.ravel()])
    print(x.ravel())
    print("=================")
    print(y.ravel())
    print("=================")
    print(np.c_[x.ravel(), y.ravel()])
    print("=================")
    dd, ii = tree.query([[0, 0], [2.2, 2.9]], k=1)
    print(dd)
    print("=================")
    print(ii)
    print("=================")
    # print(dd, ii, sep='\n')
    # [0,0] -> indeks nearest neigbor = 0 -> (0,2) -> ((0-0)^2 + (0-2)^2)^-2 = (4)^-2 = 2
    # [2.2,2.9] -> indeks nearest neigbor = 13 -> (2,3) -> ((2.2-2)^2 + (2.9-3)^2)^-2 = (0.05)^-2 = 0.223
    
    a = [1,2,3]
    b = [1,2,3]
    axes = [0,1]
    axes = [(a[ii], b[ii]) for ii in range(len(axes)) if a[ii] > 0]
    print(axes)
    
if __name__ == '__main__':
    create_ground_truth(path_sets)