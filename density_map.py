# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 15:00:37 2022

@author: Admin
"""
import h5py
import glob
import os
import cv2
import numpy as np
from PIL import Image
from matplotlib import cm
from matplotlib import pyplot as plt
import scipy.io as io

def resizeImage(src, dest, delimiter = "\\"):
    # max dimens = (width, height) = (2222,3333)
    for img_path in glob.glob(os.path.join(src, '*.jpg')):
        img = Image.open(img_path).convert('RGB')
        ratio = 1
        if img.size[0] >= 6000 or img.size[1] >= 6000:
            ratio = 3
        elif img.size[0] >= 3000 or img.size[1] >= 3000:
            ratio = 2
        if ratio > 1:
            img = img.resize((int(img.size[0]//ratio), int(img.size[1]//ratio)))
        filename = img_path.split(delimiter)
        filename = filename[len(filename)-1]
        img.save(os.path.join(dest, filename))
        print(filename)
        
def resizeDensityMap(src, dest, delimiter = "\\"):
    for gt_path in glob.glob(os.path.join(src, '*.h5')):
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
        ratio = 1
        if target.shape[0] >= 6000 or target.shape[1] >= 6000:
            ratio = 3
        elif target.shape[0] >= 3000 or target.shape[1] >= 3000:
            ratio = 2
        if ratio > 1:
            target = cv2.resize(target,(int(target.shape[1]//ratio),int(target.shape[0]//ratio)),interpolation = cv2.INTER_CUBIC)*(ratio*ratio)
        filename = gt_path.split(delimiter)
        filename = filename[len(filename)-1]
        with h5py.File(os.path.join(dest, filename), 'w') as hf:
            hf['density'] = target
        print(filename)

def resizeDensityMap(src, dest, delimiter = "\\"):
    for gt_path in glob.glob(os.path.join(src, '*.h5')):
        gt_file = h5py.File(gt_path)
        target = np.asarray(gt_file['density'])
        ratio = 1
        if target.shape[0] >= 6000 or target.shape[1] >= 6000:
            ratio = 3
        elif target.shape[0] >= 3000 or target.shape[1] >= 3000:
            ratio = 2
        if ratio > 1:
            target = cv2.resize(target,(int(target.shape[1]//ratio),int(target.shape[0]//ratio)),interpolation = cv2.INTER_CUBIC)*(ratio*ratio)
        filename = gt_path.split(delimiter)
        filename = filename[len(filename)-1]
        with h5py.File(os.path.join(dest, filename), 'w') as hf:
            hf['density'] = target
        print(filename)


def showDensityMap(file, matFile):
    density = h5py.File(file, 'r')
    density = np.asarray(density['density'])
    plt.imshow(density,cmap = cm.jet)
    plt.show()
    if matFile != None:
        mat = io.loadmat(matFile)["annPoints"]
        print("Original Count", len(mat))
    print("Crowd Count : ",int(np.sum(density)))
    
if __name__ == '__main__':
    # resizeDensityMap("D:\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\ground_truth", "D:\\TA\\Dataset\\UCF_QNRF_RESIZED\\Train\\ground_truth")
    # resizeImage("D:\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\images", "D:\\TA\\Dataset\\UCF_QNRF_RESIZED\\Train\\images")
    # showDensityMap("D:\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\ground_truth\\img_0001.h5", 
    #                 "D:\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\ground_truth\\img_0001_ann.mat")
    showDensityMap("D:\\TA\\Dataset\\UCF_QNRF_RESIZED\\Train\\ground_truth\\img_0001.h5", None)