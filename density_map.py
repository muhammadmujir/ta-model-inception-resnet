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

def cal_new_size(im_h, im_w, min_size, max_size):
    if im_h < im_w:
        if im_h < min_size:
            ratio = 1.0 * min_size / im_h
            im_h = min_size
            im_w = round(im_w*ratio)
        elif im_h > max_size:
            ratio = 1.0 * max_size / im_h
            im_h = max_size
            im_w = round(im_w*ratio)
        else:
            ratio = 1.0
    else:
        if im_w < min_size:
            ratio = 1.0 * min_size / im_w
            im_w = min_size
            im_h = round(im_h*ratio)
        elif im_w > max_size:
            ratio = 1.0 * max_size / im_w
            im_w = max_size
            im_h = round(im_h*ratio)
        else:
            ratio = 1.0
    return im_h, im_w, ratio


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

def resizeDensityMap2(src, dest, delimiter = "\\"):
    min_size = 512
    max_size = 2048
    for phase in ['train','val']:
        f = open('{}.txt'.format(phase))
        new_dest = os.path.join(dest, phase+"-density")
        for filename in f:
            gt_path = os.path.join(src, filename.strip()) 
            gt_file = h5py.File(gt_path)
            target = np.asarray(gt_file['density'])
            img_h, img_w, ratio = cal_new_size(target.shape[0], target.shape[1], min_size, max_size)
            target = cv2.resize(target,(int(target.shape[1]*ratio),int(target.shape[0]*ratio)),interpolation = cv2.INTER_CUBIC)//(ratio*ratio)
            filename = gt_path.split(delimiter)
            filename = filename[len(filename)-1]
            with h5py.File(os.path.join(new_dest, filename), 'w') as hf:
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
    resizeDensityMap2("D:\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\ground_truth", "D:\\TA\\Dataset\\UCF_QNRF_FOR_BAYESIAN\\Preprocessed")
    # resizeDensityMap("D:\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\ground_truth", "D:\\TA\\Dataset\\UCF_QNRF_RESIZED\\Train\\ground_truth")
    # resizeImage("D:\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\images", "D:\\TA\\Dataset\\UCF_QNRF_RESIZED\\Train\\images")
    # showDensityMap("D:\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\ground_truth\\img_0001.h5", 
    #                 "D:\\TA\\Dataset\\UCF-QNRF_ECCV18\\Train\\ground_truth\\img_0001_ann.mat")
    # showDensityMap("D:\\TA\\Dataset\\UCF_QNRF_RESIZED\\Train\\ground_truth\\img_0001.h5", None)