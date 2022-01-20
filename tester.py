# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:39:40 2022

@author: Admin
"""

# ====================================================================
# prediction on single image
# ====================================================================
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
import torchvision.transforms.functional as F
from matplotlib import cm as c
from image import *
from model import CSRNet
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from inception_restnet_v2.inceptionresnetv2 import InceptionResNetV2
from constant import *

model = InceptionResNetV2()
model = model.cpu()
#loading the trained weights
checkpoint = torch.load("C:\\Users\\Admin\\Desktop\\data\\TA\\Projek\\Result\\Training_10_epoch_sgd\\0model_best.pth.tar")
model.load_state_dict(checkpoint['state_dict'])
transform=transforms.Compose([
                      transforms.ToTensor(),transforms.Normalize(
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
                  ])

fileName = "IMG_1"
img = transform(Image.open('C:\\Users\\Admin\\Desktop\\data\\TA\\Projek\\Dataset\\ShanghaiTech\\part_B\\test_data\\images\\'+fileName+'.jpg').convert('RGB')).cpu()
gt_file = h5py.File("C:\\Users\\Admin\\Desktop\\data\\TA\\Projek\\Dataset\\ShanghaiTech\\part_B\\test_data\\ground-truth\\"+fileName+".h5",'r')
groundtruth = np.asarray(gt_file['density'])

output = model(img.unsqueeze(0))
print("Original Count : ", int(np.sum(groundtruth)))
print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
fig = plt.figure()
fig_ground = fig.add_subplot(121)  # left side
fig_estimate = fig.add_subplot(122) # right side
fig_ground.imshow(groundtruth, cmap = c.jet)
fig_estimate.imshow(temp,cmap = c.jet)
plt.show()
 