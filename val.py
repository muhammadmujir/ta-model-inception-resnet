# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 10:32:19 2021

@author: Admin
"""

#importing libraries
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
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
from inception_restnet_v2.inceptionresnetv2 import InceptionResNetV2
from constant import *

transform=transforms.Compose([
                      transforms.ToTensor(),transforms.Normalize(
                          mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
                  ])
isCudaAvailable = True

#defining the location of dataset
#root = 'C:\\Users\\Admin\\Desktop\\Kuliah\\TA\\ShanghaiTech\\'
root = BASE_PATH
# part_A_train = os.path.join(root,'part_A\\train_data','images')
# part_A_test = os.path.join(root,'part_A\\test_data','images')
# part_B_train = os.path.join(root,'part_B\\train_data','images')
# part_B_test = os.path.join(root,'part_B\\test_data','images')
part_A_train = os.path.join(root,DATASET1_TRAIN_A)
part_A_test = os.path.join(root,DATASET1_TEST_A)
part_B_train = os.path.join(root,DATASET1_TRAIN_B)
part_B_test = os.path.join(root,DATASET1_TEST_B)
path_sets = [part_A_test]

#defining the image path
img_paths = []
for path in path_sets:
    for img_path in glob.glob(os.path.join(path, '*.jpg')):
       img_paths.append(img_path)

#model = CSRNet()
model = InceptionResNetV2()

#defining the model
if (isCudaAvailable):
    model = model.cuda()
else:
    model = model.cpu()
#loading the trained weights
checkpoint = torch.load(CHECKPOINT_PATH+'0model_best.pth.tar')
model.load_state_dict(checkpoint['state_dict'])

mae = 0
for i in tqdm(range(len(img_paths))):
    img = None
    if (isCudaAvailable):
        img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    else:
        img = transform(Image.open(img_paths[i]).convert('RGB')).cpu()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground-truth'),'r')
    groundtruth = np.asarray(gt_file['density'])
    output = model(img.unsqueeze(0))
    if (isCudaAvailable):
        mae += abs(output.detach().cuda().sum().numpy()-np.sum(groundtruth))
    else:
        mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
print ("MAE : ",mae/len(img_paths))


# prediction on single image
# from matplotlib import cm as c
# img = transform(Image.open('C:\\Users\\Admin\\Desktop\\Kuliah\\TA\\ShanghaiTech\\part_B\\test_data\\images\\IMG_1.jpg').convert('RGB')).cpu()

# output = model(img.unsqueeze(0))
# print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
# temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
# plt.imshow(temp,cmap = c.jet)
# plt.show()



# temp = h5py.File('part_A/test_data/ground-truth/IMG_100.h5', 'r')
# temp_1 = np.asarray(temp['density'])
# plt.imshow(temp_1,cmap = c.jet)
# print("Original Count : ",int(np.sum(temp_1)) + 1)
# plt.show()
# print("Original Image")
# plt.imshow(plt.imread('part_A/test_data/images/IMG_100.jpg'))
# plt.show()