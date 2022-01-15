# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 14:39:40 2022

@author: Admin
"""

import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2
from matplotlib import pyplot as plt
from matplotlib import cm as c

# =============================================================
# Print Density
# =============================================================

gt_path = "C:\\Users\\Admin\\Desktop\\data\\TA\\Projek\\Dataset\\ShanghaiTech\\part_A\\train_data_full\\ground-truth\\IMG_2.h5"
gt_file = h5py.File(gt_path, 'r')
target = np.asarray(gt_file['density'])

plt.imshow(target,cmap = c.jet)
print("Original Count : ",int(np.sum(target)) + 1)
plt.show()
 