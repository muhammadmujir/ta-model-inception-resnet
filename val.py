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
from matplotlib import cm
from image import *
from model import CSRNet
import torch
from torchvision import datasets, transforms
from tqdm import tqdm
# from inception_restnet_v2.inceptionresnetv2 import InceptionResNetV2
from model import CSRNet
from constant import *
import cv2

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
# img_paths = []
# for path in path_sets:
#     for img_path in glob.glob(os.path.join(path, '*.jpg')):
#        img_paths.append(img_path)

# #model = CSRNet()
# model = InceptionResNetV2()

# #defining the model
# if (isCudaAvailable):
#     model = model.cuda()
# else:
#     model = model.cpu()
# #loading the trained weights
# checkpoint = torch.load(CHECKPOINT_PATH+'0model_best.pth.tar')
# model.load_state_dict(checkpoint['state_dict'])

# mae = 0
# for i in tqdm(range(len(img_paths))):
#     img = None
#     if (isCudaAvailable):
#         img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
#     else:
#         img = transform(Image.open(img_paths[i]).convert('RGB')).cpu()
#     gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground-truth'),'r')
#     groundtruth = np.asarray(gt_file['density'])
#     output = model(img.unsqueeze(0))
#     if (isCudaAvailable):
#         mae += abs(output.detach().cuda().sum().numpy()-np.sum(groundtruth))
#     else:
#         mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))
# print ("MAE : ",mae/len(img_paths))

model = None

def initModel(modelPath, isCudaAvailable):
    #defining the model
    global model
    model = CSRNet()
    if (isCudaAvailable):
        model = model.cuda()
    else:
        model = model.cpu()
    #loading the trained weights
    checkpoint = torch.load(modelPath)
    model.load_state_dict(checkpoint['state_dict'])
    
def predictCount(imagePath, groundTruth):
    global transform
    
    gtDensity = np.asarray(h5py.File(groundTruth, 'r')['density'])
    img = transform(Image.open(imagePath).convert('RGB')).cpu()
    output = model(img.unsqueeze(0))
    
    print("GroundTruth Count : ",int(np.sum(gtDensity)))
    print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
    gtDensity = cv2.resize(gtDensity,(int(gtDensity.shape[1]//8),int(gtDensity.shape[0]//8)),interpolation = cv2.INTER_CUBIC)*64
    # print("GroundTruth Count After Risze: ",int(np.sum(gtDensity)))
    # writeOutput(torch.tensor(gtDensity), "C:\\Users\\Admin\\Desktop\\blibli\\ground_truth_density.csv")
    writeOutput(torch.tensor(gtDensity), "C:\\Users\\Admin\\Desktop\\blibli\\resized_gt_density.csv")
    # writeOutput(output, "C:\\Users\\Admin\\Desktop\\blibli\\output_truth_density.csv")
    plt.imshow(gtDensity,cmap = cm.jet)
    plt.show()
    outputPlot = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
    plt.imshow(outputPlot,cmap = cm.jet)
    plt.show()

def writeOutput(data, file):
    resultCSV = open(os.path.join(file), 'w')
    for row in data:
        for col in row:
            resultCSV.write('%s;' % str(col.item()))
        print("row")
        resultCSV.write('\n')
    
def checkGroundTruthSame():
    mat = io.loadmat("D:\\TA\Dataset\\ShanghaiTech-Sample\\ShanghaiTech\\part_B\\test_data\\ground-truth\\GT_IMG_281.mat")
    print("Ground Truth Count", len(mat['image_info'][0][0][0][0][0]))
    temp = h5py.File('D:\\TA\Dataset\\ShanghaiTech-Sample\\ShanghaiTech\\part_B\\test_data\\ground-truth-h5\\IMG_281.h5', 'r')
    temp_1 = np.asarray(temp['density'])
    temp2 = h5py.File('C:\\Users\\Admin\\Desktop\\TA\\Dataset\\ShanghaiTech\\part_B\\test_data\\ground_truth\\IMG_281.h5', 'r')
    temp_2 = np.asarray(temp2['density'])
    isSame = temp_1 - temp_2
    print(len(np.nonzero(isSame)[0]) == 0)
    plt.imshow(temp_1,cmap = CM.jet)
    print("After Gaussian Count : ",int(np.sum(temp_1)))
    print(len(np.nonzero(temp_1)[1]))
    plt.show()
    print("Original Image")
    plt.imshow(plt.imread('C:\\Users\\Admin\\Desktop\\TA\\Dataset\\ShanghaiTech\\part_B\\test_data\\images\\IMG_281.jpg'))
    plt.show()

if __name__ == '__main__':
    initModel("C:\\Users\\Admin\\Desktop\\TA\\Dataset\\Result\\model_best.pth.tar", False)
    predictCount("D:\\TA\\Dataset\\ShanghaiTech\\part_B\\train_data\\images\\IMG_5.jpg", "D:\\TA\\Dataset\\ShanghaiTech\\part_B\\train_data\\ground-truth\\IMG_5.h5")
    