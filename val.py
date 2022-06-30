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
from inception_restnet_v2.inceptionresnetv2 import InceptionResNetV2
from constant import *
import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
from dataloader import DataLoader
import dataset

parser = argparse.ArgumentParser(description='Model Testing')
parser.add_argument('img_path', metavar='TEST_IMAGE', help='path to testing image')
parser.add_argument('gpu',metavar='GPU', type=str, help='GPU id to use.')
parser.add_argument('best_result_count', type=int, metavar='BEST_RESULT_COUNT', help='best result count')
parser.add_argument('--large-file', action='store_true', help='enable resize and crop for large file')
parser.add_argument('--crop', action='store_true', help='option to crop image')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

args = parser.parse_args()

def toDevice(tens):
    global args
    if args.gpu != 'None':
        tens = tens.cuda()
    else:
        tens = tens.cpu()
    return tens
    
def main():
    global args
    img_path = args.img_path
    best_result_count = args.best_result_count
    isCudaAvailable = True if args.gpu != 'None' else False 
    maeByCount = 0.0
    maeByPixel = 0.0
    bestMaeResult = []
    bestPixelMaeResult = []
    pathResult = []
    bestOutputDensity = []
    model = CSRNet().cuda() if isCudaAvailable else CSRNet().cpu()
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
    else:
        print("Checkpoint Not Set")     
    model.eval()
    maeCriterion = nn.L1Loss(size_average=False).cuda() if isCudaAvailable else nn.L1Loss(size_average=False).cpu()
    paths = glob.glob(os.path.join(img_path, '*.jpg'))
    test_loader = DataLoader(
    dataset.listDataset(paths,
                   shuffle=False,
                   isLargeSize=args.large_file,
                   isCrop=args.crop,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)
    
    for i,(img, target, path) in enumerate(test_loader):
        img = toDevice(img)
        img = Variable(img)
        output = model(img)
        target = toDevice(target.type(torch.FloatTensor).unsqueeze(0))
        target = Variable(target)
        mae = abs(output.data.sum()-toDevice(target.sum().type(torch.FloatTensor)))
        maeByCount += mae
        pixelMae = maeCriterion(output, target).item()
        maeByPixel += pixelMae
        
        if len(bestMaeResult) < best_result_count:
            bestMaeResult.append(mae)
            bestPixelMaeResult.append(pixelMae)
            pathResult.append(path)
            bestOutputDensity.append(output)
        else:
            indexOfMaxVal = bestMaeResult.index(max(bestMaeResult)) 
            if mae < max(bestMaeResult):
                bestMaeResult[indexOfMaxVal] = mae
                bestPixelMaeResult[indexOfMaxVal] = pixelMae
                pathResult[indexOfMaxVal] = path
                bestOutputDensity[indexOfMaxVal] = output
    print ("AVG MAE : ",maeByCount.item()/len(paths))
    print ("AVG MAE BY PIXEL: ", maeByPixel/len(paths))
    for (i, path) in enumerate(pathResult):
        path = path[0]
        print(path)
        # plt.figure()
        # plt.imshow(plt.imread(path))
        # temp = np.asarray(h5py.File(path.replace('.jpg','.h5').replace('images','ground_truth'), 'r')['density'])
        # plt.figure()
        # plt.imshow(temp,cmap = cm.jet)
        outputDensity = bestOutputDensity[i].detach().cpu()
        print("Output Density: ", outputDensity.shape)
        outputDensity = outputDensity.reshape(outputDensity.shape[2], outputDensity.shape[3])
        print("After Output Density: ", outputDensity.shape)
        temp = np.asarray(outputDensity)
        plt.figure()
        plt.imshow(temp,cmap = cm.jet)
        plt.show()
                
def valManyImages():
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
    checkpoint = torch.load(CHECKPOINT_PATH+'model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    
    mae = 0
    for i in tqdm(range(len(img_paths))):
        img = None
        if (isCudaAvailable):
            img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
        else:
            img = transform(Image.open(img_paths[i]).convert('RGB')).cpu()
        gt_file = h5py.File(img_paths[i].replace('.jpg','.h5').replace('images','ground-truth'),'r')
        target = np.asarray(gt_file['density'])
        output = model(img.unsqueeze(0))
        if (isCudaAvailable):
            mae += abs(output.detach().cuda().sum().numpy()-np.sum(target))
        else:
            mae += abs(output.detach().cpu().sum().numpy()-np.sum(target))
    print ("MAE : ",mae/len(img_paths))

def valSingleImage():
    # prediction on single image
    from matplotlib import cm as c
    img = transform(Image.open('C:\\Users\\Admin\\Desktop\\Kuliah\\TA\\ShanghaiTech\\part_B\\test_data\\images\\IMG_1.jpg').convert('RGB')).cpu()
    
    output = model(img.unsqueeze(0))
    print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
    plt.imshow(temp,cmap = c.jet)
    plt.show()

def checkSimilarity():
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
    main() 