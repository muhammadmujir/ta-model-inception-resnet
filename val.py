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
from image import *
from utils import saveLargeListIntoCSV

parser = argparse.ArgumentParser(description='Model Testing')
parser.add_argument('img_path', metavar='TEST_IMAGE', help='path to testing image')
parser.add_argument('gpu',metavar='GPU', type=str, help='GPU id to use.')
parser.add_argument('best_result_count', type=int, metavar='BEST_RESULT_COUNT', help='best result count')
parser.add_argument('--print-freq', '-pf', type=int, default=50, metavar='PRINT_FREQ', help='print frequency')
parser.add_argument('--large-file', action='store_true', help='enable resize and crop for large file')
parser.add_argument('--crop', action='store_true', help='option to crop image')
parser.add_argument('--print-best', action='store_true', help='choose which result will be printed')
parser.add_argument('--print-all', action='store_true', help='print all result')
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

def convertRGBShape(img):
    # input shape: (3,width,height)
    # return shape: (height,width,3)
    matrix = []
    for i in range(img.shape[1]):
        row = []
        for j in range(img.shape[2]):
            point = [img[0][i][j],img[1][i][j],img[2][i][j]]
            row.append(point)
        matrix.append(row)
    return np.array(matrix)
    
def main():
    global args
    img_path = args.img_path
    best_result_count = args.best_result_count
    isCudaAvailable = True if args.gpu != 'None' else False 
    maeByCount = 0.0
    # maeByPixel = 0.0
    
    pathBestResult = []
    cropBestResult = []
    pathWorstResult = []
    cropWorstResult = []
    
    bestMaeResult = []
    worstMaeResult = []
    # bestPixelMaeResult = []
    # worstPixelMaeResult = []
    
    # bestImage = []
    # bestTargetDensity = []
    bestOutputDensity = []
    worstOutputDensity = []
    
    # bestOutputSum = []
    # bestTargetSum = []
    # worstOutputSum = []
    # worstTargetSum = []
    transform = transforms.Compose([transforms.ToTensor()])
    
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
    # maeCriterion = nn.L1Loss(size_average=False).cuda() if isCudaAvailable else nn.L1Loss(size_average=False).cpu()
    paths = glob.glob(os.path.join(img_path, '*.jpg'))
    countList = np.load(glob.glob(os.path.join(img_path, '*.npy'))[0])
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
    
    for i,(img, target, path, dx, dy) in enumerate(test_loader):
        if i % args.print_freq == 0:
            print("Iterasi {} : {}".format(i,path))
        img = toDevice(img)
        img = Variable(img)
        output = model(img)
        # target = toDevice(target.type(torch.FloatTensor).unsqueeze(0))
        # target = Variable(target)
        mae = abs(output.data.sum()-countList[i])
        maeByCount += mae
        # pixelMae = maeCriterion(output, target).item()
        # maeByPixel += pixelMae
        
        if len(bestMaeResult) < best_result_count:
            pathBestResult.append(path)
            cropBestResult.append((dx.item(),dy.item()))
            bestMaeResult.append(mae)
            # bestPixelMaeResult.append(pixelMae)
            # bestImage.append(img)
            # bestTargetDensity.append(target)
            bestOutputDensity.append(output)
            # bestOutputSum.append(output.data.sum())
            # bestTargetSum.append(toDevice(target.sum().type(torch.FloatTensor)))
        else:
            indexOfMaxMae = bestMaeResult.index(max(bestMaeResult)) 
            if mae < max(bestMaeResult):
                pathBestResult[indexOfMaxMae] = path
                cropBestResult[indexOfMaxMae] = (dx.item(),dy.item())
                bestMaeResult[indexOfMaxMae] = mae
                # bestPixelMaeResult[indexOfMaxMae] = pixelMae
                # bestImage[indexOfMaxMae] = img
                # bestTargetDensity[indexOfMaxMae] = target
                bestOutputDensity[indexOfMaxMae] = output
                # bestOutputSum[indexOfMaxMae] = output.data.sum()
                # bestTargetSum[indexOfMaxMae] = toDevice(target.sum().type(torch.FloatTensor))
                
        if len(pathWorstResult) < best_result_count:
            pathWorstResult.append(path)
            cropWorstResult.append((dx.item(),dy.item()))
            worstMaeResult.append(mae)
            # worstPixelMaeResult.append(pixelMae)
            worstOutputDensity.append(output)
            # worstOutputSum.append(output.data.sum())
            # worstTargetSum.append(toDevice(target.sum().type(torch.FloatTensor)))
        else:
            indexOfMinMae = worstMaeResult.index(min(worstMaeResult))
            if mae > min(worstMaeResult):
                pathWorstResult[indexOfMinMae] = path
                cropWorstResult[indexOfMinMae] = (dx.item(),dy.item())
                worstMaeResult[indexOfMinMae] = mae
                # worstPixelMaeResult[indexOfMinMae] = pixelMae
                worstOutputDensity[indexOfMinMae] = output
                # worstOutputSum[indexOfMinMae] = output.data.sum()
                # worstTargetSum[indexOfMinMae] = toDevice(target.sum().type(torch.FloatTensor))
        
                
    print ("AVG MAE : ",maeByCount.item()/len(paths))
    # print ("AVG MAE BY PIXEL: ", maeByPixel/len(paths))
    print("Original Image - Target Density Map - Predicted Density Map")
    
    if args.print_all or args.print_best:
        print("---------------------------Best-----------------------------")
        for (i, path) in enumerate(pathBestResult):
            path = path[0]
            print(path)
            # img = bestImage[i].detach().cpu()
            img, target, dx, dy = load_data(path, isCrop=args.crop, dx=cropBestResult[i][0], dy=cropBestResult[i][1]) if not args.large_file else load_data_ucf(path, isCrop=args.crop, dx=cropBestResult[i][0], dy=cropBestResult[i][1])
            print("Output Sum: ", bestOutputDensity[i].data.sum().item())
            # print("Target Sum: ", target.sum())
            print("Target Sum: ", countList[i])
            print("BASED COUNT MAE: ", bestMaeResult[i].item())
            # print("BASED PIXEL MAE: ", bestPixelMaeResult[i])
            plt.figure()        
            img = Variable(toDevice(transform(img)))
            img = img.detach().cpu()
            img = convertRGBShape(img)
            plt.imshow(img)
            plt.figure()
            plt.imshow(target,cmap = cm.jet)
            outputDensity = bestOutputDensity[i].detach().cpu()
            outputDensity = outputDensity.reshape(outputDensity.shape[2], outputDensity.shape[3])
            temp = np.asarray(outputDensity)
            plt.figure()
            plt.imshow(temp,cmap = cm.jet)
            plt.show()
    
    if args.print_all or not args.print_best:
        print("---------------------------Worst-----------------------------")
        for (i, path) in enumerate(pathWorstResult):
            path = path[0]
            print(path)
            # img = bestImage[i].detach().cpu()
            img, target, dx, dy = load_data(path, isCrop=args.crop, dx=cropWorstResult[i][0], dy=cropWorstResult[i][1]) if not args.large_file else load_data_ucf(path, isCrop=args.crop, dx=cropWorstResult[i][0], dy=cropWorstResult[i][1])
            print("Output Sum: ", worstOutputDensity[i].data.sum().item())
            print("Target Sum: ", target.sum())
            print("BASED COUNT MAE: ", worstMaeResult[i].item())
            # print("BASED PIXEL MAE: ", worstPixelMaeResult[i])
            plt.figure()     
            img = Variable(toDevice(transform(img)))
            img = img.detach().cpu()
            img = convertRGBShape(img)
            plt.imshow(img)
            plt.figure()
            plt.imshow(target,cmap = cm.jet)
            outputDensity = worstOutputDensity[i].detach().cpu()
            outputDensity = outputDensity.reshape(outputDensity.shape[2], outputDensity.shape[3])
            temp = np.asarray(outputDensity)
            plt.figure()
            plt.imshow(temp,cmap = cm.jet)
            plt.show()
    plt.close('all')
              
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

def valSingleImage(modelPath, imgPath, usingCuda = False):
    # prediction on single image
    from matplotlib import cm as c
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])
    model = CSRNet(load_weights=True).cuda() if usingCuda else CSRNet(load_weights=True).cpu()
    
    # Wajib diset model.eval() ketika testing, jika tidak maka hasil akan sangat berbeda
    model.eval()
    
    if os.path.isfile(modelPath):
       print("=> loading checkpoint '{}'".format(modelPath))
       checkpoint = torch.load(modelPath, map_location=torch.device('cpu'))
       model.load_state_dict(checkpoint['state_dict'])
       print("=> loaded checkpoint '{}' (epoch {})"
             .format(modelPath, checkpoint['epoch']))
    else:
       print("=> no checkpoint found at '{}'".format(modelPath))
    
    img = transform(Image.open(imgPath).convert('RGB')).cpu()
    output = model(img.unsqueeze(0))
    print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
    plt.imshow(temp,cmap = c.jet)
    plt.show()
    saveLargeListIntoCSV(temp, "C:\\Users\\Mujir\\Desktop\\foto\\terminal")

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
    # valSingleImage("F:\\Backup\\TA\\Model\\model_best_partA_200epoch.pth.tar", "C:\\Users\\Mujir\\Desktop\\foto\\terminal\\crop1.png")
