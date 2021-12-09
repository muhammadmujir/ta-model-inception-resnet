import sys
import os

import warnings

from model import CSRNet
#modification
from inception_restnet_v2.inceptionresnetv2 import InceptionResNetV2
from utils import save_checkpoint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import argparse
import json
import cv2
import dataset
import time
from constant import *

parser = argparse.ArgumentParser(description='PyTorch CSRNet')

parser.add_argument('train_json', metavar='TRAIN',
                    help='path to train json')
parser.add_argument('test_json', metavar='TEST',
                    help='path to test json')

parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None,type=str,
                    help='path to the pretrained model')

parser.add_argument('gpu',metavar='GPU', type=str,
                    help='GPU id to use.')

parser.add_argument('task',metavar='TASK', type=str,
                    help='task id to use.')
# ===================================================================================

def generateTrainList():
    imagePath = "["
    for i in range(1,401):
        imagePath = imagePath + "\"" + DATASET1_TRAIN_B
        #imagePath = imagePath + "\"C:\\Users\\Admin\\Desktop\\Kuliah\\TA\\ShanghaiTech\\part_A\\train_data\\images\\"
        imagePath = imagePath + "IMG_"+str(i)+".jpg\""
        if (i < 400):
            imagePath = imagePath+", "
        if (i == 400):
            imagePath = imagePath+"]"
    return imagePath


def generateTestList():
    imagePath = "["
    for i in range(1,317):
        imagePath = imagePath + "\""+DATASET1_TEST_B
        #imagePath = imagePath + "\"C:\\Users\\Admin\\Desktop\\Kuliah\\TA\\ShanghaiTech\\part_A\\test_data\\images\\"
        imagePath = imagePath + "IMG_"+str(i)+".jpg\""
        if (i < 316):
            imagePath = imagePath+", "
        if (i == 316):
            imagePath = imagePath+"]"
    return imagePath

print(generateTrainList())
print(generateTestList())
# ===================================================================================


resultCSV = None

def main():
    
    global args,best_prec1,resultCSV
    
    best_prec1 = 1e6
    
    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 10
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
    args.seed = time.time()
    args.print_freq = 1
    
    with open(args.train_json, 'r') as outfile:        
        train_list = json.load(outfile)
    with open(args.test_json, 'r') as outfile:       
        val_list = json.load(outfile)
        
        
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    torch.cuda.manual_seed(args.seed)
    
    #model = CSRNet()
    model = InceptionResNetV2()
    if (args.gpu != "-1"):
        model = model.cuda()
    else :
        model = model.cpu()
    if (args.gpu != "-1"):
        criterion = nn.MSELoss(size_average=False).cuda()
    else:
        criterion = nn.MSELoss(size_average=False).cpu()
        
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)

    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))
            
    
    
# =====================================================================

    for epoch in range(args.start_epoch, args.epochs):
        
        adjust_learning_rate(optimizer, epoch)
        
        if (epoch == 0):
            resultCSV = open('/content/'+BASE_PATH+'result/result.csv', 'w')
        else:
            resultCSV = open('/content/'+BASE_PATH+'result/result.csv', 'a')
            
        train(train_list, model, criterion, optimizer, epoch)
        prec1 = validate(val_list, model, criterion)
        
        is_best = prec1 < best_prec1
        best_prec1 = min(prec1, best_prec1)
        
        resultCSV.write('%s;' % "BEST MAE")
        resultCSV.write('%s;' % str(best_prec1.numpy()).replace(".", ",",1))
        resultCSV.write('\n')
        
        resultCSV.close()
        
        print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer' : optimizer.state_dict(),
        }, is_best,args.task)
        
        
# =====================================================================


def train(train_list, model, criterion, optimizer, epoch):
    
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(
            train_list,
            shuffle=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]), 
            train=True, 
            seen=model.seen,
            batch_size=args.batch_size,
            num_workers=args.workers
        ),
        batch_size=args.batch_size
    )
    # train_loader = torch.utils.data.DataLoader(
    #     dataset.listDataset(
    #         train_list,
    #         shuffle=True,
    #         transform=None, 
    #         train=True, 
    #         seen=model.seen,
    #         batch_size=args.batch_size,
    #         num_workers=args.workers
    #     ),
    #     batch_size=args.batch_size
    # )
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()
    
    resultCSV.write('%s;' % "EPOCH: "+str(epoch))
    resultCSV.write('\n')
    resultCSV.write('%s;' % "IMAGE_PATH")
    resultCSV.write('%s;' % "LOSS")
    resultCSV.write('%s;' % "LOSS AVG")
    resultCSV.write('\n')
    
    for i,(img, target, img_path)in enumerate(train_loader):
        data_time.update(time.time() - end)
        if (args.gpu != "-1"):
            img = img.cuda()
        else:
            img = img.cpu()
        img = Variable(img)
        output = model(img)
        print("Image-", img_path)
        
        # my_tensor = torch.tensor([1,3,4])
        # tensor([1,3,4])
        # shape : (3,) --> 1 Dimension
        # my_tensor.unsqueeze(0)
        # tensor([[1,3,4]])
        # shape : (1,3) --> 2 Dimension
        
        # my_tensor.unsqueeze(1)
        # tensor([[1],
        #         [3],
        #         [4]])
        # shape : (3,1) --> 2 Dimension
        
        if (args.gpu != "-1"):
            target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        else:
            target = target.type(torch.FloatTensor).unsqueeze(0).cpu()
        #print("///PREV TARGET///\n",target[0][0].shape)
        #print("///TARGET SHAPE///\n",target.shape)
        target = Variable(target)
        
        #print("///OUTPUT///\n", output)
        #print("///SHAPE OUTPUT///\n", output.shape)
        #print("///SHAPE TARGET///\n", target.shape)
        # cv2.resize -> target is resized with division by 8 -> check image.py
        widthScale = target.shape[3]/output.shape[3]
        heightScale = target.shape[2]/output.shape[2]
        #print("WIDTH SCALE", widthScale)
        #print("HEIGHT SCALE", heightScale)
        #print("NUMPY TARGET ",np.float32(target[0][0]))
        target = cv2.resize(np.float32(target[0][0]),(output.shape[3],output.shape[2]),interpolation = cv2.INTER_CUBIC)*64
        if (args.gpu != "-1"):
            target = torch.FloatTensor(target).unsqueeze(0).unsqueeze(0).cuda()
        else:
            target = torch.FloatTensor(target).unsqueeze(0).unsqueeze(0).cpu()
        target = Variable(target)
        #print("TARGET SHAPE AFTER RESIZE",target.shape)
        
        loss = criterion(output, target)
        losses.update(loss.item(), img.size(0))
        #print(img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            resultCSV.write('%s;' % str(img_path))
            resultCSV.write('%s;' % str(losses.val).replace(".", ",",1))
            resultCSV.write('%s;' % str(losses.avg).replace(".", ",",1))
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
        resultCSV.write('\n')
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
        dataset.listDataset(
            val_list,
            shuffle=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
                   ]
                ),  
            train=False
        ),
        batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    
    resultCSV.write('\n')
    resultCSV.write('%s;' % "IMAGE_PATH")
    resultCSV.write('%s;' % "MAE")
    resultCSV.write('\n')
    
    for i,(img, target,img_path) in enumerate(test_loader):
        if (args.gpu != "-1"):
            img = img.cuda()
        else:
            img = img.cpu()
        img = Variable(img)
        output = model(img)
        
        resultCSV.write('%s;' % str(img_path))
        
        if (args.gpu != "-1"):
            currentMae = abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
            resultCSV.write('%s;' % str(currentMae.numpy()).replace(".", ",",1))
            mae += currentMae
        else:
            currentMae = abs(output.data.sum()-target.sum().type(torch.FloatTensor).cpu())
            resultCSV.write('%s;' % str(currentMae.numpy()).replace(".", ",",1))
            mae += currentMae
        
        resultCSV.write('\n')
        
    mae = mae/len(test_loader)    
    resultCSV.write('\n')
    resultCSV.write('%s;' % "MAE_AVG")
    resultCSV.write('%s;' % str(mae.numpy()).replace(".", ",",1))
    resultCSV.write('\n')
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae    
        
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count    
    
if __name__ == '__main__':
    main()        
    # to run in colab "!python train.py 0 0