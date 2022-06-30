import random
import os
from PIL import Image,ImageFilter,ImageDraw
import numpy as np
import h5py
from PIL import ImageStat
import cv2

def load_data(img_path,train = True):
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if False:
        crop_size = (img.size[0]/2,img.size[1]/2)
        if random.randint(0,9)<= -1:
            
            
            dx = int(random.randint(0,1)*img.size[0]*1./2)
            dy = int(random.randint(0,1)*img.size[1]*1./2)
        else:
            dx = int(random.random()*img.size[0]*1./2)
            dy = int(random.random()*img.size[1]*1./2)
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
    
        if random.random()>0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    target = cv2.resize(target,(int(target.shape[1]//8),int(target.shape[0]//8)),interpolation = cv2.INTER_CUBIC)*64
    
    return img,target

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


def load_data_large_size(img_path, train = True, crop=True):
    min_size = 512
    max_size = 2048
    gt_path = img_path.replace('.jpg','.h5').replace('images','ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    
    img_h, img_w, ratio = cal_new_size(target.shape[0], target.shape[1], min_size, max_size)
    target = cv2.resize(target,(img_w,img_h),interpolation = cv2.INTER_CUBIC)/(ratio*ratio)
    img = img.resize((img_w,img_h))
    
    print("CROP: ", crop)
    if crop:
        print("CROP")
        crop_size = (512,512)
        if random.randint(0,9) <= -1:
            dx = int(random.randint(0,1)*(img.size[0] - crop_size[0]))
            dy = int(random.randint(0,1)*(img.size[1] - crop_size[1]))
        else:
            dx = int(random.random()*(img.size[0] - crop_size[0]))
            dy = int(random.random()*(img.size[1] - crop_size[1]))
        
        img = img.crop((dx,dy,crop_size[0]+dx,crop_size[1]+dy))
        target = target[dy:crop_size[1]+dy,dx:crop_size[0]+dx]
    
        if random.random()>0.5:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
    
    target = cv2.resize(target,(int(target.shape[1]//8),int(target.shape[0]//8)),interpolation = cv2.INTER_CUBIC)*64
    
    return img,target