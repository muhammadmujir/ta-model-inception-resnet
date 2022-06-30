import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
import torchvision.transforms.functional as F

class listDataset(Dataset):
    def __init__(self, root, duplicate = 1, shape=None, 
                 shuffle=True, transform=None,  train=False, 
                 seen=0, batch_size=1, num_workers=4, isLargeSize=False, 
                 isCrop=True):
        if train:
            # root = root *4
            root = root * duplicate
        random.shuffle(root)
        print("Coba")
        a = 1/0
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.isLargeSize = isLargeSize
        self.isCrop = isCrop
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error' 
        
        img_path = self.lines[index]
        
        print("get item")
        img,target = load_data(img_path,self.train) if not self.isLargeSize  else load_data_large_size(img_path,self.train, self.isCrop)
        
        #img = 255.0 * F.to_tensor(img)
        
        #img[0,:,:]=img[0,:,:]-92.8207477031
        #img[1,:,:]=img[1,:,:]-95.2757037428
        #img[2,:,:]=img[2,:,:]-104.877445883

        
        if self.transform is not None:
            img = self.transform(img)
        return img,target,img_path