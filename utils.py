import h5py
import torch
import shutil
from constant import CHECKPOINT_PATH
from constant import CHECKPOINT_PATH_LOCAL
import numpy as np
import os
import re 
import math
from pathlib import Path 

def get_order(file):
    file_pattern = re.compile(r'.*?(\d+).*?')
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, path, filename='checkpoint.pth.tar'):
    torch.save(state, path+filename)
    if is_best:
        shutil.copyfile(path+filename, path+'model_best.pth.tar')            

def saveLargeListIntoCSV(arr, path):
    arr = np.array(arr)
    resultCSV = open(os.path.join(path, 'result.csv'), 'w')
    for row in arr:
        for col in row:
            resultCSV.write('%s;' % str(col))
        resultCSV.write('\n')