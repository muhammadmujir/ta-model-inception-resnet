import h5py
import torch
import shutil
from constant import CHECKPOINT_PATH
from constant import CHECKPOINT_PATH_LOCAL

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