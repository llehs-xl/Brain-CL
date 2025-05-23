from sklearn.utils import shuffle
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader

def get_patch(X,Y_dis,Y_id,windows,over_ratio):  #X（sample，times，roi）
    # get patch
    step=int(windows*(1-over_ratio))
    patch_num=(X.shape[1]-windows)//step +1
    patched_X=np.zeros((X.shape[0],patch_num,windows,X.shape[2]))
    patched_Y_dis=np.zeros((X.shape[0],patch_num))
    patched_Y_id=np.zeros((X.shape[0],patch_num))

    for i in range(len(X)):
        for j in range (patch_num):
            start = j * step
            end = start + windows
            patched_X[i,j, :, :] = X[i, start:end, :]
            patched_Y_dis[i, j]=Y_dis[i]
            patched_Y_id[i , j]=Y_id[i]
    return patched_X,patched_Y_dis,patched_Y_id


def load_fmri(fea, lab,windows,over_ratio):
    sub_id = np.arange(1, len(lab)+1)
    patched_X,patched_Y_dis,patched_Y_id=get_patch(fea,lab,sub_id,windows,over_ratio)
    patch_num=patched_X.shape[1]
    trial_id=np.zeros((len(lab),patch_num))
    Y_dis_id_trial = np.stack((patched_Y_dis, patched_Y_id,trial_id), axis=2) 
 
    return patched_X,Y_dis_id_trial 
    


class PatchAndFullDataset(Dataset):
    def __init__(self, data, dis_labels,sex_labels, patch_size=50, over_ratio=0.5):
        self.data = data
        self.dis_labels = dis_labels
        self.sex_labels=sex_labels
        self.patch_size = patch_size
        self.over_ratio = over_ratio
    def __len__(self):
        return len(self.data)
    
    def get_patch(self,X,windows=50,over_ratio=0.5):
        step=int(windows*(1-over_ratio))
        patch_num=(X.shape[0]-windows)//step +1 
        patched_X=np.zeros((patch_num,windows,X.shape[1]))
        for j in range (patch_num):
            start = j * step
            end = start + windows
            patched_X[j, :, :] = X[start:end, :]
        return patched_X,patch_num


    def __getitem__(self, idx):
        # get ts
        full_sequence = self.data[idx] 
        #get patch
        patch,patch_num=self.get_patch(X=full_sequence,windows=self.patch_size,over_ratio=self.over_ratio)
        #get label
        label_dis=self.dis_labels[idx].repeat(patch_num)
        label_sex=self.sex_labels[idx]
        label_id=np.full(patch_num, idx+1)

        return {'patch': patch, 'full_sequence': full_sequence, 'label_dis': label_dis,'label_id':label_id,'label_sex':label_sex}