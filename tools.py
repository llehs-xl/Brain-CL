import scipy.io as sio
import torch
import numpy as np
from sklearn.utils import shuffle

def Gaussian_noise(x,ratio=0.8):
    B,T,N=x.shape[0],x.shape[1],x.shape[2]
    x=x.reshape(B*T,1,N)              
    x = x.repeat(1, 50, 1)            
    roi_noise_ratio=int(ratio*N) 
    roi_indices=torch.rand(B*T,50,N).argsort(dim=2)[:, :, :roi_noise_ratio] 
    noise = torch.normal(mean=0, std=0.01, size=(B*T,50,roi_noise_ratio))  
    
    roi_indices=roi_indices.to(x.device)
    noise=noise.to(x.device)
   
    x.scatter_add_(2, roi_indices, noise) 
    x_noise=x.clone()
    return x_noise

def ROI_save(ROI_label,ts,roi_index):
    ts_del_roi=[]
    for i in range (len(ts)):
        p1=ts[i].swapaxes(0,1) # 200 256
        indices_to_keep = np.where(ROI_label == roi_index)[0] 
        rows_to_zero = np.ones(p1.shape[0], dtype=bool)      
        rows_to_zero[indices_to_keep] = False                 
        p1[rows_to_zero]=0 
        ts_del_roi.append(p1.swapaxes(0,1))
    ts_del_roi=np.array(ts_del_roi)
    return ts_del_roi
        
        
def ROI_del(ROI_label,ts,roi_index):
    ts_del_roi=[]
    for i in range (len(ts)):
        p1=ts[i].swapaxes(0,1) # 200 256
        indices = np.where(ROI_label == roi_index)[0]
        p1[indices]=0
        ts_del_roi.append(p1.swapaxes(0,1))
    ts_del_roi=np.array(ts_del_roi)
    return ts_del_roi
        
def get_sen_spe(pred_all_fold,label_all_fold):

    pred_all_fold=torch.tensor(pred_all_fold)
    label_all_fold=torch.tensor(label_all_fold)
    predicted_classes = (pred_all_fold > 0.5).long()
    TP = (predicted_classes[label_all_fold == 1] == 1).sum().item()
    FN = (predicted_classes[label_all_fold == 1] == 0).sum().item()
    sensitivity= TP / (TP + FN) if (TP + FN) > 0 else 0
    TN = (predicted_classes[label_all_fold == 0] == 0).sum().item()
    FP = (predicted_classes[label_all_fold == 0] == 1).sum().item()
    specificity= TN / (TN + FP) if (TN + FP) > 0 else 0

    return round(sensitivity,4),round(specificity,4)


def get_patch_shuffle(X,Y):
    X_all_train_patch=X.reshape(-1,X.shape[2],X.shape[3])
    Y_all_train_patch=Y.reshape(-1,Y.shape[2])
    X_all_train_shuffle,Y_all_train_shuffle=shuffle(X_all_train_patch,Y_all_train_patch,random_state=42)

    return X_all_train_shuffle,Y_all_train_shuffle

def get_k_fold_data(k,i,x,y):
    assert k>1
    fold_size=x.shape[0]//k
    x_train,y_train=None,None
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        x_part,y_part=x[idx,:],y[idx,:]
        if j==i:
            x_valid,y_valid=x_part,y_part
        elif x_train is None:
            x_train,y_train=x_part,y_part
        else:
            x_train=np.concatenate((x_train,x_part),axis=0)
            y_train=np.concatenate((y_train,y_part),axis=0)

    x_t,x_v=x_train[0:int(len(x_train)*0.9),:,:],x_train[int(len(x_train)*0.9):,:,:]
    y_t,y_v=y_train[0:int(len(x_train)*0.9),:],y_train[int(len(x_train)*0.9):,:]
    return x_t,y_t,x_v,y_v,x_valid,y_valid 

def get_k_fold_data_raw(k,i,x,y):
    assert k>1
    fold_size=x.shape[0]//k
    x_train,y_train=None,None
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        x_part,y_part=x[idx,:],y[idx]
        if j==i:
            x_valid,y_valid=x_part,y_part
        elif x_train is None:
            x_train,y_train=x_part,y_part
        else:
            x_train=np.concatenate((x_train,x_part),axis=0)
            y_train=np.concatenate((y_train,y_part),axis=0)

    return x_valid,y_valid  

def get_imblanced(fea,lab,sex,seed_adhd):
    np.random.seed(seed_adhd)  
    zero_indices = np.where(lab == 0)[0]
    random_zero_indices = np.random.choice(zero_indices, 150, replace=False)
    one_indices = np.where(lab == 1)[0]
    new_indices = np.concatenate((random_zero_indices, one_indices))
    new_features = fea[new_indices]
    new_labels = lab[new_indices]
    new_sex=sex[new_indices]

    return new_features,new_labels,new_sex


def load_mat_movie(path,task):
    ts_label_raw=sio.loadmat(path)
    time_series_raw = ts_label_raw['ts'] 
    label_raw = ts_label_raw[task]
    label_sex=ts_label_raw['sex']
    fea=time_series_raw.swapaxes(1, 2) 
    lab=label_raw.squeeze()
    return fea,lab,label_sex




def load_mat(path):
    ts_label_raw=sio.loadmat(path)
    time_series_raw = ts_label_raw['ts'] 
    label_raw = ts_label_raw['dx']
    label_sex=ts_label_raw['sex']
    ts=[]
    dx=[]
    sex=[]
    for i in range(len(label_raw)):
        if label_raw[i] == 0 or label_raw[i] == 1:
            ts.append(time_series_raw[i])
            dx.append(label_raw[i])
            sex.append(label_sex[i])
    time_series_raw=np.array(ts)
    label_raw=np.array(dx)
    label_sex=np.array(sex)
    fea=time_series_raw.swapaxes(1, 2) 
    lab=label_raw.squeeze()
    return fea,lab,label_sex


