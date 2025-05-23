from braincltrain import BrainCLtrain
from models.encoder import FTClassifier
from tasks.fine_tuning import finetune_fit
from dataloading.fmri_preprocessing import load_fmri

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
import random
import copy
import sklearn

from utils import process_batch_ts
from utils import seed_everything
from datetime import datetime
from math import sqrt
import argparse
import scipy.io as sio
from tools import *
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error 

start_time = datetime.now()
parser = argparse.ArgumentParser()

parser.add_argument("--fmri", type=str, default="disorder", help="general or disorde")
parser.add_argument("--dataset", type=str, default="cobre", help="camcan-age,camcan-iq,nki-age")
parser.add_argument("--task", type=str, default="classification", help="regression or classification") #3
parser.add_argument('--device', default='cuda:1', type=str,help='cpu or cuda')
parser.add_argument("--fold", type=int, default=5, help="fold cross-validartion")
parser.add_argument("--RANDOM_SEED", type=int, default=42, help="random seed")
parser.add_argument('--S_OVERLAPPING', type=float, default=0.5)
parser.add_argument('--S_TIMESTAMPS', type=int, default=50) 
parser.add_argument('--input_dims', type=int, default=200)
parser.add_argument('--output_dims', type=int, default=320)
parser.add_argument('--depth', type=int, default=4)

parser.add_argument('--pretrain_batch_size', type=int, default=16)
parser.add_argument('--shuffle_function', type=str, default="batch")
parser.add_argument('--verbose', type=bool, default=True)
parser.add_argument('--flag_use_multi_gpu', action='store_true', default=False)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--masks', type=str, nargs='+', default=['all_true', 'all_true', 'channel_continuous', 'frame_augmentation'])
parser.add_argument('--factors', type=float, nargs='+', default=[0.25, 0.25, 0.25, 0.25]) 
parser.add_argument('--num_classes', type=int, default=2)
parser.add_argument('--pretrain_lr', type=float, default=1e-4)

parser.add_argument('--fraction_100', type=float, default=1.0)
parser.add_argument('--finetune_batch_size_100', type=int, default=16)
parser.add_argument('--finetune_epochs_100', type=int, default=100)
parser.add_argument('--finetune_lr_100', type=float, default=1e-4)

parser.add_argument("--leixing", type=str, default="zhu", help="abla or para")
parser.add_argument("--ROI_del_index", type=int, default=None, help="del ROI")
parser.add_argument("--ROI_save_index", type=int, default=None, help="savr ROI")


import warnings
warnings.filterwarnings("ignore")

configs = parser.parse_args()
RANDOM_SEED = configs.RANDOM_SEED

print(configs)
if (configs.fmri=="general"):
    if configs.dataset=="camcan-age":
        f_path="your_path"
        l_path="your_path"
        sex_lab=np.load("your_path")
        
    fea=np.load(f_path)
    lab=np.load(l_path)
    
    print("fea={} lab={} sex_lab={}".format(fea.shape,lab.shape,sex_lab))

fea = process_batch_ts(fea, fs=configs.S_TIMESTAMPS,normalized=True, bandpass_filter=False)


random_int= random.randint(1, 50000)
random.seed(random_int)
sample_num=len(lab)
sample_list = [i for i in range(len(lab))] 
sample_list = random.sample(sample_list, sample_num)
fea,lab,sex_lab= fea[sample_list,:],lab[sample_list],sex_lab[sample_list]

X_all_train, Y_all_train=load_fmri(fea, lab,windows=configs.S_TIMESTAMPS,over_ratio=configs.S_OVERLAPPING)

working_directory = 'your_path/{}/'.format(configs.dataset) 
logging_directory = 'your_path/{}/'.format(configs.dataset)
dataset_save_path = working_directory

if (configs.leixing =="zhu"):
    pretrain_model=f"{working_directory}Norm_seed_{RANDOM_SEED}_pre{configs.n_epochs}_lr{configs.pretrain_lr}_tune{configs.finetune_epochs_100}_{configs.finetune_lr_100}.pt"
elif (configs.leixing=="para"):
    working_directory='your_path/{}/'.format(configs.dataset) 
    pretrain_model=f"{working_directory}Norm_W{configs.S_TIMESTAMPS}_O{configs.S_OVERLAPPING}_{configs.factors}.pt"
elif (configs.leixing=="abla"):
    working_directory='your_path/{}/'.format(configs.dataset) 
    pretrain_model=f"{working_directory}Norm_{configs.factors}.pt"
elif (configs.leixing=="ROI_analysis"):
    working_directory='your_path/{}/'.format(configs.dataset) 
    pretrain_model=f"{working_directory}_ROIdel{configs.ROI_del_index}_ROIsave{configs.ROI_save_index}.pt"

if not os.path.exists(working_directory):
    os.makedirs(working_directory)


def pretrain_callback(model, loss):
    n = model.n_epochs
    if n % 49 == 0: 
        print("model.n_epochs={} save model".format(n))
        model.save(pretrain_model)

label_all_fold, pred_all_fold, label_prob_all_fold,pred_prob_all_fold, result_all_fold =[],[],[],[],[]
label_all_fold_fe, pred_all_fold_fe,label_prob_all_fold_fe,pred_prob_all_fold_fe,result_all_fold_fe=[],[],[],[],[]

for fold in range(configs.fold):
    seed_everything(RANDOM_SEED)
    X_train,y_train,X_val,y_val,X_test,y_test =get_k_fold_data(configs.fold,fold,X_all_train,Y_all_train) 
   
    # num_0,num_1=np.count_nonzero(y_train[:, 0,0] == 0),np.count_nonzero(y_train[:, 0,0] == 1)
    # num_0_v,num_1_v=np.count_nonzero(y_val[:, 0,0] == 0),np.count_nonzero(y_val[:, 0,0] == 1)
    # num_0_t,num_1_t=np.count_nonzero(y_test[:, 0,0] == 0),np.count_nonzero(y_test[:, 0,0] == 1)
 

    X_train_shuffle,y_train_shuffle=get_patch_shuffle(X_train,y_train)
    X_val_shuffle,y_val_shuffle=X_val,y_val
 
    model = BrainCLtrain( input_dims=configs.input_dims,device=configs.device,lr=configs.pretrain_lr,depth=configs.depth,batch_size=configs.pretrain_batch_size,output_dims=configs.output_dims,flag_use_multi_gpu=configs.flag_use_multi_gpu,after_epoch_callback=pretrain_callback)
    
    '''(pre-train)'''
    if os.path.isfile(pretrain_model):
        state_dict = torch.load(pretrain_model)
        model.net.load_state_dict(state_dict)
    else:
        epoch_loss_list, epoch_f1_list = model.fit(fea,lab,sex_lab,windows=configs.S_TIMESTAMPS,over=configs.S_OVERLAPPING, n_epochs=configs.n_epochs,masks = configs.masks,factors = configs.factors)
        
    
    '''fine-tune'''
    seed_everything(RANDOM_SEED)
    finetune_model = FTClassifier(input_dims=configs.input_dims, output_dims=configs.output_dims, depth=configs.depth, p_output_dims=configs.num_classes, device=configs.device, flag_use_multi_gpu=configs.flag_use_multi_gpu) 
    finetune_model.net.load_state_dict(torch.load(pretrain_model)) 
    
    label,pred,label_prob,pred_prob,result_test,label_fe,pred_fe,label_prob_fe,pred_prob_fe,result_test_fe = finetune_fit(finetune_model, X_train_shuffle, y_train_shuffle[:, 0], X_val_shuffle, y_val_shuffle[:, 0,0],X_test,y_test[:,0,0],batch_size=configs.finetune_batch_size_100, finetune_epochs=configs.finetune_epochs_100, num_classes=configs.num_classes, finetune_lr=configs.finetune_lr_100, fraction=configs.fraction_100, device=configs.device)

    label_all_fold.extend(label)
    pred_all_fold.extend(pred)
    label_prob_all_fold.append(label_prob)
    pred_prob_all_fold.append(pred_prob)
    result_all_fold.append(result_test)

    label_all_fold_fe.extend(label_fe)
    pred_all_fold_fe.extend(pred_fe)
    label_prob_all_fold_fe.append(label_prob_fe)
    pred_prob_all_fold_fe.append(pred_prob_fe)
    result_all_fold_fe.append(result_test_fe)

label_prob_all_fold,pred_prob_all_fold=np.concatenate(label_prob_all_fold, axis=0),np.concatenate(pred_prob_all_fold, axis=0)
label_prob_all_fold_fe,pred_prob_all_fold_fe=np.concatenate(label_prob_all_fold_fe, axis=0),np.concatenate(pred_prob_all_fold_fe, axis=0)


label_all_fold=np.array(label_all_fold)
pred_all_fold=np.array(pred_all_fold)
print('label_all_fold',label_all_fold)
print("pred_all_fold",pred_all_fold)
print('label_prob_all_fold',label_prob_all_fold)
print("pred_prob_all_fold",pred_prob_all_fold)

print("****************************result of each fold****************************")
print("result_all_fold=",result_all_fold)

if  (configs.num_classes==1):
    pr2=pd.Series(pred_all_fold,dtype=np.float64)
    re2=pd.Series(label_all_fold,dtype=np.float64)
    metrics_dict_final= {}
    metrics_dict_final['r'] =round(pr2.corr(re2), 4)
    metrics_dict_final['mae'] = round(mean_absolute_error(re2,pr2),4)
    metrics_dict_final['r2'] = round(r2_score(re2, pr2),4)
    metrics_dict_final['mse']=round(mean_squared_error(re2,pr2),4)
    metrics_dict_final['rmse']=round(sqrt(metrics_dict_final['mse']),4)
    print("Total Time={}".format(datetime.now()-start_time))
    print("mean result: metrics_dict_final={}".format(metrics_dict_final))

if (configs.num_classes==2):
    metrics_dict_final= {}
    metrics_dict_final['Accuracy'] = round(sklearn.metrics.accuracy_score(label_all_fold, pred_all_fold),4)
    metrics_dict_final['Precision'] = round(sklearn.metrics.precision_score(label_all_fold, pred_all_fold),4)
    metrics_dict_final['Recall'] = round(sklearn.metrics.recall_score(label_all_fold, pred_all_fold),4)
    metrics_dict_final['F1'] = round(sklearn.metrics.f1_score(label_all_fold, pred_all_fold),4)
    metrics_dict_final['AUROC'] = round(sklearn.metrics.roc_auc_score(label_prob_all_fold, pred_prob_all_fold),4)
    metrics_dict_final['AUPRC'] = round(sklearn.metrics.average_precision_score(label_prob_all_fold, pred_prob_all_fold),4)
    metrics_dict_final['SEN'],metrics_dict_final['SPE']=get_sen_spe(pred_all_fold,label_all_fold)
    print("Total Time={}".format(datetime.now()-start_time))
    print("mean result: metrics_dict_final={}".format(metrics_dict_final))
   