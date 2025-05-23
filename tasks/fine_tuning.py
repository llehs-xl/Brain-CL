import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import sklearn
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import mean_absolute_error
from math import sqrt
import pandas as pd
import numpy as np
from tools import get_sen_spe

def finetune_fit(model, X_train, y_train, X_valid, y_valid,  X_test, y_test,batch_size=128, finetune_epochs=50, num_classes=2,
                 finetune_lr=0.0001, fraction=None, device='cuda'):
    if fraction:
        X_train = X_train[:int(X_train.shape[0] * fraction)]
        y_train = y_train[:int(y_train.shape[0] * fraction)]

    assert X_train.ndim == 3
    device = torch.device(device)

    model.train()

    if (num_classes==1):
        train_dataset = TensorDataset(torch.from_numpy(X_train).to(torch.float),
                                 torch.from_numpy(y_train).to(torch.float))
    else:
        train_dataset = TensorDataset(torch.from_numpy(X_train).to(torch.float),
                                  F.one_hot(torch.from_numpy(y_train).to(torch.long),
                                            num_classes=num_classes).to(torch.float))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=finetune_lr)

    if (num_classes==1):
        criterion=nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    epoch_loss_list, iter_loss_list, epoch_f1_list = [], [], []
    model.n_epochs = 1
    best_f1=0
    best_mae=100000000
    for epoch in range(finetune_epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            y_pred = model(x)
            if (num_classes==1):
                y_pred=y_pred.squeeze()
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            iter_loss_list.append(loss.item())
            grads = {}
            

        epoch_loss_list.append(sum(iter_loss_list) / len(iter_loss_list))
        model.n_epochs += 1
        #model validation
        _,_,_,_,result_val = finetune_predict(model, X_valid, y_valid, batch_size=batch_size, num_classes=num_classes, device=str(device))

        if (num_classes==2):
            print(f"Epoch number: {epoch} Finetune calssifier Loss:{epoch_loss_list[-1]} Valid_F1: {result_val['F1']} Valid_SEN: {result_val['SEN']}")
            if(result_val['F1']>best_f1 or result_val['F1']==best_f1):
                best_f1=result_val['F1']
                label,pred,label_prob,pred_prob,result_test = finetune_predict(model, X_test, y_test, batch_size=batch_size, num_classes=num_classes, device=str(device))
                print("Best Epoch={} Test_ACC={:.4f} Test_AUC={:.4f} Test_sen={:.4f}".format(epoch,result_test['Accuracy'],result_test['AUROC'],result_test['SEN']))
        
        elif(num_classes==1):
            print(f"Epoch number: {epoch} Finetune calssifier Loss:{epoch_loss_list[-1]} Valid_mae: {result_val['mae']}")
            if(result_val['mae']<best_mae or result_val['mae']==best_mae):
                best_mae=result_val['mae']
                label,pred,label_prob,pred_prob,result_test = finetune_predict(model, X_test, y_test, batch_size=batch_size, num_classes=num_classes, device=str(device))
                print("Best Epoch={} Test_r={:.4f} Test_mae={:.4f}".format(epoch,result_test['r'],result_test['mae']))

    label_fe,pred_fe,label_prob_fe,pred_prob_fe,result_test_fe=finetune_predict(model, X_test, y_test, batch_size=batch_size, num_classes=num_classes, device=str(device))
    if (num_classes==1):
        print("Final Test: Test_r={:.4f} Test_mae={:.4f}".format(result_test_fe['r'],result_test_fe['mae']))
    elif(num_classes==2):
        print("Final Test: Test_ACC={:.4f} Test_AUC={:.4f} Test_F1={:.4f}".format(result_test_fe['Accuracy'],result_test_fe['AUROC'],result_test_fe['F1']))
        
    return label,pred,label_prob,pred_prob,result_test,label_fe,pred_fe,label_prob_fe,pred_prob_fe,result_test_fe


def finetune_predict(model, X_test, y_test, batch_size=128, num_classes=2, device='cuda'):
    device = torch.device(device)
    if (num_classes==2):
        test_dataset = TensorDataset(torch.from_numpy(X_test).to(torch.float),
                                 F.one_hot(torch.from_numpy(y_test).to(torch.long), num_classes=num_classes).to(torch.float))
    elif(num_classes==1):
        test_dataset = TensorDataset(torch.from_numpy(X_test).to(torch.float),
                                 torch.from_numpy(y_test).to(torch.float))

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    org_training = model.training
    model.eval()

    sample_num = len(X_test)
    y_pred_prob_all = torch.zeros((sample_num, num_classes))
    y_target_prob_all = torch.zeros((sample_num, num_classes))
    
   
    with torch.no_grad():
        for index, (x, y) in enumerate(test_loader):
            #print("----------------index-------------",index)
            x, y = x.to(device), y.to(device)
            batch_or_number=x.shape[0]
            y_pred_prob_patch_add=torch.zeros((batch_or_number, num_classes)) 
            x=x.swapaxes(0,1)

            for i in range (x.shape[0]):
                x_patch=x[i] #(B,50,200)
                y_pred_prob_patch=model(x_patch).cpu()
                if (num_classes==2):
                    y_pred_prob_patch=F.softmax(y_pred_prob_patch,dim=1) #(B,2)
                y_pred_prob_patch_add=y_pred_prob_patch_add+y_pred_prob_patch #(B,2)

            y_pred_prob_patch_add=y_pred_prob_patch_add/(x.shape[0])  
            if (num_classes==1):
                y_target_prob=y.unsqueeze(dim=1).cpu() 
            elif (num_classes==2):
                y_target_prob= y.cpu()
            y_pred_prob_all[index*batch_size:index*batch_size+len(y)] = y_pred_prob_patch_add
            y_target_prob_all[index*batch_size:index*batch_size+len(y)] = y_target_prob
           
        if (num_classes==2):
            y_pred = y_pred_prob_all.argmax(dim=1)  # (B, )
            y_target = y_target_prob_all.argmax(dim=1)  # (b, )
            metrics_dict = {}
            metrics_dict['Accuracy'] = round(sklearn.metrics.accuracy_score(y_target, y_pred),4)
            metrics_dict['Precision'] = round(sklearn.metrics.precision_score(y_target, y_pred),4)
            metrics_dict['Recall'] = round(sklearn.metrics.recall_score(y_target, y_pred),4)
            metrics_dict['F1'] = round(sklearn.metrics.f1_score(y_target, y_pred),4)
            metrics_dict['AUROC'] = round(sklearn.metrics.roc_auc_score(y_target_prob_all, y_pred_prob_all),4)
            metrics_dict['AUPRC'] = round(sklearn.metrics.average_precision_score(y_target_prob_all, y_pred_prob_all),4)
            metrics_dict['SEN'],metrics_dict['SPE']=get_sen_spe(y_pred,y_target)
        
        elif (num_classes==1):
            y_pred = y_pred_prob_all.squeeze()
            y_target = y_target_prob_all.squeeze()
            pr2=pd.Series(y_pred,dtype=np.float64)
            re2=pd.Series(y_target,dtype=np.float64)
            metrics_dict = {}
            metrics_dict['r'] =round(pr2.corr(re2), 4)
            metrics_dict['mae'] =round(mean_absolute_error(re2,pr2),4)

    model.train(org_training)
    if (num_classes==2):
        return y_target,y_pred,y_target_prob_all,y_pred_prob_all,metrics_dict
    elif (num_classes==1):
        return y_target,y_pred,y_target_prob_all.squeeze(),y_pred_prob_all.squeeze(),metrics_dict
