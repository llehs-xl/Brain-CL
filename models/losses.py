import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def contrastive_loss(z1, z2, loss_func, id=None, hierarchical=False, factor=1.0):
    if factor == 0:
        return 0

    if not hierarchical:
        if id is not None:
            return loss_func(z1, z2, id)
        else:
            return loss_func(z1, z2)
    # enable hierarchical loss
    else:
        loss = torch.tensor(0., device=z1.device)
        d = 0
        while z1.size(1) > 1:
            if id is not None:
                # pass patient and trial loss function
                loss += loss_func(z1, z2, id)
            else:
                # pass sample and observation loss function
                loss += loss_func(z1, z2)
            d += 1
            z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
            z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)
        return loss * factor / d


def sample_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=0) 
    z = z.transpose(0, 1) 
    sim = torch.matmul(z, z.transpose(1, 2)) 
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]   
    logits += torch.triu(sim, diagonal=1)[:, :, 1:] 
    logits = -F.log_softmax(logits, dim=-1) 
    
    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def observation_contrastive_loss(z1, z2):
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.)
    z = torch.cat([z1, z2], dim=1)  # B x 2T x C
    sim = torch.matmul(z, z.transpose(1, 2))  # B x 2T x 2T
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]    # B x 2T x (2T-1)
    logits += torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    
    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss

#S
def patient_contrastive_loss(z1, z2, pid):
    return id_contrastive_loss(z1, z2, pid)
#G
def sex_contrastive_loss(z1, z2, sex):
    return id_contrastive_loss(z1, z2, sex)


def id_contrastive_loss(z1, z2, id):
    id = id.cpu().detach().numpy()
    str_pid = [str(i) for i in id]
    str_pid = np.array(str_pid, dtype=object)
    pid1, pid2 = np.meshgrid(str_pid, str_pid)
    pid_matrix = pid1 + '-' + pid2
    pids_of_interest = np.unique(str_pid + '-' + str_pid) 
    bool_matrix_of_interest = np.zeros((len(str_pid), len(str_pid)))
    for pid in pids_of_interest:
        bool_matrix_of_interest += pid_matrix == pid

    rows1, cols1 = np.where(np.triu(bool_matrix_of_interest, 1))  # upper triangle same patient combs
    rows2, cols2 = np.where(np.tril(bool_matrix_of_interest, -1))  # down triangle same patient combs
    
    B, T = z1.size(0), z1.size(1)
    loss = 0
    z1 = torch.nn.functional.normalize(z1, dim=1) 
    z2 = torch.nn.functional.normalize(z2, dim=1)
 
    view1_array = z1.permute(0, 2, 1).reshape((B, -1))
    view2_array = z2.permute(0, 2, 1).reshape((B, -1))
    norm1_vector = view1_array.norm(dim=1).unsqueeze(0) 
    norm2_vector = view2_array.norm(dim=1).unsqueeze(0)
    
    sim_matrix = torch.mm(view1_array, view2_array.transpose(0, 1))
    
    norm_matrix = torch.mm(norm1_vector.transpose(0, 1), norm2_vector)
    
    temperature = 0.1
    argument = sim_matrix/(norm_matrix*temperature)
   
    sim_matrix_exp = torch.exp(argument)
   
    mask=torch.ones_like(sim_matrix_exp, dtype=torch.bool)
    mask.fill_diagonal_(False)
    if (torch.all(mask == False)):
        return 0
    else:
        triu_sum = torch.sum(sim_matrix_exp * mask, dim=1)
        tril_sum = torch.sum(sim_matrix_exp * mask, dim=0)

        # triu_sum = torch.sum(sim_matrix_exp*(1-torch.eye(sim_matrix_exp.size(0), device=sim_matrix_exp.device)), 1)  # add column 
        # tril_sum = torch.sum(sim_matrix_exp*(1 - torch.eye(sim_matrix_exp.size(0), device=sim_matrix_exp.device)), 0)  # add row

        loss_terms = 0

        # upper triangle same patient combs exist
        if len(rows1) > 0:
            triu_elements = sim_matrix_exp[rows1, cols1]  
            loss_triu = -torch.mean(torch.log(triu_elements / triu_sum[rows1]))
            loss += loss_triu  
            loss_terms += 1

        # down triangle same patient combs exist
        if len(rows2) > 0:
        
            tril_elements = sim_matrix_exp[rows2, cols2]
        
            loss_tril = -torch.mean(torch.log(tril_elements / tril_sum[rows2]))
        
            loss += loss_tril  
            loss_terms += 1

        if loss_terms == 0:
            return 0
        else:
            loss = loss/loss_terms
            return loss
