import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from .dilated_conv import DilatedConvEncoder


def generate_continuous_mask(B, T, C=None, n=5, l=0.1):
    if C:
        res = torch.full((B, T, C), True, dtype=torch.bool)
    else:
        res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            if C:
                index = np.random.choice(C, int(C/2), replace=False)
                res[i, t:t + l, index] = False
            else:
                res[i, t:t+l] = False
    return res

def frame_augmentation(B, T, C=None):
    res = torch.full((B, T, C), True, dtype=torch.bool)   
    mask_roi=int(C*0.1)
    for i in range(B):
        index = np.random.choice(C, mask_roi, replace=False)
        res[i, :, index] = False   
    return res

def generate_binomial_mask(B, T, C=None, p=0.5):
    if C:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T, C))).to(torch.bool)
    else:
        return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class ProjectionHead(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=128):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims

        # projection head for finetune
        self.proj_head = nn.Sequential(
            nn.Linear(input_dims, hidden_dims), 
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Linear(hidden_dims, output_dims)
        )
        print(f"self.proj_head={self.proj_head} ")

        self.repr_dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.repr_dropout(self.proj_head(x))
        return x


class FTClassifier(nn.Module):
    def __init__(self, input_dims, output_dims, depth, p_output_dims, hidden_dims=64, p_hidden_dims=128,
                 device='cuda', flag_use_multi_gpu=True):
        super().__init__()
        self.input_dims = input_dims 
        self.output_dims = output_dims  
        self.hidden_dims = hidden_dims  
        self.p_hidden_dims = p_hidden_dims 
        self.p_output_dims = p_output_dims  
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        # projection head for finetune
        self.proj_head = ProjectionHead(output_dims, p_output_dims, p_hidden_dims)
        device = torch.device(device)
        if device == torch.device('cuda') and flag_use_multi_gpu:
            self._net = nn.DataParallel(self._net)
            self.proj_head = nn.DataParallel(self.proj_head)
        self._net.to(device)
        self.proj_head.to(device)

        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

    def forward(self, x):
        out = self.net(x)  
        out = F.max_pool1d(
            out.transpose(1, 2),
            kernel_size=out.size(1),
        ).transpose(1, 2)  
        out = out.squeeze(1)  
        x = self.proj_head(out)
       
      
        return x


class TSEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, hidden_dims=64, depth=10, mask_mode='binomial'):
        super().__init__()
        self.input_dims = input_dims 
        self.output_dims = output_dims  
        self.hidden_dims = hidden_dims  
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims], 
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        
    def forward(self, x, mask=None): 
        if mask is None:
            mask = 'all_true'
        
        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'channel_binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)


        elif mask == 'channel_continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1), x.size(2)).to(x.device)
            x = x * mask 
        elif mask == 'frame_augmentation':
            mask = frame_augmentation(x.size(0), x.size(1), x.size(2),).to(x.device)
            x = x * mask 


        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        else:
            raise ValueError(f'\'{mask}\' is a wrong argument for mask function!')   

        x = self.input_fc(x)
        # conv encoder
        x = x.transpose(1, 2)  
        x = self.repr_dropout(self.feature_extractor(x))  
        x = x.transpose(1, 2)
        
        return x
        