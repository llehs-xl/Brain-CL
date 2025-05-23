import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from models import TSEncoder, ProjectionHead
from models.losses import contrastive_loss
from models.losses import sample_contrastive_loss, observation_contrastive_loss, patient_contrastive_loss, sex_contrastive_loss
import math
import copy
import sklearn
from sklearn.utils import shuffle
from dataloading.fmri_preprocessing import PatchAndFullDataset
from tools import Gaussian_noise

class BrainCLtrain:
    
    def __init__(
        self,
        input_dims,
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda',
        lr=0.001,
        batch_size=128,
        after_iter_callback=None,
        after_epoch_callback=None,
        flag_use_multi_gpu=True
    ):
        
        super().__init__()
        self.device = device
        print(device)
        self.lr = lr
        self.batch_size = batch_size
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.flag_use_multi_gpu = flag_use_multi_gpu
        self._net = TSEncoder(input_dims=input_dims, output_dims=output_dims, hidden_dims=hidden_dims, depth=depth)
        device = torch.device(device)
        
        if device == torch.device('cuda') and self.flag_use_multi_gpu:
            self._net = nn.DataParallel(self._net)

        self._net.to(device)
        self.net = torch.optim.swa_utils.AveragedModel(self._net)
        self.net.update_parameters(self._net)

        self.after_iter_callback = after_iter_callback
        self.after_epoch_callback = after_epoch_callback
        
        self.n_epochs = 1
        self.n_iters = 1
    
    def fit(self, X, y, sex_y,masks=None, factors=None, n_epochs=None, n_iters=None, windows=50,over=0.5):

        if n_iters is None and n_epochs is None:
            n_iters = 200 if X.size <= 100000 else 600
        train_dataset = PatchAndFullDataset(X, y,sex_y,windows,over)
        train_loader = DataLoader(train_dataset, batch_size=min(self.batch_size, len(train_dataset)), shuffle=True,drop_last=False)                 
        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.lr)
        
        epoch_loss_list, epoch_f1_list = [], []
        
        while True:
            if n_epochs is not None and self.n_epochs >= n_epochs:
                break
            
            cum_loss = 0
            n_epoch_iters = 1
            
            interrupted = False
            for batch in train_loader:
                # count by iterations
                if n_iters is not None and self.n_iters >= n_iters:
                    interrupted = True
                    break
                patches,full_sequences,labels_dis,labels_id,labels_sex= batch['patch'].to(self.device),batch['full_sequence'].to(self.device),batch['label_dis'].to(self.device),batch['label_id'].to(self.device),batch['label_sex'].to(self.device)
                patches=patches.reshape(-1,patches.shape[2],patches.shape[3])
                labels_dis,labels_id=labels_dis.reshape(-1,),labels_id.reshape(-1,)
                x,_,pid=shuffle(patches,labels_dis,labels_id,random_state=42)
                x=torch.tensor(x, dtype=torch.float)
                full_sequences=torch.tensor(full_sequences, dtype=torch.float)

                optimizer.zero_grad()

                if masks is None:
                    masks = ['all_true', 'all_true', 'channel_continuous', 'frame_augmentation'] 

                if factors is None:
                    factors = [0.25, 0.25, 0.25, 0.25]
                
                "S"
                if factors[0] != 0:
                    patient_out1 = self._net(x, mask=masks[0]) 
                    patient_out2 = self._net(x, mask=masks[0]) 
                    patient_loss = contrastive_loss(
                        patient_out1,
                        patient_out2,
                        patient_contrastive_loss,
                        id=pid,
                        hierarchical=False,
                        factor=factors[0],
                    )
                else:
                    patient_loss = 0
                    
                "G"
                if factors[1] != 0:
                    trial_out1 = self._net(full_sequences, mask=masks[1]) 
                    trial_out2 = self._net(full_sequences, mask=masks[1])
                    trial_loss = contrastive_loss(
                        trial_out1,
                        trial_out2,
                        sex_contrastive_loss,
                        id=labels_sex,
                        hierarchical=False,
                        factor=factors[1],
                    )
                else:
                    trial_loss = 0
                
                "P" 
                if factors[2] != 0:
                    sample_out1 = self._net(x, mask=masks[2])
                    sample_out2 = self._net(x, mask=masks[2])
                    sample_loss = contrastive_loss(
                        sample_out1,
                        sample_out2,
                        sample_contrastive_loss,
                        hierarchical=True,
                        factor=factors[2],
                    )
                else:
                    sample_loss = 0
                    
                "F"
                if factors[3] != 0:
                    B,T,N=x.shape[0],x.shape[1],x.shape[2]
                    observation_out1 = self._net(Gaussian_noise(x), mask=masks[3])
                    observation_out2 = self._net(Gaussian_noise(x), mask=masks[3])
                    observation_out1=torch.mean(observation_out1,dim=1) 
                    observation_out1=observation_out1.reshape(B,T,320)
                    observation_out2=torch.mean(observation_out2,dim=1)
                    observation_out2=observation_out2.reshape(B,T,320)
                    observation_loss = contrastive_loss(
                        observation_out1,
                        observation_out2,
                        observation_contrastive_loss,
                        hierarchical=True,
                        factor=factors[3],
                    )
                else:
                    observation_loss = 0

                loss = trial_loss+patient_loss + sample_loss + observation_loss
                loss.backward()
                optimizer.step()
                self.net.update_parameters(self._net)

                cum_loss += loss.item()
                n_epoch_iters += 1
                self.n_iters += 1
            
            if interrupted:
                break
            
            cum_loss /= n_epoch_iters
            epoch_loss_list.append(cum_loss)
            print(f"Epoch #{self.n_epochs}: loss={cum_loss}")
            if self.after_epoch_callback is not None:
                linear_f1 = self.after_epoch_callback(self, cum_loss)
                epoch_f1_list.append(linear_f1)
        
            self.n_epochs += 1   
        return epoch_loss_list, epoch_f1_list
    
    def eval_with_pooling(self, x, mask=None):
        out = self.net(x, mask)  
        out = F.max_pool1d(
            out.transpose(1, 2),
            kernel_size=out.size(1),
        ).transpose(1, 2)
        out = out.squeeze(1)    
        return out
    
    def encode(self, X, mask=None, batch_size=None):
        """ Compute representations using the model.
        
        Args:
            X (numpy.ndarray): The input data. This should have a shape of (n_samples, timestamps, features).

        Returns:
            repr: The representations for data.
        """
        assert self.net is not None, 'please train or load a net first'
        assert X.ndim == 3
        if batch_size is None:
            batch_size = self.batch_size
        org_training = self.net.training
        self.net.eval()
        
        dataset = TensorDataset(torch.from_numpy(X).to(torch.float))
        loader = DataLoader(dataset, batch_size=batch_size)
        
        with torch.no_grad():
            output = []
            for batch in loader:
                x = batch[0].to(self.device)
                out = self.eval_with_pooling(x, mask)
                output.append(out)
                
            output = torch.cat(output, dim=0)
            
        self.net.train(org_training)
        return output.cpu().numpy()

    def save(self, fn):
        """ Save the model to a file.
        
        Args:
            fn (str): filename.
        """
        torch.save(self.net.state_dict(), fn)
    
    def load(self, fn):
        """ Load the model from a file.
        
        Args:
            fn (str): filename.
        """
        state_dict = torch.load(fn)
        self.net.load_state_dict(state_dict)
    
