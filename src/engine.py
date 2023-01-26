import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import os
import numpy as np
from data import classify_anomaly_dataset
from model import   classify_conv_model
from utils import get_optimizer
from collections import Counter
import torch.nn.functional as F

def is_parallel(model):
    '''
    return if model is DataParallel (DP) or DistributedDataParallel (DDP)
    '''
    return type(model) in (nn.DataParallel, nn.parallel.DistributedDataParallel)

def de_parallelize_model(model):
    '''
    De-parallelize model: return a single GPU model if model is of type DP or DDP
    '''
    return model.module if is_parallel(model) else model

class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        val_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        dist: bool,
        save_every: int,
        model_dir: str,
        epochs: int
    )-> None:
        self.dist = dist
        self.gpu_id = gpu_id
        assert gpu_id in [0,1,2,3,'cpu']

        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.model_dir = model_dir
        self.epoch = 0
        self.epochs = epochs

        self.losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def _run_batch(self,source, targets,val=False):
        '''
        '''
        if not val:
            self.optimizer.zero_grad()
            output = self.model(source)
            loss = F.cross_entropy(output,targets)
            _,preds = torch.max(output,1)
            loss.backward()
            self.optimizer.step()
            # if ind%10==0:
            #     print("Loss: ", loss.item())
            return loss.item(),preds
        else:
            with torch.no_grad():
                output = self.model(source)
                loss = F.cross_entropy(output,targets)
                _,preds = torch.max(output,1)
                # if ind%10==0:
                #     print("Loss: ", loss.item())
                return loss.item(),preds
    def _setup_train(self):
        '''
        set up training before beggining training
        helper to check if resume training
        '''
        SAVE_PATH = os.path.join(self.model_dir,'last.pt')
        if os.path.exists(SAVE_PATH):
            self._resume_checkpoint()
        
        
    def _run_epoch(self,e):
        '''
        '''
        self.model.train()
        b_sz = len(next(iter(self.train_data))[0])
        print("[Device {}] Epoch {} | Batchize: {} | Steps: {}".format(self.gpu_id,e,b_sz,len(self.train_data)))
        if self.dist == True:
            self.train_data.sampler.set_epoch(self.epoch)
        train_loss_total = 0
        val_loss_total = 0
        train_correct_total = 0
        val_correct_total = 0

        for ind, (source, targets) in enumerate(self.train_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss,preds = self._run_batch(source, targets,val=False)
            train_loss_total+=loss
            train_correct_total+=torch.sum(preds==targets.data).item()
        
        self.model.eval()
        for ind, (source, targets) in enumerate(self.val_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss,preds = self._run_batch(source, targets,val=True)
            val_loss_total+=loss
            val_correct_total+=torch.sum(preds==targets.data).item()
        # print("val_correct_total: ",train_correct_total)
        # print("len(self.train_data): ",len(self.train_data.dataset))
        avg_epoch_train_loss = train_loss_total/len(self.train_data)
        avg_train_acc = train_correct_total/len(self.train_data.dataset)
        avg_epoch_val_loss = val_loss_total/len(self.val_data)
        avg_val_acc = val_correct_total/len(self.val_data.dataset)
        self.losses.append(avg_epoch_train_loss)
        self.train_accs.append(avg_train_acc)
        self.val_losses.append(avg_epoch_val_loss)
        self.val_accs.append(avg_val_acc)
        print('epoch {} || Epoch_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(e,
        avg_epoch_train_loss,avg_epoch_val_loss))
        print(f"epoch {e} || Train Acc:", self.train_accs[-1]," || Val Acc:", self.val_accs[-1])


    def _save_checkpoint(self,e):
        '''
        '''
        ckpt = {
            'epoch': e,
            'model': de_parallelize_model(self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses':self.losses,
            'val_losses':self.val_losses,
            'train_acc':self.train_accs,
            'val_acc':self.val_accs
        }
        torch.save(ckpt,os.path.join(self.model_dir,'last.pt'))
        del ckpt
    def _resume_checkpoint(self):
        '''
        '''
        ckpt = torch.load(os.path.join(self.model_dir,'last.pt'),map_location='cpu')
        self.epoch = ckpt['epoch']+1
        self.losses = ckpt['losses']
        self.val_losses = ckpt['val_losses']
        self.train_accs = ckpt['train_accs']
        self.val_accs = ckpt['val_accs']
        self.model = classify_conv_model()
        self.model.load_state_dict(ckpt['model'])#ToDo load DP or DDP
        # get optimizer
        self.optimizer = get_optimizer(self.model)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print("Resuming Epoch {}".format(self.epoch))
    
    def train(self):
        '''
        '''
        self._setup_train()
        for e in range(self.epoch,self.epochs):
            self._run_epoch(e)
            if self.gpu_id in {'cpu',0} and e%self.save_every == 0:
                self._save_checkpoint(e)

if __name__ == '__main__':
    model = classify_conv_model()
    x_train = np.random.rand(100,12)
    print("x_train: ", x_train.shape)
    y_train = np.zeros((100))
    x_val = np.random.randn(20,12)
    y_val = np.ones((20))

    x_train = torch.from_numpy(x_train).type(torch.float32).unsqueeze(-1)
    x_val = torch.from_numpy(x_val).type(torch.float32).unsqueeze(-1)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    
    d_train = classify_anomaly_dataset(x_train,y_train)
    d_val = classify_anomaly_dataset(x_val,y_val)

    batch_size = 4
    train_dataloader = torch.utils.data.DataLoader(
        d_train, batch_size=batch_size, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(
        d_val, batch_size=batch_size, shuffle=False)
    # print(next(iter()))
    print(dict(Counter(d_train.target.tolist())))
    print(dict(Counter(d_val.target.tolist())))
    optimizer = get_optimizer(model)
    trainer = BaseTrainer(
        model=model,
        train_data=train_dataloader,
        val_data=val_dataloader,
        optimizer=optimizer,
        gpu_id='cpu',
        dist=False,
        save_every=1,
        model_dir='/Users/mendeza/Documents/2023_projects/pachyderm-determined-anomaly-det/models',
        epochs=10
    )
    trainer.train()