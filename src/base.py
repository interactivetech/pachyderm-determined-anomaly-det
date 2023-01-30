
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from utils import extract_iat_features,load_data, de_parallelize_model, init_seeds, accuracy, reduce_mean, ddp_setup
import os
from model import classify_conv_model
from utils import get_optimizer
from torch.nn.parallel import DistributedDataParallel as DDP


class BaseTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        train_sampler,
        val_data: DataLoader,
        val_sampler,
        optimizer: torch.optim.Optimizer,
        gpu_id: int,
        dist: bool,
        save_every: int,
        model_dir: str,
        epochs: int,
        resume_enabled: bool,
        nprocs: int
    )-> None:
        self.dist = dist
        self.nprocs = nprocs
        self.resume_enabled = resume_enabled
        self.gpu_id = gpu_id
        self.train_sampler = train_sampler
        self.val_sampler = val_sampler
        # print("self.gpu_id: ",self.gpu_id )
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
        raise NotImplementedError
    def _setup_train(self):
        '''
        set up training before beggining training
        helper to check if resume training
        '''
        raise NotImplementedError
        
        
    def _run_epoch(self,e):
        '''
        '''
        raise NotImplementedError
        
        

    def _save_checkpoint(self,e):
        '''
        '''
        ckpt = {
            'epoch': e,
            'model': de_parallelize_model(self.model).state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'losses':self.losses,
            'val_losses':self.val_losses,
            'train_accs':self.train_accs,
            'val_accs':self.val_accs
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
        self.model.to(self.gpu_id)
        # get optimizer
        self.optimizer = get_optimizer(self.model)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print("Resuming Epoch {}".format(self.epoch))
    def _resume_ddp_checkpoint(self):
        '''
        '''
        ckpt = torch.load(os.path.join(self.model_dir,'last.pt'),map_location='cpu')
        self.epoch = ckpt['epoch']+1
        self.losses = ckpt['losses']
        self.val_losses = ckpt['val_losses']
        self.train_accs = ckpt['train_accs']
        self.val_accs = ckpt['val_accs']
        self.model = classify_conv_model()
        print("DDP Resume Model Training")
        self.model.load_state_dict(ckpt['model'])#ToDo load DP or DDP
        self.model.to(self.gpu_id)
        self.model = DDP(self.model,device_ids = [self.gpu_id])

        # get optimizer
        self.optimizer = get_optimizer(self.model)
        self.optimizer.load_state_dict(ckpt['optimizer'])
        print("Resuming Epoch {}".format(self.epoch))
    
    def train(self):
        '''
        '''
        raise NotImplementedError