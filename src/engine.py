import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import os
import numpy as np
from data import classify_anomaly_dataset, prepare_pcap_data, load_and_prepare_pcap
from model import   classify_conv_model
from utils import get_optimizer
from collections import Counter
import torch.nn.functional as F
import torch.multiprocessing as mp
import sys
import shutil
import argparse
from utils import extract_iat_features,load_data, de_parallelize_model, init_seeds, accuracy, reduce_mean, ddp_setup
import random
from torch.distributed import init_process_group,destroy_process_group
import datetime
from base import BaseTrainer
from data import classify_anomaly_dataset, prepare_pcap_data, load_and_prepare_pcap, get_ssl_pcap_dataset, get_pcap_ssl_and_val_non_ssl_dataset

class PytorchDDPTrainer(BaseTrainer):
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

        super().__init__(model,
                       train_data,
                       train_sampler,
                       val_data,
                       val_sampler,
                       optimizer,
                       gpu_id,
                       dist,
                       save_every,
                       model_dir,
                       epochs,
                       resume_enabled,
                       nprocs)

    def _run_batch(self,source, targets,val=False):
        '''
        '''
        if not val:
            output = self.model(source)
            loss = F.cross_entropy(output,targets)
            _,preds = torch.max(output,1)
            if self.dist:
                '''
                The function of barrier() is to block the process
                and ensure that each process runs all the code before this line of code before executing
                so that the average loss and average acc will not appear bec of the process execution speed
                inconsistent error
                '''
                torch.distributed.barrier()
                reduced_loss = reduce_mean(loss,self.nprocs)
            else:
                reduced_loss=loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return reduced_loss,preds
        else:
            with torch.no_grad():
                output = self.model(source)
                loss = F.cross_entropy(output,targets)
                _,preds = torch.max(output,1)
                if self.dist:
                    torch.distributed.barrier()
                    reduced_loss = reduce_mean(loss,self.nprocs)

                else:
                    reduced_loss=loss

                return reduced_loss,preds
    def _setup_train(self):
        '''
        set up training before beggining training
        helper to check if resume training
        '''
        SAVE_PATH = os.path.join(self.model_dir,'last.pt')
        #(Andrew 1.27.23): Dont know how to resume training for DDP
        if self.resume_enabled and os.path.exists(SAVE_PATH) and not self.dist:
            print("1: Single GPU resume")
            # either restart checkpoint, or not
            # resulting in overwriting
            self._resume_checkpoint()
        elif self.resume_enabled and os.path.exists(SAVE_PATH) and self.dist:
            print("2: starting DDP from ckpt")

            self._resume_ddp_checkpoint()
        elif not self.resume_enabled and self.dist:
            print("3: starting DDP from scratch: ",self.gpu_id)
            self.model.to(self.gpu_id)
            self.model = DDP(self.model,device_ids = [self.gpu_id])
            self.optimizer = get_optimizer(self.model)
        
        
    def _run_epoch(self,e):
        '''
        '''
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(e)
        if self.val_sampler is not None:
            self.val_sampler.set_epoch(e)
        b_sz = len(next(iter(self.train_data))[0])
        if self.gpu_id in {'cpu',0}:
            print("[Device {}] Epoch {} | Batchize: {} | Steps: {}".format(self.gpu_id,e,b_sz,len(self.train_data)))
        if self.dist == True:
            self.train_data.sampler.set_epoch(self.epoch)
        train_loss_total = 0
        val_loss_total = 0
        train_correct_total = 0
        val_correct_total = 0

        for ind, (source, targets) in enumerate(self.train_data):
            batch_size = targets.size(0)
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss,preds = self._run_batch(source, targets,val=False)
            if self.dist:
                '''
                The function of barrier() is to block the process
                and ensure that each process runs all the code before this line of code before executing
                so that the average loss and average acc will not appear bec of the process execution speed
                inconsistent error
                '''
                torch.distributed.barrier()
                train_loss_total+=loss

                train_acc=accuracy(preds,targets.data)[0]
                train_correct_total += reduce_mean(train_acc,self.nprocs)

            else:
                train_loss_total+=loss
                train_acc=accuracy(preds,targets.data)[0]
                train_correct_total+=train_acc

        self.model.eval()
        for ind, (source, targets) in enumerate(self.val_data):
            batch_size = targets.size(0)
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss,preds = self._run_batch(source, targets,val=True)

            if self.dist:
                '''
                The function of barrier() is to block the process
                and ensure that each process runs all the code before this line of code before executing
                so that the average loss and average acc will not appear bec of the process execution speed
                inconsistent error
                '''
                torch.distributed.barrier()
                val_loss_total+=loss
                val_acc=accuracy(preds,targets.data)[0]
                val_correct_total += reduce_mean(val_acc,self.nprocs)
            else:
                val_loss_total+=loss
                val_acc=accuracy(preds,targets.data)[0]
                val_correct_total+=val_acc
                
        if self.gpu_id in {'cpu',0}:
            avg_epoch_train_loss = train_loss_total/len(self.train_data)
            avg_train_acc = train_correct_total/len(self.train_data)
            avg_epoch_val_loss = val_loss_total/len(self.val_data)
            avg_val_acc = val_correct_total/len(self.val_data)
            self.losses.append(avg_epoch_train_loss)
            self.train_accs.append(avg_train_acc)
            self.val_losses.append(avg_epoch_val_loss)
            self.val_accs.append(avg_val_acc)
        if self.gpu_id in {'cpu',0}:
            print('epoch {} || Epoch_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(e,
        avg_epoch_train_loss,avg_epoch_val_loss))
        if self.gpu_id in {'cpu',0}:
            print(f"epoch {e} || Train Acc:", self.train_accs[-1]," || Val Acc:", self.val_accs[-1])
        
    def train(self):
        '''
        '''
        self._setup_train()
        for e in range(self.epoch,self.epochs):
            self._run_epoch(e)
            if self.gpu_id in {'cpu',0} and e%self.save_every == 0:
                print(f"Saving checkpoint at epoch {e}")
                self._save_checkpoint(e)
        
        self._save_train_results()
        if self.dist:
            destroy_process_group()

def normal_train_setting(model:torch.nn.Module,
                      batch_size:int,
                      device:object,
                      multi:bool,
                      epochs: int,
                      resume_enabled: bool,
                      nprocs: int):
    '''
    '''
    
    # print("PATH: ", os.path.dirname(__file__))
    print("Loading Data...")
    d_train, d_val = load_and_prepare_pcap()
    print("Data Loading Done!")
    # batch_size = 4
    train_dataloader = torch.utils.data.DataLoader(
        d_train, batch_size=batch_size, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(
        d_val, batch_size=batch_size, shuffle=False)
    # print(next(iter()))
    print(dict(Counter(d_train.target.tolist())))
    print(dict(Counter(d_val.target.tolist())))
    optimizer = get_optimizer(model)
    init_seeds(0)

    
    trainer = PytorchDDPTrainer(
        model=model,
        train_data=train_dataloader,
        train_sampler=None,
        val_data=val_dataloader,
        val_sampler=None,
        optimizer=optimizer,
        gpu_id=device,
        dist=multi,
        save_every=1,
        model_dir=os.path.join(os.path.dirname(__file__),'../models'),
        epochs=epochs,
        resume_enabled=resume_enabled,
        nprocs=-1
    )
    trainer.train()


if __name__ == '__main__':
    print("Hi")