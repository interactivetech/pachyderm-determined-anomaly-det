import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import os
import numpy as np
from src.data import classify_anomaly_dataset, prepare_pcap_data, load_and_prepare_pcap_numpy
from model import   classify_conv_model
from src.utils import get_optimizer
from collections import Counter
import torch.nn.functional as F
import torch.multiprocessing as mp
import sys
import shutil
import argparse
from src.utils import extract_iat_features,load_data, de_parallelize_model, init_seeds, accuracy, reduce_mean, ddp_setup
import random
from torch.distributed import init_process_group,destroy_process_group
import datetime
from engine import PytorchDDPTrainer
from src.ssl_utils import split_ssl_data, gen_pseudo_label, consistency_loss, ce_loss
from src.base import BaseTrainer

class CoreAPIEngine(BaseTrainer):
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
        nprocs: int,
        core_context: object,
        latest_ckpt: object, 
        trial_id: int
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
        self.core_context = core_context
        self.latest_ckpt = latest_ckpt
        self.trial_id = trial_id
    def _resume_checkpoint(self):
        '''
        
        '''
        with self.core_context.checkpoint.restore_path(self.latest_ckpt) as path:
            ckpt = torch.load(os.path.join(path,'last.pt'),map_location='cpu')
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
        with self.core_context.checkpoint.restore_path(self.latest_ckpt) as path:
            ckpt = torch.load(os.path.join(path,'last.pt'),map_location='cpu')
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
    def _save_checkpoint(self,e):
        '''
        Assumes using chief worker only
        '''
        ckpt_metadata = {"steps_completed":e+1}
        # 8. Save checkpoint
        with self.core_context.checkpoint.store_path(ckpt_metadata) as (path,uuid):
            ckpt = {
                'epoch': e,
                'model': de_parallelize_model(self.model).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'losses':self.losses,
                'val_losses':self.val_losses,
                'train_accs':self.train_accs,
                'val_accs':self.val_accs
            }
            torch.save(ckpt,os.path.join(path,'last.pt'))
        del ckpt    
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
        # SAVE_PATH = os.path.join(self.model_dir,'last.pt')
        if self.latest_ckpt is not None and not self.dist:
            print("1: Single GPU resume")
            self._resume_checkpoint()
        elif self.latest_ckpt is not None and self.dist:
            print("2: starting DDP from ckpt")
            self._resume_ddp_checkpoint()
        elif self.latest_ckpt is not None and self.dist:
            print("3: starting DDP from scratch: ",self.gpu_id)
            self.model.to(self.gpu_id)
            self.model = DDP(self.model,device_ids = [self.gpu_id])
            self.optimizer = get_optimizer(self.model)
        # final else dont need to do (No checkpoint not dist)
        # because we already initalized the model an
    
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
            # 5. report training metrics
            
            self.core_context.train.report_training_metrics(
                steps_completed=e+1,
                metrics = {
                    'avg_train_acc':avg_train_acc.item(),
                    'avg_train_loss':avg_epoch_train_loss.item(),
                }
            )
            self.core_context.train.report_validation_metrics(
                steps_completed=e+1,
                metrics = {
                    'avg_val_acc':avg_val_acc.item(),
                    'avg_val_loss':avg_epoch_val_loss.item(),
                }
            )

            self.losses.append(avg_epoch_train_loss)
            self.train_accs.append(avg_train_acc)
            self.val_losses.append(avg_epoch_val_loss)
            self.val_accs.append(avg_val_acc)
            print('epoch {} || Epoch_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(e,
        avg_epoch_train_loss,avg_epoch_val_loss))
            print(f"epoch {e} || Train Acc:", self.train_accs[-1]," || Val Acc:", self.val_accs[-1])
        
    def train(self):
        '''
        '''
        self._setup_train()
        # 7. Add searcher operations
        for op in self.core_context.searcher.operations():
            if self.gpu_id in {'cpu',0}:
                self.epochs = op.length
            print("op.length: ",op.length)
            for e in range(self.epoch,op.length):
                self._run_epoch(e)
                if self.gpu_id in {'cpu',0} and e%self.save_every == 0:
                    print(f"Saving checkpoint at epoch {e}")
                    self._save_checkpoint(e)
                # 7a. report progress of training operation
                # make sure its not last epoch
                if self.gpu_id in {'cpu',0} and e <op.length-1: 
                    op.report_progress(e+1)
                # 7b.check for pre-emption signal
                if self.core_context.preempt.should_preempt():
                    print("Preemption Signal Detected, stopping training...")
                    return
            # 7c. report completion of training operation
            if self.gpu_id in {'cpu',0}:
                final_acc = self.val_accs[-1]
                op.report_completed(final_acc.item())
        
        self._save_train_results()
        if self.dist:
            destroy_process_group()

def run_train(model:torch.nn.Module,
                      batch_size:int,
                      device:object,
                      multi:bool,
                      epochs: int,
                      resume_enabled: bool,
                      nprocs: int,
                      dist: bool,
                      world_size: int,
                      core_context, 
                      latest_ckpt, 
                      trial_id):
    '''
    '''
    # print("PATH: ", os.path.dirname(__file__))
    print("Loading Data...")
    x_train, x_val,y_train, y_val = load_and_prepare_pcap_numpy()
    d_train = classify_anomaly_dataset(x_train,y_train)
    d_val = classify_anomaly_dataset(x_val,y_val)
    print("Data Loading Done!")
    # batch_size = 4
    if dist:
            # batch_size = 4
        train_sampler = torch.utils.data.distributed.DistributedSampler(d_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(d_val,shuffle=False)
        train_dataloader = torch.utils.data.DataLoader(
            d_train, batch_size=batch_size, pin_memory= True, sampler=train_sampler)

        val_dataloader = torch.utils.data.DataLoader(
            d_val, batch_size=batch_size, pin_memory=True,sampler=val_sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            d_train, batch_size=batch_size, shuffle=True)

        val_dataloader = torch.utils.data.DataLoader(
            d_val, batch_size=batch_size, shuffle=False)
    # print(next(iter()))
    print(dict(Counter(d_train.target.tolist())))
    print(dict(Counter(d_val.target.tolist())))
    optimizer = get_optimizer(model)
    if not dist:
        init_seeds(0)
    if dist:
        trainer = CoreAPIEngine(
        model=model,
        train_data=train_dataloader,
        train_sampler=train_sampler,
        val_data=val_dataloader,
        val_sampler=val_sampler,
        optimizer=optimizer,
        gpu_id=device,
        dist=multi,
        save_every=1,
        model_dir='models/',
        epochs=-1,
        resume_enabled=resume_enabled,
        nprocs = world_size,
        core_context=core_context, 
        latest_ckpt=latest_ckpt, 
        trial_id=trial_id
        )
        trainer.train()
    else:
        trainer = CoreAPIEngine(
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
            epochs=-1,
            resume_enabled=resume_enabled,
            nprocs=-1,
            core_context=core_context, 
            latest_ckpt=latest_ckpt, 
            trial_id=trial_id
        )
        trainer.train()
