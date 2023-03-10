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
from src.coreapi_engine import CoreAPIEngine
from src.ssl_utils import split_ssl_data, gen_pseudo_label, consistency_loss, ce_loss
from src.data import classify_anomaly_dataset, prepare_pcap_data, load_and_prepare_pcap, get_ssl_pcap_dataset, get_pcap_ssl_and_val_non_ssl_dataset, load_and_prepare_pcap_numpy

class SSLCoreAPITrainer(CoreAPIEngine):
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        unlabeled_train_data: DataLoader,
        train_sampler: DataLoader,
        unlabeled_train_sampler: DataLoader,
        val_data: DataLoader,
        unlabeled_val_data: DataLoader,
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
            nprocs,
            core_context,
            latest_ckpt, 
            trial_id)
        self.unlabeled_train_data = unlabeled_train_data
        self.unlabeled_val_data = unlabeled_val_data
        self.unlabeled_train_sampler = unlabeled_train_sampler
        assert self.unlabeled_train_data is not None
    
    def _run_batch_ssl(self,
                       lb_source,
                       lb_targets,
                       ulb_source,
                       ulb_targets,
                       val=False):
        '''
        '''
        if not val:
            num_lb = lb_targets.shape[0]
            noise = torch.randn(ulb_source.shape)*1e-7
            noise = noise.to(self.gpu_id)
            percent_noise = torch.FloatTensor([0.1])
            percent_noise=percent_noise.to(self.gpu_id)
            constant = torch.FloatTensor([1]) 
            constant = constant.to(self.gpu_id)
            ulb_source_noise = noise * percent_noise + ulb_source * (constant- percent_noise)
            ulb_source_noise.to(self.gpu_id)

            # ulb_source+=torch.randn((1))
            # print(f"ulb std:{ulb_source.std()}, mean: {ulb_source.mean()}")
            inputs = torch.cat((lb_source,ulb_source,ulb_source_noise))
            output = self.model(inputs)
            logits_x_lb = output[:num_lb]
            logits_x_ulb, logits_x_ulb_s = output[num_lb:].chunk(2)
            sup_loss = F.cross_entropy(logits_x_lb,lb_targets)
            _,preds = torch.max(logits_x_lb,1)
            pseudo_label = gen_pseudo_label(logits_x_ulb,use_hard_label=True,label_smoothing=0.7)
            # print("pseudo_label: ",pseudo_label)
            consistency_l = consistency_loss(logits_x_ulb_s,pseudo_label,name='mse')
            total_loss = sup_loss + 0.05*consistency_l
            if self.dist:
                '''
                The function of barrier() is to block the process
                and ensure that each process runs all the code before this line of code before executing
                so that the average loss and average acc will not appear bec of the process execution speed
                inconsistent error
                '''
                torch.distributed.barrier()
                reduced_loss = reduce_mean(total_loss,self.nprocs)
                reduced_sup_loss = reduce_mean(sup_loss,self.nprocs)
                reduced_consistency_loss = reduce_mean(consistency_l,self.nprocs)
            else:
                reduced_loss=total_loss
                reduced_sup_loss=sup_loss
                reduced_consistency_loss=consistency_l
            self.optimizer.zero_grad()
            reduced_loss.backward()
            self.optimizer.step()
            reduced_loss_dict = {
                'total_loss': reduced_loss,
                'supervised_loss': reduced_sup_loss,
                'consistency_loss':reduced_consistency_loss
            }
            return reduced_loss_dict,preds
        else:
            with torch.no_grad():
                num_lb = lb_targets.shape[0]
                inputs = torch.cat((lb_source,ulb_source))
                output = self.model(inputs)
                logits_x_lb = output[:num_lb]
                logits_u_lb = output[num_lb:]
                sup_loss = F.cross_entropy(logits_x_lb,lb_targets)
                _,preds = torch.max(output,1)
                pseudo_label = gen_pseudo_label(logits_u_lb)
                consistency_l = consistency_loss(logits_u_lb,pseudo_label)
                total_loss = sup_loss + 0.05*consistency_l
                if self.dist:
                    torch.distributed.barrier()
                    reduced_loss = reduce_mean(total_loss,self.nprocs)
                    reduced_sup_loss = reduce_mean(sup_loss,self.nprocs)
                    reduced_consistency_loss = reduce_mean(consistency_l,self.nprocs)
                else:
                    reduced_loss=total_loss
                    reduced_sup_loss=sup_loss
                    reduced_consistency_loss=consistency_l
                reduced_loss_dict = {
                'total_loss': reduced_loss,
                'supervised_loss': sup_loss,
                'consistency_loss':consistency_l
                }
                return reduced_loss_dict,preds 
    def _run_epoch(self,e):
        '''
        '''
        self.model.train()
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(e)
        if self.val_sampler is not None:
            self.val_sampler.set_epoch(e)
        if self.unlabeled_train_sampler is not None:
            self.unlabeled_train_sampler.set_epoch(e)
        b_sz = len(next(iter(self.train_data))[0])
        if self.gpu_id in {'cpu',0}:
            print("[Device {}] Epoch {} | Batchize: {} | Steps: {}".format(self.gpu_id,e,b_sz,len(self.train_data)))
        if self.dist == True:
            self.train_data.sampler.set_epoch(self.epoch)
        train_loss_total = 0
        train_supervised_loss_total = 0
        train_unsupervised_loss_total = 0
        val_loss_total = 0
        train_correct_total = 0
        val_correct_total = 0

        for ind, ((lb_source, lb_targets),(ulb_source, ulb_targets)) in enumerate(zip(self.train_data,self.unlabeled_train_data)):
            batch_size = lb_targets.size(0)
            lb_source = lb_source.to(self.gpu_id)
            lb_targets = lb_targets.to(self.gpu_id)
            ulb_source = ulb_source.to(self.gpu_id)
            ulb_targets = ulb_targets.to(self.gpu_id)
            reduced_loss_dict,preds = self._run_batch_ssl(lb_source, 
                                         lb_targets,
                                         ulb_source,
                                         ulb_targets,
                                         val=False)
            if self.dist:
                '''
                The function of barrier() is to block the process
                and ensure that each process runs all the code before this line of code before executing
                so that the average loss and average acc will not appear bec of the process execution speed
                inconsistent error
                '''
                torch.distributed.barrier()
                train_loss_total+=reduced_loss_dict['total_loss']
                train_supervised_loss_total+=reduced_loss_dict['supervised_loss']
                train_unsupervised_loss_total+=reduced_loss_dict['consistency_loss']
                train_acc=accuracy(preds,lb_targets.data)[0]
                train_correct_total += reduce_mean(train_acc,self.nprocs)

            else:
                train_loss_total+=reduced_loss_dict['total_loss']
                train_supervised_loss_total+=reduced_loss_dict['supervised_loss']
                train_unsupervised_loss_total+=reduced_loss_dict['consistency_loss']
                train_acc=accuracy(preds,lb_targets.data)[0]
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
            avg_epoch_train_supervised_loss = train_supervised_loss_total/len(self.train_data)
            avg_epoch_train_unsupervised_loss = train_unsupervised_loss_total/len(self.train_data)

            avg_train_acc = train_correct_total/len(self.train_data)
            avg_epoch_val_loss = val_loss_total/len(self.val_data)
            avg_val_acc = val_correct_total/len(self.val_data)

            self.core_context.train.report_training_metrics(
                steps_completed=e+1,
                metrics = {
                    'avg_train_acc':avg_train_acc.item(),
                    'avg_train_loss':avg_epoch_train_loss.item(),
                    'avg_train_supervised_loss':avg_epoch_train_supervised_loss.item(),
                    'avg_train_unsupervised_loss':avg_epoch_train_unsupervised_loss.item()
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
        if self.gpu_id in {'cpu',0}:
            print('epoch {} || Epoch_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(e,
        avg_epoch_train_loss,avg_epoch_val_loss))
        if self.gpu_id in {'cpu',0}:
            print(f"epoch {e} || Train Acc:", self.train_accs[-1]," || Val Acc:", self.val_accs[-1])


def ssl_train_setting(model:torch.nn.Module,
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
    d_train,d_unlabeled_train, d_val = get_pcap_ssl_and_val_non_ssl_dataset()
    print("Data Loading Done!")
    # batch_size = 4
    if dist:
        train_sampler = torch.utils.data.distributed.DistributedSampler(d_train)
        unl_train_sampler = torch.utils.data.distributed.DistributedSampler(d_unlabeled_train)
        val_sampler = torch.utils.data.distributed.DistributedSampler(d_val,shuffle=False)

        train_dataloader = torch.utils.data.DataLoader(
            d_train, batch_size=batch_size, pin_memory= True, sampler=train_sampler)
        train_unlabeled_dataloader = torch.utils.data.DataLoader(
            d_unlabeled_train, batch_size=batch_size, sampler=unl_train_sampler)
        val_dataloader = torch.utils.data.DataLoader(
            d_val, batch_size=4*batch_size, pin_memory=True,sampler=val_sampler)
    else:
        train_dataloader = torch.utils.data.DataLoader(
            d_train, batch_size=batch_size, shuffle=True)
        train_unlabeled_dataloader = torch.utils.data.DataLoader(
            d_unlabeled_train, batch_size=4*batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(
            d_val, batch_size=batch_size, shuffle=False)
    # print(next(iter()))
    # print(next(iter()))
    print(dict(Counter(d_train.target.tolist())))
    print(dict(Counter(d_val.target.tolist())))
    optimizer = get_optimizer(model)
    if not dist:
        init_seeds(0)
    if dist:
        trainer = SSLCoreAPITrainer(
        model=model,
        train_data=train_dataloader,
        unlabeled_train_data=train_unlabeled_dataloader,
        train_sampler=train_sampler,
        unlabeled_train_sampler=unl_train_sampler,
        val_data=val_dataloader,
        unlabeled_val_data=None,
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
        trainer = SSLCoreAPITrainer(
            model=model,
            train_data=train_dataloader,
            unlabeled_train_data=train_unlabeled_dataloader,
            train_sampler=None,
            unlabeled_train_sampler=None,
            val_data=val_dataloader,
            unlabeled_val_data=None,
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