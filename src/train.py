
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import os
import numpy as np
from model import   classify_conv_model
from data import load_and_prepare_pcap
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
from engine import PytorchDDPTrainer, normal_train_setting
from ssl_engine import SSLPytorchDDPTrainer, ssl_train_setting, normal_train_setting_with_ssl_data
'''
python -m torch.distributed.launch --nproc_per_node=4 main_ddp.py 2>&1 | tee out.log
'''
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
# os.environ["MASTER_ADDR"] = '127.0.0.1'
# os.environ['MASTER_PORT'] = '12355'

parser = argparse.ArgumentParser(description='')
parser.add_argument('--local_rank', default=-1,type=int,help='node rank for dist training')
parser.add_argument('--multi',nargs='?', const=True, default=False,help='node rank for dist training')
parser.add_argument('--batch_size', default=4,type=int,help='batch size')
parser.add_argument('--resume',nargs='?', const=True, default=False,help='resume training')
parser.add_argument('--epochs', default=1,type=int,help='batch size')
parser.add_argument('--ssl_train',nargs='?', const=True, default=False,help='resume training')


def main(rank:int, world_size: int,batch_size: int,resume_enabled: bool, epochs: int, multi: bool):
    '''
    '''
    # print("DDP Setup...")
    torch.cuda.set_device(rank)
    ddp_setup(rank,world_size=world_size)

    model = classify_conv_model()

    d_train, d_val = load_and_prepare_pcap()

    # batch_size = 4
    train_sampler = torch.utils.data.distributed.DistributedSampler(d_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(d_val,shuffle=False)
    train_dataloader = torch.utils.data.DataLoader(
        d_train, batch_size=batch_size, pin_memory= True, sampler=train_sampler)

    val_dataloader = torch.utils.data.DataLoader(
        d_val, batch_size=batch_size, pin_memory=True,sampler=val_sampler)
    # print(next(iter()))
    print(dict(Counter(d_train.target.tolist())))
    print(dict(Counter(d_val.target.tolist())))
    # print("Loading Optimizer...")
    
    optimizer = get_optimizer(model)
    # print("Optimizer Loaded!")

    trainer = PytorchDDPTrainer(
        model=model,
        train_data=train_dataloader,
        train_sampler=train_sampler,
        val_data=val_dataloader,
        val_sampler=val_sampler,
        optimizer=optimizer,
        gpu_id=rank,
        dist=multi,
        save_every=1,
        model_dir='models/',
        epochs=epochs,
        resume_enabled=resume_enabled,
        nprocs = world_size
    )
    trainer.train()
    
    # parser.add_argument('')
if __name__ == '__main__':
    args= parser.parse_args()
    args.nprocs = torch.cuda.device_count()
    single_gpu=not args.multi
    batch_size=args.batch_size
    resume_enabled = args.resume
    epochs = args.epochs
    multi=args.multi
    ssl_train=args.ssl_train
    print("args.multi: ",args.multi)
    print("single_gpu: ",single_gpu)
    print("args.resume: ",args.resume)
    print(args.local_rank,args.nprocs)
    if single_gpu and not ssl_train:
        device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        model = classify_conv_model()# Make DDP
        normal_train_setting(model,batch_size,device,multi,epochs,resume_enabled,nprocs=-1)
    elif single_gpu and ssl_train:
        model = classify_conv_model()
        device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"
        ssl_setting=True
        if ssl_setting:
            print("Entering SSL Setting...")
            ssl_train_setting(model,batch_size,device,multi,epochs,resume_enabled,nprocs=-1)
        else:
            print("Entering Normal Setting with limited data...")
            normal_train_setting_with_ssl_data(model,batch_size,device,multi,epochs,resume_enabled,nprocs=-1)
    else:
        # print("single_gpu: ",single_gpu)
        init_seeds(args.local_rank)
        # print("Running Main...")

        main(args.local_rank,args.nprocs,args.batch_size,resume_enabled,epochs,multi)
        # mp.spawn(main, args=(args.local_rank,args.nprocs), nprocs=args.nprocs)

