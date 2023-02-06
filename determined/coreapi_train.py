import determined as det
from src.utils import extract_iat_features,load_data, de_parallelize_model, init_seeds, accuracy, reduce_mean, ddp_setup
from src.model import   classify_conv_model
import torch
from pprint import pprint
from src.coreapi_engine import run_train
import argparse
import os
parser = argparse.ArgumentParser(description='')
parser.add_argument('--local_rank', default=-1,type=int,help='node rank for dist training')

def main(rank:int,
         world_size: int,
         model: torch.nn.Module,
         batch_size: int,
         resume_enabled: bool,
         epochs: int, 
         multi: bool, 
         ssl_train: bool,
         latest_ckpt: object,
         trial_id: int,
         hparams: dict
         ):
    '''
    '''
    if multi:
        torch.cuda.set_device(rank)
        ddp_setup(rank,world_size=world_size)
    #3. define distributed option for det.core.init()
    try:
        # required if using multiple GPUs
        distributed = det.core.DistributedContext.from_torch_distributed()
    except:
        distributed = None 

    # 4. create a context, and pass it to the main function
    with det.core.init(distributed=distributed) as core_context:
        # core_context, latest_checkpoint, trial_id
        run_train(model,
            batch_size,
            rank,
            multi,
            epochs,
            resume_enabled,
            nprocs=world_size,
            dist=multi,
            world_size=world_size,
            core_context=core_context, 
            latest_ckpt=latest_ckpt, 
            trial_id=trial_id)

def dist_main(rank:int,
         world_size: int,
         batch_size: int,
         resume_enabled: bool,
         epochs: int, 
         multi: bool, 
         ssl_train: bool,
         latest_ckpt: object,
         trial_id: int,
         hparams: dict
         ):
    '''
    '''
    # print("DDP Setup...")
    torch.cuda.set_device(rank)
    ddp_setup(rank,world_size=world_size)
    
    #3. define distributed option for det.core.init()
    try:
        # required if using multiple GPUs
        distributed = det.core.DistributedContext.from_torch_distributed()
    except:
        distributed = None 

    model = classify_conv_model()

    # 4. create a context, and pass it to the main function
    with det.core.init(distributed=distributed) as core_context:
        # core_context, latest_checkpoint, trial_id
        if ssl_train:
            # ssl_train_setting(model,batch_size,rank,multi,epochs,resume_enabled,nprocs=world_size,dist=multi,world_size=world_size)
            raise NotImplementedError
        else:
            # normal_train_setting(model,batch_size,rank,multi,epochs,resume_enabled,nprocs=world_size,dist=multi,world_size=world_size)
            raise NotImplementedError

    # parser.add_argument('')

if __name__ == '__main__':
    nprocs = torch.cuda.device_count()
    # 1. Get ClusterInfo API to gather info
    # about the task running on the cluster
    # specifically a checkpoint to load from and 
    # the TrialID
    info = det.get_cluster_info()
    print("Info")
    print(info)
    # 1a. check if we run in distributed training mode,
    # we suppose LOCAL_RANK is not defined if that's not the case
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        # args.local_rank=local_rank
        print("os.environ['LOCAL_RANK']", os.environ['LOCAL_RANK'])

    # check if running local machine
    if info is not None:
        trial_id = info.trial.trial_id
        if info.latest_checkpoint is not None:
            latest_ckpt = info.latest_checkpoint
        else:
            latest_ckpt = None
    else:
        latest_ckpt = None
        trial_id = -1
    print("info.latest_checkpoint: ",info.latest_checkpoint)
    print("latest_ckpt: ",latest_ckpt)
    print("trial_id: ",trial_id)
    # 2. Get hyperparameter values chosen for Trial
    # using ClusterInfo API
    hparams = info.trial.hparams
    #  non-hyperparameters like paths are usuallly found
    #  in the "data" part of the config file
    data_conf = info.user_data

    '''
    Hparams:
    batch_size
    multi - bool if doing DDP training
    epochs
    dist - bool, same as multi (redundant), use to set up DDP dataloaders 

    '''
    print("Hparams:")
    pprint(hparams)
    batch_size = hparams['batch_size']
    multi = hparams['multi']
    epochs = hparams['epochs']
    dist = hparams['dist']
    print("latest_ckpt: ",latest_ckpt)
    print("trial_id: ",trial_id)
    '''
    number of epochs will be defined from 
    searcher's max_length field in the config file
    set any value as it's still a mandatory argument 
    for main function
    '''
    if not multi:
        print("Entering Single GPU...")
        #ssl_train_setting(model,batch_size,device,multi,epochs,resume_enabled,nprocs=-1,dist=False,world_size=-1)
        # raise NotImplementedError
        device = 0 if torch.cuda.is_available() else "cpu"
        model = classify_conv_model()
        main(rank=device,
            model=model,
            world_size=-1,
            batch_size=batch_size,
            resume_enabled=False,
            epochs=epochs,
            multi=multi,
            ssl_train=False,
            latest_ckpt=latest_ckpt,
            trial_id=trial_id,
            hparams=hparams)
        
    else:
        '''
        init_seeds(args.local_rank)
        main(args.local_rank,args.nprocs,args.batch_size,resume_enabled,epochs,multi,ssl_train=ssl_train)
        '''
        print("Entering Multi GPU...")
        print("args.local_rank: ",local_rank)
        init_seeds(local_rank+1)
        model = classify_conv_model()
        main(rank=local_rank,
            model=model,
            world_size=nprocs,
            batch_size=batch_size,
            resume_enabled=False,
            epochs=epochs,
            multi=multi,
            ssl_train=False,
            latest_ckpt=latest_ckpt,
            trial_id=trial_id,
            hparams=hparams)