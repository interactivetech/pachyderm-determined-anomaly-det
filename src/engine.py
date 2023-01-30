import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import os
import numpy as np
from data import classify_anomaly_dataset, prepare_pcap_data
from model import   classify_conv_model
from utils import get_optimizer
from collections import Counter
import torch.nn.functional as F
from torch.distributed import init_process_group,destroy_process_group
import sys
import shutil
import argparse
from utils import extract_iat_features,load_data
'''
python -m torch.distributed.launch --nproc_per_node=4 main_ddp.py 2>&1 | tee out.log
'''

def reduce_mean(tensor,nprocs):
    '''
    '''
    rt = tensor.clone()
    torch.distributed.all_reduce(rt,op=torch.distributed.ReduceOp.SUM)
    rt/=nprocs
    return rt
def ddp_setup(rank,world_size):
    '''
    rank: Unique identifier of each process
    world_size: total number of processes
    '''
    os.environ["MASTER_ADDR"] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    init_process_group(backend='nccl',rank=rank,world_size=world_size)

# def generate_ddp_file(trainer):
#     import_path = '.'.join(str(trainer.__class__).split(".")[1:-1])

#     if not trainer.resume:
#         shutil.rmtree(trainer.save_dir)  # remove the save_dir
#     content = f'''cfg = {vars(trainer.args)} \nif __name__ == "__main__":
#     from ultralytics.{import_path} import {trainer.__class__.__name__}
#     trainer = {trainer.__class__.__name__}(cfg=cfg)
#     trainer.train()'''
#     (USER_CONFIG_DIR / 'DDP').mkdir(exist_ok=True)
#     with tempfile.NamedTemporaryFile(prefix="_temp_",
#                                      suffix=f"{id(trainer)}.py",
#                                      mode="w+",
#                                      encoding='utf-8',
#                                      dir=USER_CONFIG_DIR / 'DDP',
#                                      delete=False) as file:
#         file.write(content)
#     return file.name

# def generate_ddp_command(world_size,trainer):
#     '''
#     '''
#     import __main__
#     file_name = os.path.abspath(sys.argv[0])
#     using_cli = not file_name.endswith("py")# for colab
#     if using_cli:
#         file_name = generate_ddp_file(trainer)
#     return [
#         sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", f"{world_size}",
#         "--master_port", "12355", file_name] + sys.argv[1]
parser = argparse.ArgumentParser(description='')
parser.add_argument('--local_rank', default=-1,type=int,help='node rank for dist training')
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
        epochs: int,
        resume_enabled: bool,
        nprocs: int
    )-> None:
        self.dist = dist
        self.nprocs = nprocs
        self.resume_enabled = resume_enabled
        self.batch_size = batch_size
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
            if self.dist:
                '''
                The function of barrier() is to block the process
                and ensure that each process runs all the code before this line of code before executing
                so that the average loss and average acc will not appear bec of the process execution speed
                inconsistent error
                '''
                torch.distributed.barrier()
                reduced_loss = reduce_mean(loss,self.nprocs)
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
        #(Andrew 1.27.23): Dont know how to resume training for DDP
        if self.resume_enabled and os.path.exists(SAVE_PATH) and not self.dist:
            # either restart checkpoint, or not
            # resulting in overwriting
            self._resume_checkpoint()
        
        
    def _run_epoch(self,e):
        '''
        '''
        self.model.train()
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
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss,preds = self._run_batch(source, targets,val=False)
            train_loss_total+=loss
            train_correct_total+=torch.sum(preds==targets.data)
            if self.dist:
                '''
                The function of barrier() is to block the process
                and ensure that each process runs all the code before this line of code before executing
                so that the average loss and average acc will not appear bec of the process execution speed
                inconsistent error
                '''
                torch.distributed.barrier()
                train_loss_total = reduce_mean(train_loss_total,self.nprocs)
                train_correct_total = reduce_mean(train_correct_total,self.nprocs)
        self.model.eval()
        for ind, (source, targets) in enumerate(self.val_data):
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss,preds = self._run_batch(source, targets,val=True)
            val_loss_total+=loss
            val_correct_total+=torch.sum(preds==targets.data)
            if self.dist:
                '''
                The function of barrier() is to block the process
                and ensure that each process runs all the code before this line of code before executing
                so that the average loss and average acc will not appear bec of the process execution speed
                inconsistent error
                '''
                torch.distributed.barrier()
                val_loss_total = reduce_mean(val_loss_total,self.nprocs)
                val_correct_total = reduce_mean(val_correct_total,self.nprocs)
                
        # print("val_correct_total: ",train_correct_total)
        # print("len(self.train_data): ",len(self.train_data.dataset))
        if self.gpu_id in {'cpu',0}:
            avg_epoch_train_loss = train_loss_total/len(self.train_data)
            avg_train_acc = train_correct_total/len(self.train_data.dataset)
            avg_epoch_val_loss = val_loss_total/len(self.val_data)
            avg_val_acc = val_correct_total/len(self.val_data.dataset)
            self.losses.append(avg_epoch_train_loss)
            self.train_accs.append(avg_train_acc)
            self.val_losses.append(avg_epoch_val_loss)
            self.val_accs.append(avg_val_acc)
        if self.gpu_id in {'cpu',0}:
            print('epoch {} || Epoch_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(e,
        avg_epoch_train_loss,avg_epoch_val_loss))
        if self.gpu_id in {'cpu',0}:
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
                print(f"Saving checkpoint at epoch {e}")
                self._save_checkpoint(e)

def main(rank:int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
    '''
    '''
    model = classify_conv_model()
    model = DDP(model,device_ids = [rank])
    print("PATH: ", os.path.dirname(__file__))
    normal_pcap_path = os.path.join(os.path.dirname(__file__),'../data/srcIP_10.42.0.1_normal.pcap')
    abnormal_pcap_path = os.path.join(os.path.dirname(__file__),'../data/srcIP_10.42.0.119_anomaly.pcap')
    x_train, x_val,y_train, y_val = prepare_pcap_data(normal_pcap_path, abnormal_pcap_path)
    x_train=x_train.unsqueeze(-1)# (N,12) -> (N,12,1)
    x_val=x_val.unsqueeze(-1)# (N,12) -> (N,12,1)
    
    d_train = classify_anomaly_dataset(x_train,y_train)
    d_val = classify_anomaly_dataset(x_val,y_val)

    batch_size = 4
    train_sampler = torch.utils.data.distributed.DistributedSampler(d_train)
    val_sampler = torch.utils.data.distributed.DistributedSampler(d_val)
    train_dataloader = torch.utils.data.DataLoader(
        d_train, batch_size=batch_size, pin_memory= True, sampler=train_sampler, shuffle=True)

    val_dataloader = torch.utils.data.DataLoader(
        d_val, batch_size=batch_size, shuffle=False, pin_memory=True,sampler=val_sampler)
    # print(next(iter()))
    print(dict(Counter(d_train.target.tolist())))
    print(dict(Counter(d_val.target.tolist())))
    optimizer = get_optimizer(model)
    trainer = BaseTrainer(
        model=model,
        train_data=train_dataloader,
        val_data=val_dataloader,
        optimizer=optimizer,
        gpu_id=rank,
        dist=True,
        save_every=1,
        model_dir='/Users/mendeza/Documents/2023_projects/pachyderm-determined-anomaly-det/models',
        epochs=30,
        resume_enabled=True,
        nprocs = world_size
    )
    trainer.train()
    
    # parser.add_argument('')
if __name__ == '__main__':
    single_gpu = True
    model = classify_conv_model()# Make DDP

    # x_train = np.random.rand(100,12)
    # print("x_train: ", x_train.shape)
    # y_train = np.zeros((100))
    # x_val = np.random.randn(20,12)
    # y_val = np.ones((20))

    # x_train = torch.from_numpy(x_train).type(torch.float32).unsqueeze(-1)
    # x_val = torch.from_numpy(x_val).type(torch.float32).unsqueeze(-1)
    # y_train = torch.LongTensor(y_train)
    # y_val = torch.LongTensor(y_val)
    print("PATH: ", os.path.dirname(__file__))
    normal_pcap_path = os.path.join(os.path.dirname(__file__),'../data/srcIP_10.42.0.1_normal.pcap')
    abnormal_pcap_path = os.path.join(os.path.dirname(__file__),'../data/srcIP_10.42.0.119_anomaly.pcap')
    x_train, x_val,y_train, y_val = prepare_pcap_data(normal_pcap_path, abnormal_pcap_path)
    x_train=x_train.unsqueeze(-1)# (N,12) -> (N,12,1)
    x_val=x_val.unsqueeze(-1)# (N,12) -> (N,12,1)
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
    if single_gpu:
        trainer = BaseTrainer(
            model=model,
            train_data=train_dataloader,
            val_data=val_dataloader,
            optimizer=optimizer,
            gpu_id='cpu',
            dist=False,
            save_every=1,
            model_dir='models',
            epochs=30,
            resume_enabled=True,
            nprocs=-1
        )
        trainer.train()
    else:
        args= parser.parse_args()
        args.nprocs = torch.cuda.device_count()
        main(args.local_rank,args.nprocs, args)