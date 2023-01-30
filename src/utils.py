import torch
from netml.pparser.parser import PCAP
import torch.optim as optim
import random
import numpy as np
import torch.backends.cudnn as cudnn
from torch.distributed import init_process_group,destroy_process_group
import datetime
import torch.nn as nn
def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=5e-4)

def extract_iat_features(pcap):
    pcap.pcap2flows(q_interval=0.9)
    pcap.flow2features('IAT', fft=False, header=False)
    return pcap.features

def load_data(pcap_normal_path, pcap_anomaly_path):
    '''
    '/Users/mendeza/Documents/2023_projects/netml/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.1_normal.pcap'
    '/Users/mendeza/Documents/2023_projects/netml/examples/data/srcIP_10.42.0.1/srcIP_10.42.0.119_anomaly2.pcap'
    '''
    pcap_normal = PCAP(
    pcap_normal_path,
    flow_ptks_thres=2,
    random_state=42,
    verbose=10,
    )
    pcap_anomaly = PCAP(
        pcap_anomaly_path,
        flow_ptks_thres=2,
        random_state=42,
        verbose=10,
    )
    # print(df)

    normal_feats = extract_iat_features(pcap_normal)
    abnormal_feats = extract_iat_features(pcap_anomaly)
    # print("normal_feats: ",normal_feats.shape)
    return normal_feats, abnormal_feats, pcap_normal, pcap_anomaly

def normalize(input):
    input = torch.FloatTensor(input)
    input -= torch.mean(input)
    input /= torch.std(input)
    return input.unsqueeze(0).unsqueeze(-1) 
def predict_model(model,input):
    output = model(input)
    cl = torch.argmax(output)
    print(cl.item(),output[0,cl.item()].item())
    return cl.item(),output[0,cl].item()

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
def init_seeds(seed=0, cuda_deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True
def accuracy_old(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        # print("output: ",output.shape)
        # print("target: ",target.data.shape)

        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred)).contiguous()

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def accuracy(output, target):
    """Computes the accuracy"""
    with torch.no_grad():
        batch_size = target.size(0)
        # print("output: ",output.shape)
        # print("target: ",target.data.shape)
        acc = torch.sum(output==target).float()
        res = acc.mul_(1/batch_size)
        return [res]
def reduce_mean(tensor,nprocs):
    '''
    '''
    rt = tensor.clone()
    torch.distributed.all_reduce(rt,op=torch.distributed.ReduceOp.SUM)
    rt/=nprocs
    return rt
def reduce_acc_mean(tensor,len_data,nprocs):
    '''
    '''
    rt = tensor.clone()
    torch.distributed.all_reduce(rt,op=torch.distributed.ReduceOp.SUM)
    rt/=(nprocs*len_data)
    return rt
def ddp_setup(rank,world_size):
    '''
    rank: Unique identifier of each process
    world_size: total number of processes
    '''

    init_method = 'env://'

    init_process_group(backend='nccl',
                       rank=rank,
                       world_size=world_size,
                       timeout=datetime.timedelta(seconds=18000),
                       init_method=init_method)
