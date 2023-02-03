import torch
import numpy as np
# import torch.utils.data as data
from sklearn.model_selection import train_test_split
from netml.pparser.parser import PCAP
import sys
import os
PATH = os.path.join(os.path.dirname(__file__))
print(PATH)
sys.path.insert(0,PATH)
from utils import extract_iat_features

from ssl_utils import split_ssl_data

class classify_anomaly_dataset(torch.utils.data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.data)
    
    def pullitem(self, index):
        # pull and normalize
        ex = self.data[index]
        ex -= torch.mean(ex)
        ex /= torch.std(ex)
        
        return ex
        
    def __getitem__(self, index):
        ex = self.pullitem(index)
        return ex, self.target[index]

class ssl_classify_anomaly_dataset(torch.utils.data.Dataset):
    def __init__(self, lb_data, lb_target, ulb_data, ulb_target):
        self.lb_data = lb_data
        self.lb_target = lb_target
        self.ulb_data = ulb_data
        self.ulb_target = ulb_target
        
    def __len__(self):
        return len(self.data)
    
    def pullitem(self, index):
        # pull and normalize
        lb_ex = self.lb_data[index]
        lb_ex -= torch.mean(lb_ex)
        lb_ex /= torch.std(lb_ex)

        ulb_ex = self.ulb_data[index]
        ulb_ex -= torch.mean(ulb_ex)
        ulb_ex /= torch.std(ulb_ex)

        return lb_ex,ulb_ex
        
    def __getitem__(self, index):
        lb_ex,ulb_ex = self.pullitem(index)
        return lb_ex, self.lb_target[index], ulb_ex, self.ulb_target[index]

def prepare_pcap_data(normal_pcap_path, abnormal_pcap_path):
    '''
    '''
    pcap_normal = PCAP(
    normal_pcap_path,
    flow_ptks_thres=2,
    random_state=42,
    verbose=10,
    )
    pcap_anomaly = PCAP(
        abnormal_pcap_path,
        flow_ptks_thres=2,
        random_state=42,
        verbose=10,
    )
    # print(df)

    # normal_feats = extract_iat_features(pcap_normal)
    # abnormal_feats = extract_iat_features(pcap_anomaly)
    # abnormal_feats=abnormal_feats[:,:-1]
    # np.save('data/normal_feats.npy', normal_feats) # save
    # np.save('data/abnormal_feats.npy', abnormal_feats) # save 
    # print("Loading...")   
    normal_feats = np.load('data/normal_feats.npy') # load
    abnormal_feats = np.load('data/abnormal_feats.npy') # load
    # print("Done Loading!")  
    x_train, x_val,y_train, y_val = split_dataset(normal_feats,abnormal_feats)

    return x_train, x_val,y_train, y_val

def split_dataset(normal_feats,abnormal_feats):
    lab = np.zeros(normal_feats.shape[0])
    lab_anom = np.ones(abnormal_feats.shape[0])
    print(lab.shape,lab_anom.shape)
    input = np.concatenate([normal_feats,abnormal_feats])
    labels = np.concatenate([lab,lab_anom])
    print(labels.shape)
    x_train, x_val,y_train, y_val = train_test_split(input,labels,test_size=0.2,stratify=labels)
    x_train = torch.from_numpy(x_train).type(torch.float32)
    x_val = torch.from_numpy(x_val).type(torch.float32)
    y_train = torch.LongTensor(y_train)
    y_val = torch.LongTensor(y_val)
    # print(y_val)
    return x_train, x_val,y_train, y_val

def load_and_prepare_pcap():
    '''
    '''
    normal_pcap_path = os.path.join(os.path.dirname(__file__),'../data/srcIP_10.42.0.1_normal.pcap')
    abnormal_pcap_path = os.path.join(os.path.dirname(__file__),'../data/srcIP_10.42.0.119_anomaly.pcap')
    x_train, x_val,y_train, y_val = prepare_pcap_data(normal_pcap_path, abnormal_pcap_path)
    x_train=x_train.unsqueeze(-1)# (N,12) -> (N,12,1)
    x_val=x_val.unsqueeze(-1)# (N,12) -> (N,12,1)
    
    d_train = classify_anomaly_dataset(x_train,y_train)
    d_val = classify_anomaly_dataset(x_val,y_val)
    return d_train, d_val

def load_and_prepare_pcap_numpy():
    '''
    '''
    normal_pcap_path = os.path.join(os.path.dirname(__file__),'../data/srcIP_10.42.0.1_normal.pcap')
    abnormal_pcap_path = os.path.join(os.path.dirname(__file__),'../data/srcIP_10.42.0.119_anomaly.pcap')
    x_train, x_val,y_train, y_val = prepare_pcap_data(normal_pcap_path, abnormal_pcap_path)
    x_train=x_train.unsqueeze(-1)# (N,12) -> (N,12,1)
    x_val=x_val.unsqueeze(-1)# (N,12) -> (N,12,1)
    return x_train, x_val,y_train, y_val

def get_ssl_pcap_dataset():
    '''
    Total for train dataset:
    class labels: 0: (4979,) 1: (310,) = 5,289
    train classes:  {0: 3983, 1: 248} = 4979
    labeled: 0.01*4979 = 44
    -- 4 examples = 2 per class!
    unlabeled: 0.9*4979 = 4400
    val classes:  {0: 996, 1: 62}   
    labeled: 0.01*310 = 4
    unlabeled: 0.9*310 = 270
    '''
    x_train, x_val,y_train, y_val = load_and_prepare_pcap_numpy()
    n_classes = len(np.unique(y_train))
    classes = {i:np.where(y_train==i)[0].shape[0] for i in range(n_classes)}
    print("train classes: ",classes )
    classes = {i:np.where(y_val==i)[0].shape[0] for i in range(n_classes)}
    print("val classes: ",classes )
    lb_num_labels=10
    ulb_num_labels=4400
    lb_imbalance_ratio=1.0
    ulb_imbalance_ratio=1.0



    lb_data, lb_targets, ulb_data, ulb_targets, lb_idx, ulb_idx = split_ssl_data(x_train,
                                                                y_train,
                                                                n_classes,
                                                                lb_num_labels,
                                                                ulb_num_labels,
                                                                lb_imbalance_ratio,
                                                                ulb_imbalance_ratio,
                                                                include_lb_to_ulb=True)
    train_ssl_pcap={
        'lb_data': torch.from_numpy(lb_data).type(torch.float32),
        'lb_targets': torch.LongTensor(lb_targets),
        'ulb_data': torch.from_numpy(ulb_data).type(torch.float32),
        'ulb_targets': torch.LongTensor(ulb_targets),
        'lb_idx': lb_idx,
        'ulb_idx': ulb_idx,
        'lb_class_distribution': {i:np.where(ulb_targets==i)[0].shape[0] for i in range(n_classes)},
        'ulb_class_distribution': {i:np.where(ulb_targets==i)[0].shape[0] for i in range(n_classes)}
    }
    print("Train Data: ")
    print("lb_data: ", train_ssl_pcap['lb_data'].shape)
    print("lb_targets: ", train_ssl_pcap['lb_targets'].shape)
    classes = {i:np.where(train_ssl_pcap['lb_targets']==i)[0].shape[0] for i in range(n_classes)}
    print("lb_targets classes: ",classes )
    print("ulb_data: ", train_ssl_pcap['ulb_data'].shape)
    print("ulb_targets: ", train_ssl_pcap['ulb_targets'].shape)
    classes = {i:np.where(train_ssl_pcap['ulb_targets']==i)[0].shape[0] for i in range(n_classes)}
    print("lb_targets classes: ",classes )

    print("Validation Data: ")

    lb_num_labels=4
    ulb_num_labels=270
    lb_imbalance_ratio=1.0
    ulb_imbalance_ratio=1.0
    lb_data, lb_targets, ulb_data, ulb_targets, lb_idx, ulb_idx = split_ssl_data(x_val,
                                                                y_val,
                                                                n_classes,
                                                                lb_num_labels,
                                                                ulb_num_labels,
                                                                lb_imbalance_ratio,
                                                                ulb_imbalance_ratio,
                                                                include_lb_to_ulb=True)
    val_ssl_pcap={
        'lb_data': torch.from_numpy(lb_data).type(torch.float32),
        'lb_targets': torch.LongTensor(lb_targets),
        'ulb_data': torch.from_numpy(ulb_data).type(torch.float32),
        'ulb_targets': torch.LongTensor(ulb_targets),
        'lb_idx': lb_idx,
        'ulb_idx': ulb_idx,
        'lb_class_distribution': {i:np.where(ulb_targets==i)[0].shape[0] for i in range(n_classes)},
        'ulb_class_distribution': {i:np.where(ulb_targets==i)[0].shape[0] for i in range(n_classes)}
    }
    print("Val Data: ")
    print("lb_data: ", val_ssl_pcap['lb_data'].shape)
    print("lb_targets: ", val_ssl_pcap['lb_targets'].shape)
    classes = {i:np.where(val_ssl_pcap['lb_targets']==i)[0].shape[0] for i in range(n_classes)}
    print("lb_targets classes: ",classes )
    print("ulb_data: ", val_ssl_pcap['ulb_data'].shape)
    print("ulb_targets: ", val_ssl_pcap['ulb_targets'].shape)
    classes = {i:np.where(val_ssl_pcap['ulb_targets']==i)[0].shape[0] for i in range(n_classes)}
    print("lb_targets classes: ",classes )
    
    return train_ssl_pcap, val_ssl_pcap

def get_pcap_ssl_and_val_non_ssl_dataset():
    '''
    '''
    train_ssl_pcap, val_ssl_pcap = get_ssl_pcap_dataset()

    '''
    train_ssl_pcap={
        'lb_data': lb_data,
        'lb_targets': lb_targets,
        'ulb_data': ulb_data,
        'ulb_targets': ulb_targets,
        'lb_idx': lb_idx,
        'ulb_idx': ulb_idx,
        'lb_class_distribution': {i:np.where(ulb_targets==i)[0].shape[0] for i in range(n_classes)},
        'ulb_class_distribution': {i:np.where(ulb_targets==i)[0].shape[0] for i in range(n_classes)}
    }
    Train Data: 
    lb_data:  torch.Size([4, 12, 1])
    lb_targets:  torch.Size([4])
    lb_targets classes:  {0: 2, 1: 2}
    ulb_data:  torch.Size([2450, 12, 1])
    ulb_targets:  torch.Size([2450])
    lb_targets classes:  {0: 2202, 1: 248}
    '''
    d_train = classify_anomaly_dataset(train_ssl_pcap['lb_data'],train_ssl_pcap['lb_targets'])
    d_unlabeled_train = classify_anomaly_dataset(train_ssl_pcap['ulb_data'],train_ssl_pcap['ulb_targets'])
    # d_val = classify_anomaly_dataset(val_ssl_pcap['lb_data'],val_ssl_pcap['lb_targets'])
    _,d_val = load_and_prepare_pcap()
    return d_train,d_unlabeled_train, d_val