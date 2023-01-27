import torch
import numpy as np
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from netml.pparser.parser import PCAP
from utils import extract_iat_features

class classify_anomaly_dataset(data.Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
        
    def __len__(self):
        return len(self.data)
    
    def pullitem(self, index):
        # pull and normalize
        img = self.data[index]
        img -= torch.mean(img)
        img /= torch.std(img)
        
        return img
        
    def __getitem__(self, index):
        img = self.pullitem(index)
        return img, self.target[index]

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

    normal_feats = extract_iat_features(pcap_normal)
    abnormal_feats = extract_iat_features(pcap_anomaly)
    abnormal_feats=abnormal_feats[:,:-1]
    np.save('data/normal_feats.npy', normal_feats) # save
    np.save('data/abnormal_feats.npy', abnormal_feats) # save    
    normal_feats = np.load('data/normal_feats.npy') # load
    abnormal_feats = np.load('data/abnormal_feats.npy') # load
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