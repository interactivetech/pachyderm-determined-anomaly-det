# from src import classify_conv_model
import torchvision
from torchvision import transforms
from collections import Counter
import torch
import numpy as np
from src.data import load_and_prepare_pcap_numpy, classify_anomaly_dataset, get_ssl_pcap_dataset, ssl_classify_anomaly_dataset
from src.ssl_utils import split_ssl_data, gen_pseudo_label, consistency_loss, ce_loss
from src.model import classify_conv_model
import torch.nn.functional as F


# print(classify_conv_model())

def get_cifar_dataset():
    '''
    '''
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data',
                                            train=True,
                                            download=True,
                                            transform=transform)
    # trainloader = torch.utils.data.DataLoader(trainset,
    #                                           batch_size=4,
    #                                           shuffle=True,
    #                                           num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            download=True,
                                            transform=transform)
    # testloader = torch.utils.data.DataLoader(testset,
    #                                          batch_size=4,
    #                                          shuffle=False,
    #                                          num_workers=2)
    # print(trainset)
    data, targets = trainset.data, trainset.targets
    return data, targets




cifar = False
if cifar:
    data, targets = get_cifar_dataset()
    targets = np.array(targets)
    lab_count = Counter(targets)
    print(lab_count)
    num_classes = len(lab_count.keys())
    print(len(data),len(targets),num_classes)#50000 examples 50000 labels
    lb_num_labels=50
    ulb_num_labels=5000
    lb_imbalance_ratio=1.0
    ulb_imbalance_ratio=1.0



    lb_data, lb_targets, ulb_data, ulb_targets, lb_idx, ulb_idx = split_ssl_data(data,
                                                                targets,
                                                                num_classes,
                                                                lb_num_labels,
                                                                ulb_num_labels,
                                                                lb_imbalance_ratio,
                                                                ulb_imbalance_ratio)
else:
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

    val_ssl_pcap={
        'lb_data': lb_data,
        'lb_targets': lb_targets,
        'ulb_data': ulb_data,
        'ulb_targets': ulb_targets,
        'lb_idx': lb_idx,
        'ulb_idx': ulb_idx,
        'lb_class_distribution': {i:np.where(ulb_targets==i)[0].shape[0] for i in range(n_classes)},
        'ulb_class_distribution': {i:np.where(ulb_targets==i)[0].shape[0] for i in range(n_classes)}
    }
    '''
    # Two seperate Data loaders!!
    d_train_lb = classify_anomaly_dataset(train_ssl_pcap['lb_data'],train_ssl_pcap['lb_targets'])
    d_train_ulb = classify_anomaly_dataset(train_ssl_pcap['ulb_data'],train_ssl_pcap['ulb_targets'])
    lb_ex,lb_ex_y = next(iter(d_train_lb))
    ulb_ex, ulb_ex_y = next(iter(d_train_ulb))
    print(lb_ex.shape,lb_ex_y.shape, ulb_ex.shape, ulb_ex_y.shape)
    
    d_val_lb = classify_anomaly_dataset(val_ssl_pcap['lb_data'],val_ssl_pcap['lb_targets'])
    d_val_ulb = classify_anomaly_dataset(val_ssl_pcap['ulb_data'],val_ssl_pcap['ulb_targets'])
    lb_ex,lb_ex_y = next(iter(d_val_lb))
    ulb_ex, ulb_ex_y = next(iter(d_val_ulb))
    print(lb_ex.shape,lb_ex_y.shape, ulb_ex.shape, ulb_ex_y.shape)
    # d_train = classify_anomaly_dataset(train_ssl_pcap['lb_data'],train_ssl_pcap['lb_targets'])
    # d_val = classify_anomaly_dataset(val_ssl_pcap['lb_data'],val_ssl_pcap['lb_targets'])
    m = classify_conv_model()
    o_l = m(lb_ex)
    o_ul = m(ulb_ex)

    # Pseudolabel
    h_l = gen_pseudo_label(o_ul)
    print("o_l: ",o_l)
    print("o_ul: ",o_ul)
    print("pseudo_label: ",h_l)
    print("lb_ex_y: ",lb_ex_y)
    # print("torch.nn.functional.one_hot(target): ",torch.nn.functional.one_hot(lb_ex_y))
    # test losses
    sup_loss = ce_loss(o_l,lb_ex_y.unsqueeze(-1),reduction='mean')
    print("sup_loss",sup_loss )
    l = F.cross_entropy(o_l,lb_ex_y.unsqueeze(-1))
    print("F.cross_entropy(o,lb_ex_y): ",l)
    consistency_l = consistency_loss(o_ul,h_l)
    print("consistency_l: ",consistency_l)

    total_loss = sup_loss + consistency_l
    print("total_loss: ",total_loss)
    
    

