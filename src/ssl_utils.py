import numpy as np
import torch
import torch.nn.functional as F

def smooth_targets(logits, targets, smoothing=0.1):
    """
    label smoothing
    """
    with torch.no_grad():
        true_dist = torch.zeros_like(logits)
        # print("true_dist: ",true_dist)
        true_dist.fill_(smoothing / (logits.shape[-1] - 1))
        # print("true_dist: ",true_dist)
        # print("targets.data.unsqueeze(1): ",targets.data.unsqueeze(1))
        true_dist.scatter_(1, targets.data.unsqueeze(1), (1 - smoothing))
        # print("true_dist: ",true_dist)
    return true_dist


def ce_loss(logits, targets, reduction='none'):
    """
    wrapper for cross entropy loss in pytorch.
    Args:
        logits: logit values, shape=[Batch size, # of classes]
        targets: integer or vector, shape=[Batch size] or [Batch size, # of classes]
        # use_hard_labels: If True, targets have [Batch size] shape with int values. If False, the target is vector (default True)
        reduction: the reduction argument
    """
    if logits.shape == targets.shape:
        # one-hot target
        log_pred = F.log_softmax(logits, dim=-1)
        nll_loss = torch.sum(-targets * log_pred, dim=1)
        if reduction == 'none':
            return nll_loss
        else:
            return nll_loss.mean()
    else:
        log_pred = F.log_softmax(logits, dim=-1)
        return F.nll_loss(log_pred, targets, reduction=reduction)


def consistency_loss(logits, targets, name='ce', mask=None):
    """
    wrapper for consistency regularization loss in semi-supervised learning.
    Args:
        logits: logit to calculate the loss on and back-propagion, usually being the strong-augmented unlabeled samples
        targets: pseudo-labels (either hard label or soft label)
        name: use cross-entropy ('ce') or mean-squared-error ('mse') to calculate loss
        mask: masks to mask-out samples when calculating the loss, usually being used as confidence-masking-out
    """

    assert name in ['ce', 'mse']
    # logits_w = logits_w.detach()
    if name == 'mse':
        probs = torch.softmax(logits, dim=-1)
        loss = F.mse_loss(probs, targets, reduction='none').mean(dim=1)
    else:
        loss = ce_loss(logits, targets, reduction='none')

    if mask is not None:
        # mask must not be boolean type
        loss = loss * mask

    return loss.mean()
def gen_pseudo_label(logits,
                     use_hard_label=True,
                     T=1.0,
                     softmax=True,
                     label_smoothing=0.0):
    '''
    Args:
    - logits
    - use_hard_label
    - T
    - softmax
    - label_smoothing: bool to determine whether to compute softmax for logits. Note Input must be logits
    '''
    logits = logits.detach()
    if use_hard_label:
        pseudo_label = torch.argmax(logits,dim=-1)
        if label_smoothing:
            pseudo_label = smooth_targets(logits, pseudo_label, label_smoothing)
        return pseudo_label
    # return soft label
    if softmax:
        pseudo_label = torch.softmax(logits/T, dim=-1)
    else:
        pseudo_label = logits # input logits converted to probs
    return pseudo_label

def get_samples_per_class(num_classes:int,
                          lb_num_labels:int,
                          ulb_num_labels:int,
                          lb_imbalance_ratio:int,
                          ulb_imbalance_ratio:int,
                          targets):
    '''
    Args:
    - lb_num_labels
    - ulb_num_labels
    - lb_imbalanc_ratio
    - ulb_imbalance_ratio
    '''

    if lb_imbalance_ratio == 1.0:
        # you want the lb_samples per class to sum up to the lb_num_labels. 50 = sum([50/N]*N), 5 per class of 10 classes
        assert lb_num_labels % num_classes == 0, "lb_num_labels must be divisible by num_classes in balanved setting"
        lb_samples_per_class = [int(lb_num_labels/num_classes)]*num_classes# returns a list of labels per class
    else:
        #ToDo: integrate imbalance setting: 1-3, 3 unlabeled data for every labeled
        # lb_num_labels would be the maximum amount of labels a single class can have
        # https://github.com/determined-ai/demos/blob/8eb1ef0e0a7480230597256d17ffa8edd5535375/semi_supervised_learning/semi/datasets/utils.py#L67
        assert NotImplementedError
    
    if ulb_imbalance_ratio == 1.0:
        assert ulb_num_labels % num_classes == 0, "ulb_num_labels must be divisible by num_classes in balanced setting"
        ulb_samples_per_class = [int(ulb_num_labels/num_classes)]*num_classes
    else:
        #ToDo: integrate imbalance setting: 1-3, 3 unlabeled data for every labeled
        # lb_num_labels would be the maximum amount of labels a single class can have
        # https://github.com/determined-ai/demos/blob/8eb1ef0e0a7480230597256d17ffa8edd5535375/semi_supervised_learning/semi/datasets/utils.py#L80
        assert NotImplementedError
    
    print(lb_samples_per_class)
    print(ulb_samples_per_class)

    lb_idx = []
    ulb_idx = []

    for c in range(num_classes):
        idx = np.where(targets==c)[0]
        np.random.shuffle(idx)
        lb_idx.extend(idx[:lb_samples_per_class[c]])# keep the only K samples
        if ulb_samples_per_class is None:
            ulb_idx.extend(idx[lb_samples_per_class[c]:])
        else:
            ulb_idx.extend(idx[lb_samples_per_class[c] :lb_samples_per_class[c]+ulb_samples_per_class[c] ])
    return lb_idx, ulb_idx

def split_ssl_data(data,
                   targets,
                   num_classes,
                   lb_num_labels,
                   ulb_num_labels,
                   lb_imbalance_ratio,
                   ulb_imbalance_ratio,
                   include_lb_to_ulb=False):
    '''
    Args:
    - data
    - targets
    - num_classes: Total number of classes in dataset
    - lb_num_labels: Total number of examples in labeled dataset
    - ulb_num_labels: Total number of examples in unlabeled dataset
    - lb_imbalance_ratio: ratio of labeled data to unlabeled data
    - ulb_imbalance_ratio: ratio of unlabeled data to labeled data
    '''
    if not isinstance(data, np.ndarray) and not isinstance(data, np.ndarray):
        data, targets = np.array(data), np.array(targets)
    lb_idx, ulb_idx = get_samples_per_class(num_classes,
                                            lb_num_labels,
                                            ulb_num_labels,
                                            lb_imbalance_ratio,
                                            ulb_imbalance_ratio,
                                            targets)
    if include_lb_to_ulb:
        print("Include labeled input with unlabeled!")
        print(f"[lb_idx:{len(lb_idx)}] + [ulb_idx:{len(ulb_idx)}] = {len(lb_idx)+len(ulb_idx)}")
        ulb_idx = np.concatenate([lb_idx, ulb_idx], axis=0)
    # ToDo: Add option to include labeled input examples with unlabeled examples, for extra supervision!
    lb_data, lb_targets, ulb_data, ulb_targets =data[lb_idx], targets[lb_idx], data[ulb_idx], targets[ulb_idx]
    return lb_data, lb_targets, ulb_data, ulb_targets, lb_idx, ulb_idx