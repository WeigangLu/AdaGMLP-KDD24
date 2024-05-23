import os
import dgl
import torch
import shutil
import random
import numpy as np


def get_device(param):
    return torch.device('cuda:{}'.format(param["device"]) if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def check_writable(path, overwrite=True):
    if not os.path.exists(path):
        os.makedirs(path)
    elif overwrite:
        shutil.rmtree(path)
        os.makedirs(path)
    else:
        pass


def idx_split(idx, ratio, seed=0):
    set_seed(seed)
    n = len(idx)
    cut = int(n * ratio)
    idx_idx_shuffle = torch.randperm(n)

    idx1_idx, idx2_idx = idx_idx_shuffle[:cut], idx_idx_shuffle[cut:]
    idx1, idx2 = idx[idx1_idx], idx[idx2_idx]

    return idx1, idx2


def graph_split(idx_train, idx_val, idx_test, labels, param):
    if param['dataset'] == 'cora' or param['dataset'] == 'citeseer' or param['dataset'] == 'pubmed':
        idx_test_ind = idx_test
        idx_test_tran = torch.tensor(list(
            set(torch.randperm(labels.shape[0]).tolist()) - set(idx_train.tolist()) - set(idx_val.tolist()) - set(
                idx_test_ind.tolist())))
    else:
        idx_test_ind, idx_test_tran = idx_split(idx_test, param['split_rate'], param['seed'])

    idx_obs = torch.cat([idx_train, idx_val, idx_test_tran])
    N1, N2 = idx_train.shape[0], idx_val.shape[0]
    obs_idx_all = torch.arange(idx_obs.shape[0])
    obs_idx_train = obs_idx_all[:N1]
    obs_idx_val = obs_idx_all[N1: N1 + N2]
    obs_idx_test = obs_idx_all[N1 + N2:]

    return obs_idx_train, obs_idx_val, obs_idx_test, idx_obs, idx_test_ind


def get_evaluator(dataset):
    def evaluator(out, labels):
        pred = out.argmax(1)
        return pred.eq(labels).float().mean().item()

    return evaluator


def extract_indices(g):
    edge_idx_loop = g.adjacency_matrix(transpose=True)._indices()
    edge_idx_no_loop = dgl.remove_self_loop(g).adjacency_matrix(transpose=True)._indices()
    edge_idx = (edge_idx_loop, edge_idx_no_loop)

    return edge_idx


def mask_features(X, p):
    """
    Randomly masks a portion of features in each row of feature matrix X based on the probability p.

    Args:
        X: Feature matrix of shape (N, C) where N is the number of samples and C is the number of features.
        p: The proportion of features to be randomly masked, should be a value between 0 and 1.

    Returns:
        masked_X: The processed feature matrix with the same shape as X.
    """
    N, C = X.shape
    num_features_to_mask = int(C * p)
    mask = torch.rand(N, C).to(X.device)
    _, indices = mask.topk(num_features_to_mask, dim=1)
    masked_X = X.clone()
    masked_X.scatter_(1, indices, 1e-32)

    # Compute the sum of elements along each row (axis=1)
    row_sums = masked_X.sum(dim=1, keepdim=True)
    # Normalize each row by dividing by its sum
    normalized_X = masked_X / row_sums

    return normalized_X