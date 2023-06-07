import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from math import cos, pi
from copy import deepcopy
import sys
import torch.nn.functional as F
# from image_classification.compute_flops import print_model_param_flops
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
# from . import logger as log
import time
from torch.distributions import MultivariateNormal
from . import utils

l1 = [1]
l2 = [2,3,4,5,6,7,8]
l3 = [9,10,11]
skip = [12,13]
prev_layers = [None,None,1,2,3,4,5,6,7,8,5,2,(9,10),(9,10,11)]

l1_3d = [1]
l2_3d = [2,3,4,5,6,7,8,9,10]
l3_3d = [11,12]
skip_3d = [13]
prev_layers_3d = [None,None,1,2,3,4,5,6,7,5,3,1,1,11]
conv_3d_layers = [1,2,3,4,5,6,7,11,12,13]
trans_conv_3d_layers = [8,9,10]
pair_layers_3d = {5:8,3:9,1:10,10:1,9:3,8:5}

def filter_all_conv3d(model):
    filtered = [None]
    for m in model.modules():
        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            filtered.append(m)
    return filtered

def filter_all_bn3d(model):
    filtered = [None]
    for m in model.modules():
        if isinstance(m, nn.BatchNorm3d):
            filtered.append(m)
    return filtered

def L1_norm(layer):
    weight_copy = layer.weight.data.abs().clone().cpu().numpy()
    if isinstance(layer, nn.Conv2d):
        norm = np.sum(weight_copy, axis=(1,2,3))
    elif isinstance(layer, nn.Conv3d):
        norm = np.sum(weight_copy, axis=(1,2,3,4))
    elif isinstance(layer, nn.ConvTranspose3d):
        norm = np.sum(weight_copy, axis=(0,2,3,4))
    else:
        assert 0
    return norm #(out_channels,)

'''
def Laplacian(layer):
    weight = layer.weight.data.detach()
    x = weight.view(weight.shape[0], -1)
    X_inner = torch.matmul(x, x.t())
    X_norm = torch.diag(X_inner, diagonal=0)
    X_dist_sq = X_norm + torch.reshape(X_norm, [-1,1]) - 2 * X_inner
    X_dist = torch.sqrt(X_dist_sq)
    laplace = torch.sum(X_dist, dim=0).cpu().numpy()
    return laplace
'''
    

def CSS(layer, k, pair_layer = None):
    '''
    k: pruning rate, i.e. select (1-k)*C columns
    '''
    weight = layer.weight.data.detach()
    if isinstance(layer, nn.ConvTranspose3d):
        weight = torch.transpose(weight,0,1)
    if pair_layer != None:
        pair_weight = pair_layer.weight.data.detach()
        if isinstance(pair_layer, nn.ConvTranspose3d):
            pair_weight = torch.transpose(pair_weight,0,1)
    
    if pair_layer == None:
        X = weight.contiguous().view(weight.contiguous().shape[0], -1)
    else:
        #print(weight.shape)
        #print(pair_weight.shape)
        weight1 = weight.contiguous().view(weight.contiguous().shape[0],-1)
        weight2 = pair_weight.contiguous().view(pair_weight.contiguous().shape[0],-1)
        assert weight1.shape[0] == weight2.shape[0]
        X = torch.cat((weight1, weight2), dim=1)

    X = torch.transpose(X, 0, 1)
    if X.shape[0] >= X.shape[1]:
        _, _, V = torch.svd(X, some=True)
        Vk = V[:,:int((1-k)*X.shape[1])]
        lvs = torch.norm(Vk, dim=1)
        lvs = lvs.cpu().numpy()
        return lvs
    else:
        weight_copy = layer.weight.data.abs().clone().cpu().numpy()
        if isinstance(layer, nn.Conv2d):
            norm = np.sum(weight_copy, axis=(1,2,3))
        elif isinstance(layer, nn.Conv3d):
            norm = np.sum(weight_copy, axis=(1,2,3,4))
        elif isinstance(layer, nn.ConvTranspose3d):
            norm = np.sum(weight_copy, axis=(0,2,3,4))
        return norm


def get_layer_ratio (model, sparsity):
    total = 0
    bn_count = 1
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if bn_count in l1 + l2:
                total += m.weight.data.shape[0]
                bn_count += 1
                continue
            bn_count += 1
        if isinstance(m, nn.BatchNorm3d):
            if bn_count in l1_3d + l2_3d:
                if bn_count in pair_layers_3d and pair_layers_3d[bn_count] < bn_count:
                    pass
                else:
                    total += m.weight.data.shape[0]
            bn_count += 1
    bn = torch.zeros(total)
    index = 0
    bn_count = 1
    indexes = [0]*20
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if bn_count in l1 + l2:
                size = m.weight.data.shape[0]
                bn[index:(index+size)] = m.weight.data.abs().clone()
                index += size
                bn_count += 1
                continue
            bn_count += 1
        if isinstance(m, nn.BatchNorm3d):
            if bn_count in l1_3d + l2_3d:
                if bn_count in pair_layers_3d and pair_layers_3d[bn_count] < bn_count:
                    size = m.weight.data.shape[0]
                    my_index = indexes[pair_layers_3d[bn_count]]
                    bn[my_index:(my_index+size)] += m.weight.data.abs().cpu()
                    bn[my_index:(my_index+size)] = bn[my_index:(my_index+size)]/2
                else:
                    size = m.weight.data.shape[0]
                    bn[index:(index+size)] = m.weight.data.abs().cpu()
                    indexes[bn_count] = index
                    index += size
            bn_count += 1
    y, i = torch.sort(bn)
    thre_index = int(total * sparsity)
    thre = y[thre_index]
    layer_ratio = []
    bn_count = 1
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            if bn_count in l1 + l2:
                weight_copy = m.weight.data.abs().clone()
                mask = weight_copy.gt(thre).float().cuda()
                mask_sum = torch.sum(mask).item()
                if mask_sum > 0:
                    layer_ratio.append((mask.shape[0] - mask_sum) / mask.shape[0])
                else:
                    layer_ratio.append((mask.shape[0] - 1) / mask.shape[0])
                bn_count += 1
                continue
            bn_count += 1
    all_bn_3d = filter_all_bn3d(model)
    if len(all_bn_3d) <= 1:
        assert len(layer_ratio) > 0
        return layer_ratio
    for bn_count, m in enumerate(all_bn_3d):
        if bn_count in l1_3d + l2_3d:
            if bn_count in pair_layers_3d:
                pair_m = all_bn_3d[pair_layers_3d[bn_count]]
                weight_copy = (m.weight.data.abs().clone()+pair_m.weight.data.abs().clone())/2
            else:
                weight_copy = m.weight.data.abs().clone()
            mask = weight_copy.gt(thre).float().cuda()
            mask_sum = torch.sum(mask).item()
            if mask_sum > 0:
                layer_ratio.append((mask.shape[0] - mask_sum) / mask.shape[0])
            else:
                layer_ratio.append((mask.shape[0] - 1) / mask.shape[0])
    return layer_ratio
    

def init_channel_mask(model, ratio): #ratio: ratio of channels to be pruned
    prev_model = deepcopy(model)
    layer_id = 1
    cfg_mask = [None]*15
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in l1 + l2:
                num_keep = int(out_channels * (1 - ratio))
                rank = np.argsort(L1_norm(m)) # (out_channels,)
                arg_max_rev = rank[::-1][:num_keep] # (num_keep, )
                mask = torch.zeros(out_channels) # (out_channels,)
                mask[arg_max_rev.tolist()] = 1
                cfg_mask[layer_id] = mask
                layer_id += 1
                continue
            layer_id += 1
    all_conv3d = filter_all_conv3d(model)
    if len(all_conv3d) <= 1:
        return cfg_mask, prev_model
    for layer_id, m in enumerate(all_conv3d):
        if isinstance(m, nn.Conv3d):
            out_channels = m.weight.data.shape[0]
        elif isinstance(m, nn.ConvTranspose3d):
            out_channels = m.weight.data.shape[1]
        if layer_id in l1_3d + l2_3d:
            num_keep = int(out_channels*(1-ratio))
            if layer_id not in pair_layers_3d:
                rank = np.argsort(L1_norm(m))
                assert len(rank) == out_channels
            else:
                pair_m = all_conv3d[pair_layers_3d[layer_id]]
                rank = np.argsort(L1_norm(m) + L1_norm(pair_m)) 
                assert len(rank) == out_channels   
            arg_max_rev = rank[::-1][:num_keep]
            mask = torch.zeros(out_channels)
            mask[arg_max_rev.tolist()] = 1
            cfg_mask[layer_id] = mask

    return cfg_mask, prev_model


def apply_channel_mask(model, cfg_mask):
    # layer_id_in_cfg = 0
    conv_count = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if conv_count in l1:
                mask = cfg_mask[conv_count].float().cuda()
                mask = mask.view(m.weight.data.shape[0],1,1,1)
                m.weight.data.mul_(mask)
                # layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count in l2:
                mask = cfg_mask[conv_count].float().cuda()
                mask = mask.view(m.weight.data.shape[0],1,1,1)
                m.weight.data.mul_(mask)
                prev_mask = cfg_mask[prev_layers[conv_count]].float().cuda()
                prev_mask = prev_mask.view(1,m.weight.data.shape[1],1,1)
                m.weight.data.mul_(prev_mask)
                # layer_id_in_cfg += 1
                conv_count += 1
                continue
            if conv_count in l3:
                prev_mask = cfg_mask[prev_layers[conv_count]].float().cuda()
                prev_mask = prev_mask.view(1,m.weight.data.shape[1],1,1)
                m.weight.data.mul_(prev_mask)
                conv_count += 1
                continue
            if conv_count in skip:
                conv_count += 1
                continue
            conv_count += 1
        elif isinstance(m, nn.BatchNorm2d):
            if conv_count-1 in l1 + l2:
                mask = cfg_mask[conv_count-1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)
                continue
        elif isinstance(m, nn.Conv3d):
            if conv_count in l1_3d:
                mask = cfg_mask[conv_count].float().cuda()
                mask = mask.view(m.weight.data.shape[0],1,1,1,1)
                m.weight.data.mul_(mask)
            elif conv_count in l2_3d:
                mask = cfg_mask[conv_count].float().cuda()
                mask = mask.view(m.weight.data.shape[0],1,1,1,1)
                m.weight.data.mul_(mask)
                prev_mask = cfg_mask[prev_layers_3d[conv_count]].float().cuda()
                prev_mask = prev_mask.view(1,m.weight.data.shape[1],1,1,1)
                m.weight.data.mul_(prev_mask)
            elif conv_count in l3_3d:
                prev_mask = cfg_mask[prev_layers_3d[conv_count]].float().cuda()
                prev_mask = prev_mask.view(1,m.weight.data.shape[1],1,1,1)
                m.weight.data.mul_(prev_mask)
            conv_count += 1
        elif isinstance(m, nn.ConvTranspose3d):
            if conv_count in l1_3d:
                mask = cfg_mask[conv_count].float().cuda()
                mask = mask.view(1,m.weight.data.shape[1],1,1,1)
                m.weight.data.mul_(mask)
            elif conv_count in l2_3d:
                mask = cfg_mask[conv_count].float().cuda()
                mask = mask.view(1,m.weight.data.shape[1],1,1,1)
                m.weight.data.mul_(mask)
                prev_mask = cfg_mask[prev_layers_3d[conv_count]].float().cuda()
                prev_mask = prev_mask.view(m.weight.data.shape[0],1,1,1,1)
                m.weight.data.mul_(prev_mask)
            elif conv_count in l3_3d:
                prev_mask = cfg_mask[prev_layers_3d[conv_count]].float().cuda()
                prev_mask = prev_mask.view(m.weight.data.shape[0],1,1,1,1)
                m.weight.data.mul_(prev_mask)
            conv_count += 1
        elif isinstance(m, nn.BatchNorm3d):
            if conv_count-1 in l1_3d + l2_3d:
                mask = cfg_mask[conv_count-1].float().cuda()
                m.weight.data.mul_(mask)
                m.bias.data.mul_(mask)        


def detect_channel_zero (model):
    total_zero = 0
    total_c = 0
    conv_count = 1
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if conv_count in l1 + l2:
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                norm = np.sum(weight_copy, axis=(1,2,3)) # (out_channels,)
                total_zero += len(np.where(norm == 0)[0])
                total_c += m.weight.data.shape[0]
                conv_count += 1
                continue
            conv_count += 1
        elif isinstance(m, nn.Conv3d):
            if conv_count in l1_3d + l2_3d:
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                norm = np.sum(weight_copy, axis=(1,2,3,4)) # (out_channels,)
                total_zero += len(np.where(norm == 0)[0])
                total_c += m.weight.data.shape[0]
            conv_count += 1
        elif isinstance(m, nn.ConvTranspose3d):
            if conv_count in l1_3d + l2_3d:
                weight_copy = m.weight.data.abs().clone().cpu().numpy()
                norm = np.sum(weight_copy, axis=(0,2,3,4)) # (out_channels,)
                total_zero += len(np.where(norm == 0)[0])
                total_c += m.weight.data.shape[1]
            conv_count += 1
    return total_zero / total_c


def projection_formula(M):
    scatter = torch.matmul(M.t(), M)
    inv = torch.pinverse(scatter)
    return torch.matmul(torch.matmul(M, inv), M.t())


def IS_update_channel_mask(model, layer_ratio_up, layer_ratio_down, old_model):
    layer_id = 1
    idx = 0
    cfg_mask = [None]*20
    copy_indexes = [None]*20
    for [m, m0] in zip(model.modules(), old_model.modules()):
        if isinstance(m, nn.Conv2d):
            out_channels = m.weight.data.shape[0]
            if layer_id in l1:
                # number of channels
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep
                # pruning criterion
                rank = np.argsort(CSS(m, layer_ratio_down[idx])) # (out_channels,)
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                # restore MRU
                copy_idx = np.where(L1_norm(m) == 0)[0]
                copy_indexes[layer_id] = copy_idx
                w = m0.weight.data[copy_idx.tolist(), :, :, :].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                # importance sampling
                weight_copy = m.weight.data.detach().cpu()
                weight_copy = weight_copy.view(weight_copy.shape[0], -1)
                weight_copy = torch.transpose(weight_copy, 0, 1)
                base_weight = weight_copy[:, selected.tolist()]
                proj = projection_formula(base_weight)
                candidate = weight_copy[:, freedom.tolist()]
                candidate_prime = torch.matmul(proj, candidate)
                sampling_prob = F.softmax(torch.norm(candidate - candidate_prime, dim=0), dim=0)
                if num_free <= 0:
                    grow = np.random.permutation(freedom)[:num_free]
                else: 
                    grow = freedom[np.unique(torch.multinomial(sampling_prob, num_free).numpy())]
                # channel mask
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask[layer_id] = mask
                layer_id += 1
                idx += 1
                continue
            if layer_id in l2:
                # number of channels
                num_keep = int(out_channels*(1-layer_ratio_down[idx]))
                num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep
                # pruning criterion
                rank = np.argsort(CSS(m, layer_ratio_down[idx]))
                selected = rank[::-1][:num_keep]
                freedom = rank[::-1][num_keep:]
                # restore MRU
                prev_copy_idx = deepcopy(copy_indexes[prev_layers[layer_id]])
                copy_idx = np.where(L1_norm(m) == 0)[0]
                copy_indexes[layer_id] = copy_idx
                w = m0.weight.data[:,prev_copy_idx.tolist(),:,:].clone()
                m.weight.data[:,prev_copy_idx.tolist(),:,:] = w.clone()
                w = m0.weight.data[copy_idx.tolist(),:,:,:].clone()
                m.weight.data[copy_idx.tolist(),:,:,:] = w.clone()
                # importance sampling
                weight_copy = m.weight.data.detach().cpu()
                weight_copy = weight_copy.view(weight_copy.shape[0], -1)
                weight_copy = torch.transpose(weight_copy, 0, 1)
                base_weight = weight_copy[:, selected.tolist()]
                proj = projection_formula(base_weight)
                candidate = weight_copy[:, freedom.tolist()]
                candidate_prime = torch.matmul(proj, candidate)
                sampling_prob = F.softmax(torch.norm(candidate - candidate_prime, dim=0), dim=0) # (out_channels-num_keep,)
                if num_free <= 0:
                    grow = np.random.permutation(freedom)[:num_free]
                else: 
                    grow = freedom[np.unique(torch.multinomial(sampling_prob, num_free).numpy())]
                # channel mask
                mask = torch.zeros(out_channels)
                mask[selected.tolist() + grow.tolist()] = 1
                cfg_mask[layer_id] = mask
                layer_id += 1
                idx += 1
                continue
            if layer_id in l3:
                copy_idx = copy_indexes[prev_layers[layer_id]]
                w = m0.weight.data[:,copy_idx.tolist(),:,:].clone()
                m.weight.data[:,copy_idx.tolist(),:,:] = w.clone()
                layer_id += 1
                continue
            if layer_id in skip:
                layer_id += 1
                continue
            layer_id += 1
        elif isinstance(m, nn.BatchNorm2d):
            if layer_id-1 in l1 +l2:
                copy_idx = copy_indexes[layer_id-1]
                w = m0.weight.data[copy_idx.tolist()].clone()
                m.weight.data[copy_idx.tolist()] = w.clone()
                b = m0.bias.data[copy_idx.tolist()].clone()
                m.bias.data[copy_idx.tolist()] = b.clone()
                rm = m0.running_mean[copy_idx.tolist()].clone()
                m.running_mean[copy_idx.tolist()] = rm.clone()
                rv = m0.running_var[copy_idx.tolist()].clone()
                m.running_var[copy_idx.tolist()] = rv.clone()
                continue
    all_conv3d = filter_all_conv3d(model)
    prev_all_conv3d = filter_all_conv3d(old_model)
    all_bn3d = filter_all_bn3d(model)
    prev_all_bn3d = filter_all_bn3d(old_model)
    if len(all_conv3d) <= 1:
        assert len(prev_all_conv3d) <= 1
        assert len(all_bn3d) <= 1
        assert len(prev_all_bn3d) <= 1
        prev_model = deepcopy(model)
        return cfg_mask, prev_model
    # print(all_conv3d, all_bn3d)
    assert len(all_conv3d) == len(all_bn3d) + 1, f"{len(all_conv3d)}, {len(all_bn3d)}, {len(list(model.modules()))}"
    assert len(prev_all_conv3d) == len(prev_all_bn3d) + 1, f"{len(prev_all_conv3d)}, {len(prev_all_bn3d)}"
    layer_id = 0
    idx = 0
    cfg_mask = [None]*20
    copy_indexes = [None]*20
    for [m,m0] in zip(all_conv3d, prev_all_conv3d):
        if isinstance(m, nn.Conv3d):
            out_channels = m.weight.data.shape[0]
        elif isinstance(m, nn.ConvTranspose3d):
            out_channels = m.weight.data.shape[1]
        if layer_id in l1_3d:
            if layer_id in pair_layers_3d and layer_id >= pair_layers_3d[layer_id]:
                copy_idx = np.where(L1_norm(m) == 0)[0]
                copy_indexes[layer_id] = copy_idx
                if isinstance(m, nn.Conv3d):
                    w = m0.weight.data[copy_idx.tolist(),:,:,:,:].clone()
                    m.weight.data[copy_idx.tolist(),:,:,:,:] = w.clone()
                else:
                    w = m0.weight.data[:,copy_idx.tolist(),:,:,:].clone()
                    m.weight.data[:,copy_idx.tolist(),:,:,:,:] = w.clone()
                cfg_mask[layer_id] = cfg_mask[pair_layers_3d[layer_id]].clone()
                layer_id += 1
                idx += 1 
                continue

            # number of channels
            num_keep = int(out_channels*(1-layer_ratio_down[idx]))
            num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep

            # pruning criterion
            if layer_id in pair_layers_3d:
                rank = np.argsort(CSS(m, layer_ratio_down[idx], pair_layer=all_conv3d[pair_layers_3d[layer_id]]))
            else:
                rank = np.argsort(CSS(m, layer_ratio_down[idx])) # (out_channels,)
            selected = rank[::-1][:num_keep]
            freedom = rank[::-1][num_keep:]

            # restore MRU
            copy_idx = np.where(L1_norm(m) == 0)[0]
            copy_indexes[layer_id] = copy_idx
            if isinstance(m, nn.Conv3d):
                w = m0.weight.data[copy_idx.tolist(),:,:,:,:].clone()
                m.weight.data[copy_idx.tolist(),:,:,:,:] = w.clone()
            else:
                w = m0.weight.data[:,copy_idx.tolist(),:,:,:].clone()
                m.weight.data[:,copy_idx.tolist(),:,:,:] = w.clone()

            # importance sampling
            weight_copy = m.weight.data.detach().cpu()
            if isinstance(m, nn.ConvTranspose3d):
                weight_copy = torch.transpose(weight_copy,0,1)
                # weight_copy = weight_copy.contiguous()
            if layer_id in pair_layers_3d:
                pair_m = all_conv3d[pair_layers_3d[layer_id]] 
                pair_weight_copy = pair_m.weight.data.detach().cpu()
                if isinstance(pair_m, nn.ConvTranspose3d):
                    pair_weight_copy = torch.transpose(pair_weight_copy,0,1)
                    # pair_weight_copy = pair_weight_copy.contiguous()
                    # pair_weight_copy = pair_weight_copy.view(pair_weight_copy.shape[0], -1)
                    pair_weight_copy = torch.reshape(pair_weight_copy, (pair_weight_copy.shape[0],-1))
            # weight_copy = weight_copy.view(weight_copy.shape[0], -1)
            weight_copy = torch.reshape(weight_copy, (weight_copy.shape[0], -1))
            if layer_id in pair_layers_3d:
                weight_copy = torch.cat((weight_copy, pair_weight_copy),dim=1)
            weight_copy = torch.transpose(weight_copy, 0, 1)
            base_weight = weight_copy[:, selected.tolist()]
            proj = projection_formula(base_weight)
            candidate = weight_copy[:, freedom.tolist()]
            candidate_prime = torch.matmul(proj, candidate)
            sampling_prob = F.softmax(torch.norm(candidate - candidate_prime, dim=0), dim=0)
            if num_free <= 0:
                grow = np.random.permutation(freedom)[:num_free]
            else: 
                grow = freedom[np.unique(torch.multinomial(sampling_prob, num_free).numpy())]

            # channel mask
            mask = torch.zeros(out_channels)
            mask[selected.tolist() + grow.tolist()] = 1
            cfg_mask[layer_id] = mask
            layer_id += 1
            idx += 1
            continue
        elif layer_id in l2_3d:
            if layer_id in pair_layers_3d and layer_id >= pair_layers_3d[layer_id]:
                prev_copy_idx = deepcopy(copy_indexes[prev_layers_3d[layer_id]])
                copy_idx = np.where(L1_norm(m) == 0)[0]
                copy_indexes[layer_id] = copy_idx
                if isinstance(m, nn.Conv3d):
                    w = m0.weight.data[:,prev_copy_idx.tolist(),:,:,:].clone()
                    m.weight.data[:,prev_copy_idx.tolist(),:,:,:] = w.clone()
                    w = m0.weight.data[copy_idx.tolist(),:,:,:,:].clone()
                    m.weight.data[copy_idx.tolist(),:,:,:,:] = w.clone()
                else:
                    w = m0.weight.data[prev_copy_idx.tolist(),:,:,:,:].clone()
                    m.weight.data[prev_copy_idx.tolist(),:,:,:,:] = w.clone()
                    w = m0.weight.data[:,copy_idx.tolist(),:,:,:].clone()
                    m.weight.data[:,copy_idx.tolist(),:,:,:] = w.clone()
                cfg_mask[layer_id] = cfg_mask[pair_layers_3d[layer_id]].clone()
                layer_id += 1
                idx += 1 
                continue

            # number of channels
            num_keep = int(out_channels*(1-layer_ratio_down[idx]))
            num_free = int(out_channels*(1-layer_ratio_up[idx])) - num_keep

            # pruning criterion
            if layer_id in pair_layers_3d:
                rank = np.argsort(CSS(m, layer_ratio_down[idx], pair_layer=all_conv3d[pair_layers_3d[layer_id]]))
            else:
                rank = np.argsort(CSS(m, layer_ratio_down[idx])) # (out_channels,)
            selected = rank[::-1][:num_keep]
            freedom = rank[::-1][num_keep:]

            # restore MRU
            prev_copy_idx = deepcopy(copy_indexes[prev_layers_3d[layer_id]])
            copy_idx = np.where(L1_norm(m) == 0)[0]
            copy_indexes[layer_id] = copy_idx
            if isinstance(m, nn.Conv3d):
                w = m0.weight.data[:,prev_copy_idx.tolist(),:,:,:].clone()
                m.weight.data[:,prev_copy_idx.tolist(),:,:,:] = w.clone()
                w = m0.weight.data[copy_idx.tolist(),:,:,:,:].clone()
                m.weight.data[copy_idx.tolist(),:,:,:,:] = w.clone()
            else:
                w = m0.weight.data[prev_copy_idx.tolist(),:,:,:,:].clone()
                m.weight.data[prev_copy_idx.tolist(),:,:,:,:] = w.clone()
                w = m0.weight.data[:,copy_idx.tolist(),:,:,:].clone()
                m.weight.data[:,copy_idx.tolist(),:,:,:] = w.clone()

            # importance sampling
            weight_copy = m.weight.data.detach().cpu()
            if isinstance(m, nn.ConvTranspose3d):
                weight_copy = torch.transpose(weight_copy,0,1)
            if layer_id in pair_layers_3d:
                pair_m = all_conv3d[pair_layers_3d[layer_id]] 
                pair_weight_copy = pair_m.weight.data.detach().cpu()
                if isinstance(pair_m, nn.ConvTranspose3d):
                    pair_weight_copy = torch.transpose(pair_weight_copy,0,1)
                    # pair_weight_copy = pair_weight_copy.view(pair_weight_copy.shape[0], -1)
                    pair_weight_copy = torch.reshape(pair_weight_copy, (pair_weight_copy.shape[0],-1))
            # weight_copy = weight_copy.view(weight_copy.shape[0], -1)
            weight_copy = torch.reshape(weight_copy, (weight_copy.shape[0],-1))
            if layer_id in pair_layers_3d:
                weight_copy = torch.cat((weight_copy, pair_weight_copy),dim=1)
            weight_copy = torch.transpose(weight_copy, 0, 1)
            base_weight = weight_copy[:, selected.tolist()]
            proj = projection_formula(base_weight)
            candidate = weight_copy[:, freedom.tolist()]
            candidate_prime = torch.matmul(proj, candidate)
            sampling_prob = F.softmax(torch.norm(candidate - candidate_prime, dim=0), dim=0)
            if num_free <= 0:
                grow = np.random.permutation(freedom)[:num_free]
            else: 
                grow = freedom[np.unique(torch.multinomial(sampling_prob, num_free).numpy())]

            # channel mask
            mask = torch.zeros(out_channels)
            mask[selected.tolist() + grow.tolist()] = 1
            cfg_mask[layer_id] = mask
            layer_id += 1
            idx += 1
            continue
        elif layer_id in l3_3d:
            # restore MRU
            prev_copy_idx = deepcopy(copy_indexes[prev_layers_3d[layer_id]])
            if isinstance(m, nn.Conv3d):
                w = m0.weight.data[:,prev_copy_idx.tolist(),:,:,:].clone()
                m.weight.data[:,prev_copy_idx.tolist(),:,:,:] = w.clone()
            else:
                w = m0.weight.data[prev_copy_idx.tolist(),:,:,:,:].clone()
                m.weight.data[prev_copy_idx.tolist(),:,:,:,:] = w.clone()
            layer_id += 1
            continue
        elif layer_id in skip:
            layer_id += 1
            pass
        layer_id += 1
        
    layer_id = 0
    for [m,m0] in zip(all_bn3d, prev_all_bn3d):
        if layer_id in l1_3d + l2_3d:
            copy_idx = copy_indexes[layer_id]
            w = m0.weight.data[copy_idx.tolist()].clone()
            m.weight.data[copy_idx.tolist()] = w.clone()
            b = m0.bias.data[copy_idx.tolist()].clone()
            m.bias.data[copy_idx.tolist()] = b.clone()
            rm = m0.running_mean[copy_idx.tolist()].clone()
            m.running_mean[copy_idx.tolist()] = rm.clone()
            rv = m0.running_var[copy_idx.tolist()].clone()
            m.running_var[copy_idx.tolist()] = rv.clone()
        layer_id += 1

    prev_model = deepcopy(model)
    return cfg_mask, prev_model




