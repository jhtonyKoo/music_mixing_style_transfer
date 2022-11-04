"""
    Implementation of objective functions used in the task 'End-to-end Remastering System'
"""
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(currentdir))

from modules.training_utils import *
from modules.front_back_end import *



'''
    Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss
    below source code (class NT_Xent) is a replication from the github repository - https://github.com/Spijkervet/SimCLR
    the original implementation can be found here: https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py
'''
class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
            # mask[i, batch_size * world_size + i] = 0
            # mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        # combine embeddings from all GPUs
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss



# Root Mean Squared Loss
#   penalizes the volume factor with non-linearlity
class RMSLoss(nn.Module):
    def __init__(self, reduce, loss_type="l2"):
        super(RMSLoss, self).__init__()
        self.weight_factor = 100.
        if loss_type=="l2":
            self.loss = nn.MSELoss(reduce=None)


    def forward(self, est_targets, targets):
        est_targets = est_targets.reshape(est_targets.shape[0]*est_targets.shape[1], est_targets.shape[2])
        targets = targets.reshape(targets.shape[0]*targets.shape[1], targets.shape[2])
        normalized_est = torch.sqrt(torch.mean(est_targets**2, dim=-1))
        normalized_tgt = torch.sqrt(torch.mean(targets**2, dim=-1))

        weight = torch.clamp(torch.abs(normalized_tgt-normalized_est), min=1/self.weight_factor) * self.weight_factor

        return torch.mean(weight**1.5 * self.loss(normalized_est, normalized_tgt))



# Multi-Scale Spectral Loss proposed at the paper "DDSP: DIFFERENTIABLE DIGITAL SIGNAL PROCESSING" (https://arxiv.org/abs/2001.04643)
#   we extend this loss by applying it to mid/side channels
class MultiScale_Spectral_Loss_MidSide_DDSP(nn.Module):
    def __init__(self, mode='midside', \
                        reduce=True, \
                        n_filters=None, \
                        windows_size=None, \
                        hops_size=None, \
                        window="hann", \
                        eps=1e-7, \
                        device=torch.device("cpu")):
        super(MultiScale_Spectral_Loss_MidSide_DDSP, self).__init__()
        self.mode = mode
        self.eps = eps
        self.mid_weight = 0.5   # value in the range of 0.0 ~ 1.0
        self.logmag_weight = 0.1

        if n_filters is None:
            n_filters = [4096, 2048, 1024, 512]
            # n_filters = [4096]
        if windows_size is None:
            windows_size = [4096, 2048, 1024, 512]
            # windows_size = [4096]
        if hops_size is None:
            hops_size = [1024, 512, 256, 128]
            # hops_size = [1024]

        self.multiscales = []
        for i in range(len(windows_size)):
            cur_scale = {'window_size' : float(windows_size[i])}
            if self.mode=='midside':
                cur_scale['front_end'] = FrontEnd(channel='mono', \
                                                    n_fft=n_filters[i], \
                                                    hop_length=hops_size[i], \
                                                    win_length=windows_size[i], \
                                                    window=window, \
                                                    device=device)
            elif self.mode=='ori':
                cur_scale['front_end'] = FrontEnd(channel='stereo', \
                                                    n_fft=n_filters[i], \
                                                    hop_length=hops_size[i], \
                                                    win_length=windows_size[i], \
                                                    window=window, \
                                                    device=device)
            self.multiscales.append(cur_scale)

        self.objective_l1 = nn.L1Loss(reduce=reduce)
        self.objective_l2 = nn.MSELoss(reduce=reduce)


    def forward(self, est_targets, targets):
        if self.mode=='midside':
            return self.forward_midside(est_targets, targets)
        elif self.mode=='ori':
            return self.forward_ori(est_targets, targets)


    def forward_ori(self, est_targets, targets):
        total_loss = 0.0
        total_mag_loss = 0.0
        total_logmag_loss = 0.0
        for cur_scale in self.multiscales:
            est_mag = cur_scale['front_end'](est_targets, mode=["mag"])
            tgt_mag = cur_scale['front_end'](targets, mode=["mag"])

            mag_loss = self.magnitude_loss(est_mag, tgt_mag)
            logmag_loss = self.log_magnitude_loss(est_mag, tgt_mag)
            # cur_loss = mag_loss + logmag_loss
            # total_loss += cur_loss
            total_mag_loss += mag_loss
            total_logmag_loss += logmag_loss
        # return total_loss
        # print(f"ori - mag : {total_mag_loss}\tlog mag : {total_logmag_loss}")
        return (1-self.logmag_weight)*total_mag_loss + \
                (self.logmag_weight)*total_logmag_loss


    def forward_midside(self, est_targets, targets):
        est_mid, est_side = self.to_mid_side(est_targets)
        tgt_mid, tgt_side = self.to_mid_side(targets)
        total_loss = 0.0
        total_mag_loss = 0.0
        total_logmag_loss = 0.0
        for cur_scale in self.multiscales:
            est_mid_mag = cur_scale['front_end'](est_mid, mode=["mag"])
            est_side_mag = cur_scale['front_end'](est_side, mode=["mag"])
            tgt_mid_mag = cur_scale['front_end'](tgt_mid, mode=["mag"])
            tgt_side_mag = cur_scale['front_end'](tgt_side, mode=["mag"])

            mag_loss = self.mid_weight*self.magnitude_loss(est_mid_mag, tgt_mid_mag) + \
                        (1-self.mid_weight)*self.magnitude_loss(est_side_mag, tgt_side_mag)
            logmag_loss = self.mid_weight*self.log_magnitude_loss(est_mid_mag, tgt_mid_mag) + \
                        (1-self.mid_weight)*self.log_magnitude_loss(est_side_mag, tgt_side_mag)
            # cur_loss = mag_loss + logmag_loss
            # total_loss += cur_loss
            total_mag_loss += mag_loss
            total_logmag_loss += logmag_loss
        # return total_loss
        # print(f"midside - mag : {total_mag_loss}\tlog mag : {total_logmag_loss}")
        return (1-self.logmag_weight)*total_mag_loss + \
                (self.logmag_weight)*total_logmag_loss


    def to_mid_side(self, stereo_in):
        mid = stereo_in[:,0] + stereo_in[:,1]
        side = stereo_in[:,0] - stereo_in[:,1]
        return mid, side


    def magnitude_loss(self, est_mag_spec, tgt_mag_spec):
        return torch.norm(self.objective_l1(est_mag_spec, tgt_mag_spec))


    def log_magnitude_loss(self, est_mag_spec, tgt_mag_spec):
        est_log_mag_spec = torch.log10(est_mag_spec+self.eps)
        tgt_log_mag_spec = torch.log10(tgt_mag_spec+self.eps)
        return self.objective_l2(est_log_mag_spec, tgt_log_mag_spec)



# hinge loss for discriminator
def dis_hinge(dis_fake, dis_real):
    return torch.mean(torch.relu(1. - dis_real)) + torch.mean(torch.relu(1. + dis_fake))


# hinge loss for generator
def gen_hinge(dis_fake, dis_real=None):
    return -torch.mean(dis_fake)


# DirectCLR's implementation of infoNCE loss
def infoNCE(nn, p, temperature=0.1):
    nn = torch.nn.functional.normalize(nn, dim=1)
    p = torch.nn.functional.normalize(p, dim=1)
    nn = gather_from_all(nn)
    p = gather_from_all(p)
    logits = nn @ p.T
    logits /= temperature
    n = p.shape[0]
    labels = torch.arange(0, n, dtype=torch.long).cuda()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss




# Class of available loss functions
class Loss:
    def __init__(self, args, reduce=True):
        device = torch.device("cpu")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{args.gpu}")
        self.l1 = nn.L1Loss(reduce=reduce)
        self.mse = nn.MSELoss(reduce=reduce)
        self.ce = nn.CrossEntropyLoss()
        self.triplet = nn.TripletMarginLoss(margin=1., p=2)

        # self.ntxent = NT_Xent(args.train_batch*2, args.temperature, world_size=len(args.using_gpu.split(",")))
        self.ntxent = NT_Xent(args.batch_size_total*(args.num_strong_negatives+1), args.temperature, world_size=1)
        self.multi_scale_spectral_midside = MultiScale_Spectral_Loss_MidSide_DDSP(mode='midside', eps=args.eps, device=device)
        self.multi_scale_spectral_ori = MultiScale_Spectral_Loss_MidSide_DDSP(mode='ori', eps=args.eps, device=device)
        self.gain = RMSLoss(reduce=reduce)
        self.infonce = infoNCE

