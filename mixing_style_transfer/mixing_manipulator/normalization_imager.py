"""
    Implementation of the normalization process of stereo-imaging and panning effects
"""
import numpy as np
import sys
import os

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from common_audioeffects import AugmentationChain, Haas


'''
    ### normalization algorithm for stereo imaging and panning effects ###
    process :
        1. inputs 2-channeled audio
        2. apply Haas effects if the input audio is almost mono
        3. normalize mid-side channels according to target precomputed feature value
        4. normalize left-right channels 50-50
        5. normalize mid-side channels again
'''
def normalize_imager(data, \
                        target_side_mid_bal=0.9, \
                        mono_threshold=0.95, \
                        sr=44100, \
                        eps=1e-04, \
                        verbose=False):

    # to mid-side channels
    mid, side = lr_to_ms(data[:,0], data[:,1])

    if verbose:
        print_balance(data[:,0], data[:,1])
        print_balance(mid, side)
        print()

    # apply mid-side weights according to energy
    mid_e, side_e = np.sum(mid**2), np.sum(side**2)
    total_e = mid_e + side_e
    # apply haas effect to almost-mono signal
    if mid_e/total_e > mono_threshold:
        aug_chain = AugmentationChain(fxs=[(Haas(sample_rate=sr), 1, True)])
        data = aug_chain([data])[0]
        mid, side = lr_to_ms(data[:,0], data[:,1])

    if verbose:
        print_balance(data[:,0], data[:,1])
        print_balance(mid, side)
        print()

    # normalize mid-side channels (stereo imaging)
    new_mid, new_side = process_balance(mid, side, tgt_e1_bal=target_side_mid_bal, eps=eps)
    left, right = ms_to_lr(new_mid, new_side)
    imaged = np.stack([left, right], 1)

    if verbose:
        print_balance(new_mid, new_side)
        print_balance(left, right)
        print()

    # normalize panning to have the balance of left-right channels 50-50
    left, right = process_balance(left, right, tgt_e1_bal=0.5, eps=eps)
    mid, side = lr_to_ms(left, right)

    if verbose:
        print_balance(mid, side)
        print_balance(left, right)
        print()

    # normalize again mid-side channels (stereo imaging)
    new_mid, new_side = process_balance(mid, side, tgt_e1_bal=target_side_mid_bal, eps=eps)
    left, right = ms_to_lr(new_mid, new_side)
    imaged = np.stack([left, right], 1)

    if verbose:
        print_balance(new_mid, new_side)
        print_balance(left, right)
        print()

    return imaged


# balance out 2 input data's energy according to given balance
# tgt_e1_bal range = [0.0, 1.0]
    # tgt_e2_bal = 1.0 - tgt_e1_bal_range
def process_balance(data_1, data_2, tgt_e1_bal=0.5, eps=1e-04):

    e_1, e_2 = np.sum(data_1**2), np.sum(data_2**2)
    total_e = e_1 + e_2

    tgt_1_gain = np.sqrt(tgt_e1_bal * total_e / (e_1 + eps))

    new_data_1 = data_1 * tgt_1_gain
    new_e_1 = e_1 * (tgt_1_gain ** 2)
    left_e_1 = total_e - new_e_1
    tgt_2_gain = np.sqrt(left_e_1 / (e_2 + 1e-3))
    new_data_2 = data_2 * tgt_2_gain

    return new_data_1, new_data_2


# left-right channeled signal to mid-side signal
def lr_to_ms(left, right):
    mid = left + right
    side = left - right
    return mid, side


# mid-side channeled signal to left-right signal
def ms_to_lr(mid, side):
    left = (mid + side) / 2
    right = (mid - side) / 2
    return left, right


# print energy balance of 2 inputs
def print_balance(data_1, data_2):
    e_1, e_2 = np.sum(data_1**2), np.sum(data_2**2)
    total_e = e_1 + e_2
    print(total_e, e_1/total_e, e_2/total_e)

