"""
    Data Loaders for 
        1. contrastive learning of audio effects 
        2. music mixing style transfer
    introduced in "Music Mixing Style Transfer: A Contrastive Learning Approach to Disentangle Audio Effects"
"""
import numpy as np
import wave
import soundfile as sf
import time
import random
from glob import glob

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
sys.path.append(os.path.dirname(os.path.dirname(currentdir)))
from loader_utils import *
from mixing_manipulator import *



'''
    Collate Functions
'''
class Collate_Variable_Length_Segments:
    def __init__(self, args):
        self.segment_length = args.segment_length
        self.random_length = args.reference_length
        self.num_strong_negatives = args.num_strong_negatives
        if 'musdb' in args.using_dataset.lower():
            self.instruments = ["drums", "bass", "other", "vocals"]
        else:
            raise NotImplementedError


    # collate function to trim segments A and B to random duration
        # this function can handle different number of 'strong negative' inputs
    def random_duration_segments_strong_negatives(self, batch):
        num_inst = len(self.instruments)
        # randomize current input length
        max_length = batch[0][0].shape[-1]
        min_length = max_length//2
        input_length_a, input_length_b = torch.randint(low=min_length, high=max_length, size=(2,))
        
        output_dict_A = {}
        output_dict_B = {}
        for cur_inst in self.instruments:
            output_dict_A[cur_inst] = []
            output_dict_B[cur_inst] = []
        for cur_item in batch:
            # set starting points
            start_point_a = torch.randint(low=0, high=max_length-input_length_a, size=(1,))[0]
            start_point_b = torch.randint(low=0, high=max_length-input_length_b, size=(1,))[0]
            # append to output dictionary
            for cur_i, cur_inst in enumerate(self.instruments):
                # append A# and B# with its strong negative samples
                for cur_neg_idx in range(self.num_strong_negatives+1):
                    output_dict_A[cur_inst].append(cur_item[cur_i*(self.num_strong_negatives+1)*2+2*cur_neg_idx][:, start_point_a : start_point_a+input_length_a])
                    output_dict_B[cur_inst].append(cur_item[cur_i*(self.num_strong_negatives+1)*2+1+2*cur_neg_idx][:, start_point_b : start_point_b+input_length_b])
        
        '''
            Output format : 
                [drums_A, bass_A, other_A, vocals_A], 
                [drums_B, bass_B, other_B, vocals_B]
        '''
        return [torch.stack(cur_segments, dim=0) for cur_inst, cur_segments in output_dict_A.items()], \
                [torch.stack(cur_segments, dim=0) for cur_inst, cur_segments in output_dict_B.items()]


    # collate function for training mixing style transfer
    def style_transfer_collate(self, batch):
        output_dict_A1 = {}
        output_dict_A2 = {}
        output_dict_B2 = {}
        for cur_inst in self.instruments:
            output_dict_A1[cur_inst] = []
            output_dict_A2[cur_inst] = []
            output_dict_B2[cur_inst] = []
        for cur_item in batch:
            # append to output dictionary
            for cur_i, cur_inst in enumerate(self.instruments):
                output_dict_A1[cur_inst].append(cur_item[cur_i*3])
                output_dict_A2[cur_inst].append(cur_item[cur_i*3+1])
                output_dict_B2[cur_inst].append(cur_item[cur_i*3+2])

        '''
            Output format : 
                [drums_A1, bass_A1, other_A1, vocals_A1],
                [drums_A2, bass_A2, other_A2, vocals_A2],
                [drums_B2, bass_B2, other_B2, vocals_B2]
        '''
        return [torch.stack(cur_segments, dim=0) for cur_inst, cur_segments in output_dict_A1.items()], \
                [torch.stack(cur_segments, dim=0) for cur_inst, cur_segments in output_dict_A2.items()], \
                [torch.stack(cur_segments, dim=0) for cur_inst, cur_segments in output_dict_B2.items()]


'''
    Data Loaders
'''

# Data loader for training the 'FXencoder'
    # randomly loads two segments (A and B) from the dataset
    # both segments are manipulated via FXmanipulator using (1+number of strong negative samples) sets of parameters (resulting A1, A2, ..., A#, and B1, B2, ..., B#) (# = number of strong negative samples)
    # segments with the same effects applied (A1 and B1) are assigned as the positive pair during the training
    # segments with the same content but with different effects applied (A2, A3, ..., A3 for A1) are also formed in a batch as 'strong negative' samples
        # in the paper, we use strong negative samples = 1
class MUSDB_Dataset_Mixing_Manipulated_FXencoder(Dataset):
    def __init__(self, args, \
                    mode, \
                    applying_effects='full', \
                    apply_prob_dict=None):
        self.args = args
        self.data_dir = args.data_dir + mode + "/"
        self.mode = mode
        self.applying_effects = applying_effects
        self.normalization_order = args.normalization_order
        self.fixed_random_seed = args.random_seed
        self.pad_b4_manipulation = args.pad_b4_manipulation
        self.pad_length = 2048

        if 'musdb' in args.using_dataset.lower():
            self.instruments = ["drums", "bass", "other", "vocals"]
        else:
            raise NotImplementedError

        # path to contents
        self.data_paths = {}
        self.data_length_ratio_list = {}
        # load data paths for each instrument
        for cur_inst in self.instruments:
            self.data_paths[cur_inst] = glob(f'{self.data_dir}{cur_inst}_normalized_{self.normalization_order}_silence_trimmed*.wav') \
                                                if args.use_normalized else glob(f'{self.data_dir}{cur_inst}_silence_trimmed*.wav')
            self.data_length_ratio_list[cur_inst] = []
            # compute audio duration and its ratio
            for cur_file_path in self.data_paths[cur_inst]:
                cur_wav_length = load_wav_length(cur_file_path)
                cur_inst_length_ratio = cur_wav_length / get_total_audio_length(self.data_paths[cur_inst])
                self.data_length_ratio_list[cur_inst].append(cur_inst_length_ratio)

        # load effects chain
        if applying_effects=='full':
            if apply_prob_dict==None:
                # initial (default) applying probabilities of each FX
                apply_prob_dict = {'eq' : 0.9, \
                                    'comp' : 0.9, \
                                    'pan' : 0.3, \
                                    'imager' : 0.8, \
                                    'gain': 0.5}
                reverb_prob = {'drums' : 0.5, \
                                'bass' : 0.01, \
                                'vocals' : 0.9, \
                                'other' : 0.7}

            self.mixing_manipulator = {}
            for cur_inst in self.data_paths.keys():
                if 'reverb' in apply_prob_dict.keys():
                    if cur_inst=='drums':
                        cur_reverb_weight = 0.5
                    elif cur_inst=='bass':
                        cur_reverb_weight = 0.1
                    else:
                        cur_reverb_weight = 1.0
                    apply_prob_dict['reverb'] *= cur_reverb_weight
                else:
                    apply_prob_dict['reverb'] = reverb_prob[cur_inst]
                # create FXmanipulator for current instrument
                self.mixing_manipulator[cur_inst] = create_inst_effects_augmentation_chain_(cur_inst, \
                                                                                                apply_prob_dict=apply_prob_dict, \
                                                                                                ir_dir_path=args.ir_dir_path, \
                                                                                                sample_rate=args.sample_rate)
        # for single effects
        else:
            self.mixing_manipulator = {}
            if not isinstance(applying_effects, list):
                applying_effects = [applying_effects]
            for cur_inst in self.data_paths.keys():
                self.mixing_manipulator[cur_inst] = create_effects_augmentation_chain(applying_effects, \
                                                                                        ir_dir_path=args.ir_dir_path)


    def __len__(self):
        if self.mode=='train':
            return self.args.batch_size_total * 40
        else:
            return self.args.batch_size_total


    def __getitem__(self, idx):
        if self.mode=="train":
            torch.manual_seed(int(time.time())*(idx+1) % (2**32-1))
            np.random.seed(int(time.time())*(idx+1) % (2**32-1))
            random.seed(int(time.time())*(idx+1) % (2**32-1))
        else:
            # fixed random seed for evaluation
            torch.manual_seed(idx*self.fixed_random_seed)
            np.random.seed(idx*self.fixed_random_seed)
            random.seed(idx*self.fixed_random_seed)

        manipulated_segments = {}
        for cur_neg_idx in range(self.args.num_strong_negatives+1):
            manipulated_segments[cur_neg_idx] = {}

        # load already-saved data to save time for on-the-fly manipulation
        cur_data_dir_path = f"{self.data_dir}manipulated_encoder/{self.args.data_save_name}/{self.applying_effects}/{idx}/"
        if self.mode=="val" and os.path.exists(cur_data_dir_path):
            for cur_inst in self.instruments:
                for cur_neg_idx in range(self.args.num_strong_negatives+1):
                    cur_A_file_path = f"{cur_data_dir_path}{cur_inst}_A{cur_neg_idx+1}.wav"
                    cur_B_file_path = f"{cur_data_dir_path}{cur_inst}_B{cur_neg_idx+1}.wav"
                    cur_A = load_wav_segment(cur_A_file_path, axis=0, sample_rate=self.args.sample_rate)
                    cur_B = load_wav_segment(cur_B_file_path, axis=0, sample_rate=self.args.sample_rate)
                    manipulated_segments[cur_neg_idx][cur_inst] = [torch.from_numpy(cur_A).float(), torch.from_numpy(cur_B).float()]
        else:
            # repeat for number of instruments
            for cur_inst, cur_paths in self.data_paths.items():
                # choose file_path to be loaded
                cur_chosen_paths = np.random.choice(cur_paths, 2, p = self.data_length_ratio_list[cur_inst])
                # get random 2 starting points for each instrument
                last_point_A = load_wav_length(cur_chosen_paths[0])-self.args.segment_length_ref
                last_point_B = load_wav_length(cur_chosen_paths[1])-self.args.segment_length_ref
                # simply load more data to prevent artifacts likely to be caused by the manipulator
                if self.pad_b4_manipulation:
                    last_point_A -= self.pad_length*2
                    last_point_B -= self.pad_length*2
                cur_inst_start_point_A = torch.randint(low=0, \
                                                        high=last_point_A, \
                                                        size=(1,))[0]
                cur_inst_start_point_B = torch.randint(low=0, \
                                                        high=last_point_B, \
                                                        size=(1,))[0]
                # load wav segments from the selected starting points
                load_duration = self.args.segment_length_ref+self.pad_length*2 if self.pad_b4_manipulation else self.args.segment_length_ref
                cur_inst_segment_A = load_wav_segment(cur_chosen_paths[0], \
                                                        start_point=cur_inst_start_point_A, \
                                                        duration=load_duration, \
                                                        axis=1, \
                                                        sample_rate=self.args.sample_rate)
                cur_inst_segment_B = load_wav_segment(cur_chosen_paths[1], \
                                                        start_point=cur_inst_start_point_B, \
                                                        duration=load_duration, \
                                                        axis=1, \
                                                        sample_rate=self.args.sample_rate)
                # mixing manipulation
                # append A# and B# with its strong negative samples
                for cur_neg_idx in range(self.args.num_strong_negatives+1):
                    cur_manipulated_segment_A, cur_manipulated_segment_B = self.mixing_manipulator[cur_inst]([cur_inst_segment_A, cur_inst_segment_B])

                    # remove over-loaded area
                    if self.pad_b4_manipulation:
                        cur_manipulated_segment_A = cur_manipulated_segment_A[self.pad_length:-self.pad_length]
                        cur_manipulated_segment_B = cur_manipulated_segment_B[self.pad_length:-self.pad_length]
                    manipulated_segments[cur_neg_idx][cur_inst] = [torch.clamp(torch.transpose(torch.from_numpy(cur_manipulated_segment_A).float(), 1, 0), min=-1, max=1), \
                                                                    torch.clamp(torch.transpose(torch.from_numpy(cur_manipulated_segment_B).float(), 1, 0), min=-1, max=1)]

        # check manipulated data by saving them
        if self.mode=="val" and not os.path.exists(cur_data_dir_path):
            os.makedirs(cur_dir_path, exist_ok=True)
            for cur_inst in manipulated_segments[0].keys():
                for cur_manipulated_key, cur_manipualted_dict in manipulated_segments.items():
                    sf.write(f"{cur_dir_path}{cur_inst}_A{cur_manipulated_key+1}.wav", torch.transpose(cur_manipualted_dict[cur_inst][0], 1, 0), self.args.sample_rate, 'PCM_16')
                    sf.write(f"{cur_dir_path}{cur_inst}_B{cur_manipulated_key+1}.wav", torch.transpose(cur_manipualted_dict[cur_inst][1], 1, 0), self.args.sample_rate, 'PCM_16')

        output_list = []
        output_list_param = []
        for cur_inst in manipulated_segments[0].keys():
            for cur_manipulated_key, cur_manipualted_dict in manipulated_segments.items():
                output_list.extend(cur_manipualted_dict[cur_inst])

        '''
            Output format:
                list of effects manipulated stems of each instrument
                    drums_A1, drums_B1, drums_A2, drums_B2, drums_A3, drums_B3, ... ,
                    bass_A1, bass_B1, bass_A2, bass_B2, bass_A3, bass_B3, ... ,
                    other_A1, other_B1, other_A2, other_B2, other_A3, other_B3, ... ,
                    vocals_A1, vocals_B1, vocals_A2, vocals_B2, vocals_A3, vocals_B3, ...
                each stem has the shape of (number of channels, segment duration)
        '''
        return output_list


    # generate random manipulated results for evaluation
    def generate_contents_w_effects(self, num_content, num_effects, out_dir):
        print(f"start generating random effects of {self.applying_effects} applied contents")
        os.makedirs(out_dir, exist_ok=True)

        manipulated_segments = {}
        for cur_fx_idx in range(num_effects):
            manipulated_segments[cur_fx_idx] = {}
        # repeat for number of instruments
        for cur_inst, cur_paths in self.data_paths.items():
            # choose file_path to be loaded
            cur_path = np.random.choice(cur_paths, 1, p = self.data_length_ratio_list[cur_inst])[0]
            print(f"\tgenerating instrument : {cur_inst}")
            # get random 2 starting points for each instrument
            last_point = load_wav_length(cur_path)-self.args.segment_length_ref
            # simply load more data to prevent artifacts likely to be caused by the manipulator
            if self.pad_b4_manipulation:
                last_point -= self.pad_length*2
            cur_inst_start_points = torch.randint(low=0, \
                                                    high=last_point, \
                                                    size=(num_content,))
            # load wav segments from the selected starting points
            cur_inst_segments = []
            for cur_num_content in range(num_content):
                cur_ori_sample = load_wav_segment(cur_path, \
                                                        start_point=cur_inst_start_points[cur_num_content], \
                                                        duration=self.args.segment_length_ref, \
                                                        axis=1, \
                                                        sample_rate=self.args.sample_rate)
                cur_inst_segments.append(cur_ori_sample)

                sf.write(f"{out_dir}{cur_inst}_ori_{cur_num_content}.wav", cur_ori_sample, self.args.sample_rate, 'PCM_16')
            
            # mixing manipulation
            for cur_fx_idx in range(num_effects):
                cur_manipulated_segments = self.mixing_manipulator[cur_inst](cur_inst_segments)
                # remove over-loaded area
                if self.pad_b4_manipulation:
                    for cur_man_idx in range(len(cur_manipulated_segments)):
                        cur_segment_trimmed = cur_manipulated_segments[cur_man_idx][self.pad_length:-self.pad_length]
                        cur_manipulated_segments[cur_man_idx] = torch.clamp(torch.transpose(torch.from_numpy(cur_segment_trimmed).float(), 1, 0), min=-1, max=1)
                manipulated_segments[cur_fx_idx][cur_inst] = cur_manipulated_segments

        # write generated data
        # save each instruments
        for cur_inst in manipulated_segments[0].keys():
            for cur_manipulated_key, cur_manipualted_dict in manipulated_segments.items():
                for cur_content_idx in range(num_content):
                    sf.write(f"{out_dir}{cur_inst}_{chr(65+cur_content_idx//26)}{chr(65+cur_content_idx%26)}{cur_manipulated_key+1}.wav", torch.transpose(cur_manipualted_dict[cur_inst][cur_content_idx], 1, 0), self.args.sample_rate, 'PCM_16')
        # save mixture
        for cur_manipulated_key, cur_manipualted_dict in manipulated_segments.items():
            for cur_content_idx in range(num_content):
                for cur_idx, cur_inst in enumerate(manipulated_segments[0].keys()):
                    if cur_idx==0:
                        cur_mixture = cur_manipualted_dict[cur_inst][cur_content_idx]
                    else:
                        cur_mixture += cur_manipualted_dict[cur_inst][cur_content_idx]
                sf.write(f"{out_dir}mixture_{chr(65+cur_content_idx//26)}{chr(65+cur_content_idx%26)}{cur_manipulated_key+1}.wav", torch.transpose(cur_mixture, 1, 0), self.args.sample_rate, 'PCM_16')
        
        return



# Data loader for training the 'Mastering Style Converter'
    # loads two segments (A and B) from the dataset
    # both segments are manipulated via Mastering Effects Manipulator (resulting A1, A2, and B2)
        # one of the manipulated segment is used as a reference segment (B2), which is randomly manipulated the same as the ground truth segment (A2)
class MUSDB_Dataset_Mixing_Manipulated_Style_Transfer(Dataset):
    def __init__(self, args, \
                    mode, \
                    applying_effects='full', \
                    apply_prob_dict=None):
        self.args = args
        self.data_dir = args.data_dir + mode + "/"
        self.mode = mode
        self.applying_effects = applying_effects
        self.fixed_random_seed = args.random_seed
        self.pad_b4_manipulation = args.pad_b4_manipulation
        self.pad_length = 2048

        if 'musdb' in args.using_dataset.lower():
            self.instruments = ["drums", "bass", "other", "vocals"]
        else:
            raise NotImplementedError

        # load data paths for each instrument
        self.data_paths = {}
        self.data_length_ratio_list = {}
        for cur_inst in self.instruments:
            self.data_paths[cur_inst] = glob(f'{self.data_dir}{cur_inst}_normalized_{self.args.normalization_order}_silence_trimmed*.wav') \
                                            if args.use_normalized else glob(f'{self.data_dir}{cur_inst}_silence_trimmed.wav')
            self.data_length_ratio_list[cur_inst] = []
            # compute audio duration and its ratio
            for cur_file_path in self.data_paths[cur_inst]:
                cur_wav_length = load_wav_length(cur_file_path)
                cur_inst_length_ratio = cur_wav_length / get_total_audio_length(self.data_paths[cur_inst])
                self.data_length_ratio_list[cur_inst].append(cur_inst_length_ratio)
        
        self.mixing_manipulator = {}
        if applying_effects=='full':
            if apply_prob_dict==None:
                # initial (default) applying probabilities of each FX
                    # we don't update these probabilities for training the MixFXcloner
                apply_prob_dict = {'eq' : 0.9, \
                                    'comp' : 0.9, \
                                    'pan' : 0.3, \
                                    'imager' : 0.8, \
                                    'gain': 0.5}
                reverb_prob = {'drums' : 0.5, \
                                'bass' : 0.01, \
                                'vocals' : 0.9, \
                                'other' : 0.7}
            for cur_inst in self.data_paths.keys():
                if 'reverb' in apply_prob_dict.keys():
                    if cur_inst=='drums':
                        cur_reverb_weight = 0.5
                    elif cur_inst=='bass':
                        cur_reverb_weight = 0.1
                    else:
                        cur_reverb_weight = 1.0
                    apply_prob_dict['reverb'] *= cur_reverb_weight
                else:
                    apply_prob_dict['reverb'] = reverb_prob[cur_inst]
                self.mixing_manipulator[cur_inst] = create_inst_effects_augmentation_chain(cur_inst, \
                                                                                                apply_prob_dict=apply_prob_dict, \
                                                                                                ir_dir_path=args.ir_dir_path, \
                                                                                                sample_rate=args.sample_rate)
        # for single effects
        else:
            if not isinstance(applying_effects, list):
                applying_effects = [applying_effects]
            for cur_inst in self.data_paths.keys():
                self.mixing_manipulator[cur_inst] = create_effects_augmentation_chain(applying_effects, \
                                                                                        ir_dir_path=args.ir_dir_path)


    def __len__(self):
        min_length = get_total_audio_length(glob(f'{self.data_dir}vocals_normalized_{self.args.normalization_order}*.wav'))
        data_len = min_length // self.args.segment_length
        return data_len


    def __getitem__(self, idx):
        if self.mode=="train":
            torch.manual_seed(int(time.time())*(idx+1) % (2**32-1))
            np.random.seed(int(time.time())*(idx+1) % (2**32-1))
            random.seed(int(time.time())*(idx+1) % (2**32-1))
        else:
            # fixed random seed for evaluation
            torch.manual_seed(idx*self.fixed_random_seed)
            np.random.seed(idx*self.fixed_random_seed)
            random.seed(idx*self.fixed_random_seed)

        manipulated_segments = {}

        # load already-saved data to save time for on-the-fly manipulation
        cur_data_dir_path = f"{self.data_dir}manipulated_converter/{self.args.data_save_name}/{self.applying_effects}/{idx}/"
        if self.mode=="val" and os.path.exists(cur_data_dir_path):
            for cur_inst in self.instruments:
                cur_A1_file_path = f"{cur_data_dir_path}{cur_inst}_A1.wav"
                cur_A2_file_path = f"{cur_data_dir_path}{cur_inst}_A2.wav"
                cur_B2_file_path = f"{cur_data_dir_path}{cur_inst}_B2.wav"
                cur_manipulated_segment_A1 = load_wav_segment(cur_A1_file_path, axis=0, sample_rate=self.args.sample_rate)
                cur_manipulated_segment_A2 = load_wav_segment(cur_A2_file_path, axis=0, sample_rate=self.args.sample_rate)
                cur_manipulated_segment_B2 = load_wav_segment(cur_B2_file_path, axis=0, sample_rate=self.args.sample_rate)
                manipulated_segments[cur_inst] = [torch.from_numpy(cur_manipulated_segment_A1).float(), \
                                                    torch.from_numpy(cur_manipulated_segment_A2).float(), \
                                                    torch.from_numpy(cur_manipulated_segment_B2).float()]
        else:
            # repeat for number of instruments
            for cur_inst, cur_paths in self.data_paths.items():
                # choose file_path to be loaded
                cur_chosen_paths = np.random.choice(cur_paths, 2, p = self.data_length_ratio_list[cur_inst])
                # cur_chosen_paths = [cur_paths[idx], cur_paths[idx+1]]
                # get random 2 starting points for each instrument
                last_point_A = load_wav_length(cur_chosen_paths[0])-self.args.segment_length_ref
                last_point_B = load_wav_length(cur_chosen_paths[1])-self.args.segment_length_ref
                # simply load more data to prevent artifacts likely to be caused by the manipulator
                if self.pad_b4_manipulation:
                    last_point_A -= self.pad_length*2
                    last_point_B -= self.pad_length*2
                cur_inst_start_point_A = torch.randint(low=0, \
                                                        high=last_point_A, \
                                                        size=(1,))[0]
                cur_inst_start_point_B = torch.randint(low=0, \
                                                        high=last_point_B, \
                                                        size=(1,))[0]
                # load wav segments from the selected starting points
                load_duration = self.args.segment_length_ref+self.pad_length*2 if self.pad_b4_manipulation else self.args.segment_length_ref
                cur_inst_segment_A = load_wav_segment(cur_chosen_paths[0], \
                                                        start_point=cur_inst_start_point_A, \
                                                        duration=load_duration, \
                                                        axis=1, \
                                                        sample_rate=self.args.sample_rate)
                cur_inst_segment_B = load_wav_segment(cur_chosen_paths[1], \
                                                        start_point=cur_inst_start_point_B, \
                                                        duration=load_duration, \
                                                        axis=1, \
                                                        sample_rate=self.args.sample_rate)
                ''' mixing manipulation '''
                # manipulate segment A and B to produce
                    # input : A1 (normalized sample)
                    # ground truth : A2
                    # reference : B2
                cur_manipulated_segment_A1 = cur_inst_segment_A
                cur_manipulated_segment_A2, cur_manipulated_segment_B2 = self.mixing_manipulator[cur_inst]([cur_inst_segment_A, cur_inst_segment_B])
                # remove over-loaded area
                if self.pad_b4_manipulation:
                    cur_manipulated_segment_A1 = cur_manipulated_segment_A1[self.pad_length:-self.pad_length]
                    cur_manipulated_segment_A2 = cur_manipulated_segment_A2[self.pad_length:-self.pad_length]
                    cur_manipulated_segment_B2 = cur_manipulated_segment_B2[self.pad_length:-self.pad_length]
                manipulated_segments[cur_inst] = [torch.clamp(torch.transpose(torch.from_numpy(cur_manipulated_segment_A1).float(), 1, 0), min=-1, max=1), \
                                                    torch.clamp(torch.transpose(torch.from_numpy(cur_manipulated_segment_A2).float(), 1, 0), min=-1, max=1), \
                                                    torch.clamp(torch.transpose(torch.from_numpy(cur_manipulated_segment_B2).float(), 1, 0), min=-1, max=1)]
        
        # check manipulated data by saving them
        if (self.mode=="val" and not os.path.exists(cur_data_dir_path)):
            mixture_dict = {}
            for cur_inst in manipulated_segments.keys():
                cur_inst_dir_path = f"{cur_data_dir_path}{idx}/{cur_inst}/"
                os.makedirs(cur_inst_dir_path, exist_ok=True)
                sf.write(f"{cur_inst_dir_path}A1.wav", torch.transpose(manipulated_segments[cur_inst][0], 1, 0), self.args.sample_rate, 'PCM_16')
                sf.write(f"{cur_inst_dir_path}A2.wav", torch.transpose(manipulated_segments[cur_inst][1], 1, 0), self.args.sample_rate, 'PCM_16')
                sf.write(f"{cur_inst_dir_path}B2.wav", torch.transpose(manipulated_segments[cur_inst][2], 1, 0), self.args.sample_rate, 'PCM_16')
                mixture_dict['A1'] = torch.transpose(manipulated_segments[cur_inst][0], 1, 0)
                mixture_dict['A2'] = torch.transpose(manipulated_segments[cur_inst][1], 1, 0)
                mixture_dict['B2'] = torch.transpose(manipulated_segments[cur_inst][2], 1, 0)
            cur_mix_dir_path = f"{cur_data_dir_path}{idx}/mixture/"
            os.makedirs(cur_mix_dir_path, exist_ok=True)
            sf.write(f"{cur_mix_dir_path}A1.wav", mixture_dict['A1'], self.args.sample_rate, 'PCM_16')
            sf.write(f"{cur_mix_dir_path}A2.wav", mixture_dict['A2'], self.args.sample_rate, 'PCM_16')
            sf.write(f"{cur_mix_dir_path}B2.wav", mixture_dict['B2'], self.args.sample_rate, 'PCM_16')

        output_list = []
        for cur_inst in manipulated_segments.keys():
            output_list.extend(manipulated_segments[cur_inst])

        '''
            Output format:
                list of effects manipulated stems of each instrument
                    drums_A1, drums_A2, drums_B2,
                    bass_A1, bass_A2, bass_B2,
                    other_A1, other_A2, other_B2,
                    vocals_A1, vocals_A2, vocals_B2,
                each stem has the shape of (number of channels, segment duration)
            Notation :
                A1 = input of the network
                A2 = ground truth
                B2 = reference track
        '''
        return output_list



# Data loader for inferencing the task 'Mixing Style Transfer'
### loads whole mixture or stems from the target directory
class Song_Dataset_Inference(Dataset):
    def __init__(self, args):
        self.args = args
        self.data_dir = args.target_dir
        self.interpolate = args.interpolation
        
        self.instruments = args.instruments

        self.data_dir_paths = sorted(glob(f"{self.data_dir}*/"))

        self.input_name = args.input_file_name
        self.reference_name = args.reference_file_name
        self.stem_level_directory_name = args.stem_level_directory_name \
                if self.args.do_not_separate else os.path.join(args.stem_level_directory_name, args.separation_model)
        if self.interpolate:
            self.reference_name_B = args.reference_file_name_2interpolate

        # audio effects normalizer
        if args.normalize_input:
            self.normalization_chain = Audio_Effects_Normalizer(precomputed_feature_path=args.precomputed_normalization_feature, \
                                                                                    STEMS=args.instruments, \
                                                                                    EFFECTS=args.normalization_order)


    def __len__(self):
        return len(self.data_dir_paths)


    def __getitem__(self, idx):
        ''' stem-level conversion '''
        input_stems = []
        reference_stems = []
        reference_B_stems = []
        for cur_inst in self.instruments:
            cur_input_file_path = os.path.join(self.data_dir_paths[idx], self.stem_level_directory_name, self.input_name, cur_inst+'.wav')
            cur_reference_file_path = os.path.join(self.data_dir_paths[idx], self.stem_level_directory_name, self.reference_name, cur_inst+'.wav')

            # load wav
            cur_input_wav = load_wav_segment(cur_input_file_path, axis=0, sample_rate=self.args.sample_rate)
            cur_reference_wav = load_wav_segment(cur_reference_file_path, axis=0, sample_rate=self.args.sample_rate)

            if self.args.normalize_input:
                cur_input_wav = self.normalization_chain.normalize_audio(cur_input_wav.transpose(), src=cur_inst).transpose()

            input_stems.append(torch.clamp(torch.from_numpy(cur_input_wav).float(), min=-1, max=1))
            reference_stems.append(torch.clamp(torch.from_numpy(cur_reference_wav).float(), min=-1, max=1))

            # for interpolation
            if self.interpolate:
                cur_reference_B_file_path = os.path.join(self.data_dir_paths[idx], self.stem_level_directory_name, self.reference_name_B, cur_inst+'.wav')
                cur_reference_B_wav = load_wav_segment(cur_reference_B_file_path, axis=0, sample_rate=self.args.sample_rate)
                reference_B_stems.append(torch.clamp(torch.from_numpy(cur_reference_B_wav).float(), min=-1, max=1))

        dir_name = os.path.dirname(self.data_dir_paths[idx])

        if self.interpolate:
            return torch.stack(input_stems, dim=0), torch.stack(reference_stems, dim=0), torch.stack(reference_B_stems, dim=0), dir_name
        else:
            return torch.stack(input_stems, dim=0), torch.stack(reference_stems, dim=0), dir_name



# check dataset
if __name__ == '__main__':
    """
    Test code of data loaders
    """
    import time
    print('checking dataset...')

    total_epochs = 1
    bs = 5
    check_step_size = 3
    collate_class = Collate_Variable_Length_Segments(args)


    print('\n========== Effects Encoder ==========')
    from config import args
    ##### generate samples with ranfom configuration
    # args.normalization_order = 'eqcompimagegain'
    # for cur_effect in ['full', 'gain', 'comp', 'reverb', 'eq', 'imager', 'pan']:
    #     start_time = time.time()
    #     dataset = MUSDB_Dataset_Mixing_Manipulated_FXencoder(args, mode='val', applying_effects=cur_effect, check_data=True)
    #     dataset.generate_contents_w_effects(num_content=25, num_effects=10)
    #     print(f'\t---time taken : {time.time()-start_time}---')

    ### training data loder
    dataset = MUSDB_Dataset_Mixing_Manipulated_FXencoder(args, mode='train', applying_effects=['comp'])
    data_loader = DataLoader(dataset, \
                            batch_size=bs, \
                            shuffle=False, \
                            collate_fn=collate_class.random_duration_segments_strong_negatives, \
                            drop_last=False, \
                            num_workers=0)

    for epoch in range(total_epochs):
        start_time_loader = time.time()
        for step, output_list in enumerate(data_loader):
            if step==check_step_size:
                break
            print(f'Epoch {epoch+1}/{total_epochs}\tStep {step+1}/{len(data_loader)}')
            print(f'num contents : {len(output_list)}\tnum instruments : {len(output_list[0])}\tcontent A shape : {output_list[0][0].shape}\t content B shape : {output_list[1][0].shape} \ttime taken: {time.time()-start_time_loader:.4f}')
            start_time_loader = time.time()
    

    print('\n========== Mixing Style Transfer ==========')
    from trainer_mixing_transfer.config_conv import args
    ### training data loder
    dataset = MUSDB_Dataset_Mixing_Manipulated_Style_Transfer(args, mode='train')
    data_loader = DataLoader(dataset, \
                            batch_size=bs, \
                            shuffle=False, \
                            collate_fn=collate_class.style_transfer_collate, \
                            drop_last=False, \
                            num_workers=0)

    for epoch in range(total_epochs):
        start_time_loader = time.time()
        for step, output_list in enumerate(data_loader):
            if step==check_step_size:
                break
            print(f'Epoch {epoch+1}/{total_epochs}\tStep {step+1}/{len(data_loader)}')
            print(f'num contents : {len(output_list)}\tnum instruments : {len(output_list[0])}\tA1 shape : {output_list[0][0].shape}\tA2 shape : {output_list[1][0].shape}\tA3 shape : {output_list[2][0].shape}\ttime taken: {time.time()-start_time_loader:.4f}')
            start_time_loader = time.time()


    print('\n--- checking dataset completed ---')

