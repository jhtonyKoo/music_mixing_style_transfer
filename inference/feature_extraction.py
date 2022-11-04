"""
    Inference code of extracting embeddings from music recordings using FXencoder
    of the work "Music Mixing Style Transfer: A Contrastive Learning Approach to Disentangle Audio Effects"

    Process : extracts FX embeddings of each song inside the target directory.
"""
from glob import glob
import os
import librosa
import numpy as np
import torch

import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(os.path.dirname(currentdir), "mixing_style_transfer"))
from networks import FXencoder
from data_loader import *


class FXencoder_Inference:
    def __init__(self, args, trained_w_ddp=True):
        if args.inference_device!='cpu' and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        # inference computational hyperparameters
        self.segment_length = args.segment_length
        self.batch_size = args.batch_size
        self.sample_rate = 44100    # sampling rate should be 44100
        self.time_in_seconds = int(args.segment_length // self.sample_rate)

        # directory configuration
        self.output_dir = args.target_dir if args.output_dir==None else args.output_dir
        self.target_dir = args.target_dir

        # load model and its checkpoint weights
        self.models = {}
        self.models['effects_encoder'] = FXencoder(args.cfg_encoder).to(self.device)
        ckpt_paths = {'effects_encoder' : args.ckpt_path_enc}
        # reload saved model weights
        ddp = trained_w_ddp
        self.reload_weights(ckpt_paths, ddp=ddp)

        # save current arguments
        self.save_args(args)


    # reload model weights from the target checkpoint path
    def reload_weights(self, ckpt_paths, ddp=True):
        for cur_model_name in self.models.keys():
            checkpoint = torch.load(ckpt_paths[cur_model_name], map_location=self.device)

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint["model"].items():
                # remove `module.` if the model was trained with DDP
                name = k[7:] if ddp else k
                new_state_dict[name] = v
        
            # load params
            self.models[cur_model_name].load_state_dict(new_state_dict)

            print(f"---reloaded checkpoint weights : {cur_model_name} ---")


    # save averaged embedding from whole songs
    def save_averaged_embeddings(self, ):
        # embedding output directory path
        emb_out_dir = f"{self.output_dir}"
        print(f'\n\n=====Inference seconds : {self.time_in_seconds}=====')

        # target_file_paths = glob(f"{self.target_dir}/**/*.wav", recursive=True)
        target_file_paths = glob(os.path.join(self.target_dir, '**', '*.wav'), recursive=True)
        for step, target_file_path in enumerate(target_file_paths):
            print(f"\nInference step : {step+1}/{len(target_file_paths)}")
            print(f"---current file path : {target_file_path}---")

            ''' load waveform signal '''
            target_song_whole = load_wav_segment(target_file_path, axis=0)
            # check if mono -> convert to stereo by duplicating mono signal
            if len(target_song_whole.shape)==1:
                target_song_whole = np.stack((target_song_whole, target_song_whole), axis=0)
            # check axis dimension
            # signal shape should be : [channel, signal duration]
            elif target_song_whole.shape[1]==2:
                target_song_whole = target_song_whole.transpose()
            target_song_whole = torch.from_numpy(target_song_whole).float()
            ''' segmentize whole songs into batch '''
            whole_batch_data = self.batchwise_segmentization(target_song_whole, target_file_path)

            ''' inference '''
            # infer whole song
            infered_data_list = []
            infered_c_list = []
            infered_z_list = []
            for cur_idx, cur_data in enumerate(whole_batch_data):
                cur_data = cur_data.to(self.device)
            
                with torch.no_grad():
                    self.models["effects_encoder"].eval()
                    # FXencoder
                    out_c_emb = self.models["effects_encoder"](cur_data)
                infered_c_list.append(out_c_emb.cpu().detach())
            avg_c_feat = torch.mean(torch.cat(infered_c_list, dim=0), dim=0).squeeze().cpu().detach().numpy()

            # save outputs
            cur_output_path = target_file_path.replace(self.target_dir, self.output_dir).replace('.wav', '_fx_embedding.npy')
            os.makedirs(os.path.dirname(cur_output_path), exist_ok=True)
            np.save(cur_output_path, avg_c_feat)


    # function that segmentize an entire song into batch
    def batchwise_segmentization(self, target_song, target_file_path, discard_last=False):
        assert target_song.shape[-1] >= self.segment_length, \
                f"Error : Insufficient duration!\n\t \
                Target song's length is shorter than segment length.\n\t \
                Song name : {target_file_path}\n\t \
                Consider changing the 'segment_length' or song with sufficient duration"

        # discard restovers (last segment)
        if discard_last:
            target_length = target_song.shape[-1] - target_song.shape[-1] % self.segment_length
            target_song = target_song[:, :target_length]
        # pad last segment
        else:
            pad_length = self.segment_length - target_song.shape[-1] % self.segment_length
            target_song = torch.cat((target_song, torch.zeros(2, pad_length)), axis=-1)

        whole_batch_data = []
        batch_wise_data = []
        for cur_segment_idx in range(target_song.shape[-1]//self.segment_length):
            batch_wise_data.append(target_song[..., cur_segment_idx*self.segment_length:(cur_segment_idx+1)*self.segment_length])
            if len(batch_wise_data)==self.batch_size:
                whole_batch_data.append(torch.stack(batch_wise_data, dim=0))
                batch_wise_data = []
        if batch_wise_data:
            whole_batch_data.append(torch.stack(batch_wise_data, dim=0))

        return whole_batch_data


    # save current inference arguments
    def save_args(self, params):
        info = '\n[args]\n'
        for sub_args in parser._action_groups:
            if sub_args.title in ['positional arguments', 'optional arguments', 'options']:
                continue
            size_sub = len(sub_args._group_actions)
            info += f'  {sub_args.title} ({size_sub})\n'
            for i, arg in enumerate(sub_args._group_actions):
                prefix = '-'
                info += f'      {prefix} {arg.dest:20s}: {getattr(params, arg.dest)}\n'
        info += '\n'

        os.makedirs(self.output_dir, exist_ok=True)
        record_path = f"{self.output_dir}feature_extraction_inference_configurations.txt"
        f = open(record_path, 'w')
        np.savetxt(f, [info], delimiter=" ", fmt="%s")
        f.close()



if __name__ == '__main__':
    ''' Configurations for inferencing music effects encoder '''
    currentdir = os.path.dirname(os.path.realpath(__file__))
    default_ckpt_path = os.path.join(os.path.dirname(currentdir), 'weights', 'FXencoder_ps.pt')

    import argparse
    import yaml
    parser = argparse.ArgumentParser()

    directory_args = parser.add_argument_group('Directory args')
    directory_args.add_argument('--target_dir', type=str, default='./samples/')
    directory_args.add_argument('--output_dir', type=str, default=None, help='if no output_dir is specified (None), the results will be saved inside the target_dir')
    directory_args.add_argument('--ckpt_path_enc', type=str, default=default_ckpt_path)

    inference_args = parser.add_argument_group('Inference args')
    inference_args.add_argument('--segment_length', type=int, default=44100*10) # segmentize input according to this duration
    inference_args.add_argument('--batch_size', type=int, default=1)            # for processing long audio
    inference_args.add_argument('--inference_device', type=str, default='cpu', help="if this option is not set to 'cpu', inference will happen on gpu only if there is a detected one")

    args = parser.parse_args()

    # load network configurations
    with open(os.path.join(currentdir, 'configs.yaml'), 'r') as f:
        configs = yaml.full_load(f)
    args.cfg_encoder = configs['Effects_Encoder']['default']

    # Extract features using pre-trained FXencoder
    inference_encoder = FXencoder_Inference(args)
    inference_encoder.save_averaged_embeddings()
    
    