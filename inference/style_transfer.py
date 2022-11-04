"""
    Inference code of music style transfer
    of the work "Music Mixing Style Transfer: A Contrastive Learning Approach to Disentangle Audio Effects"

    Process : converts the mixing style of the input music recording to that of the refernce music.
                files inside the target directory should be organized as follow 
                    "path_to_data_directory"/"song_name_#1"/input.wav
                    "path_to_data_directory"/"song_name_#1"/reference.wav
                    ...
                    "path_to_data_directory"/"song_name_#n"/input.wav
                    "path_to_data_directory"/"song_name_#n"/reference.wav
                where the 'input' and 'reference' should share the same names.
"""
import numpy as np
from glob import glob
import os
import torch

import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(os.path.dirname(currentdir), "mixing_style_transfer"))
from networks import FXencoder, TCNModel
from data_loader import *



class Mixing_Style_Transfer_Inference:
    def __init__(self, args, trained_w_ddp=True):
        if args.inference_device!='cpu' and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        
        # inference computational hyperparameters
        self.args = args
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
        self.models['mixing_converter'] = TCNModel(nparams=args.cfg_converter["condition_dimension"], \
                                                    ninputs=2, \
                                                    noutputs=2, \
                                                    nblocks=args.cfg_converter["nblocks"], \
                                                    dilation_growth=args.cfg_converter["dilation_growth"], \
                                                    kernel_size=args.cfg_converter["kernel_size"], \
                                                    channel_width=args.cfg_converter["channel_width"], \
                                                    stack_size=args.cfg_converter["stack_size"], \
                                                    cond_dim=args.cfg_converter["condition_dimension"], \
                                                    causal=args.cfg_converter["causal"]).to(self.device)
        
        ckpt_paths = {'effects_encoder' : args.ckpt_path_enc, \
                        'mixing_converter' : args.ckpt_path_conv}
        # reload saved model weights
        ddp = trained_w_ddp
        self.reload_weights(ckpt_paths, ddp=ddp)

        # load data loader for the inference procedure
        inference_dataset = Song_Dataset_Inference(args)
        self.data_loader = DataLoader(inference_dataset, \
                                        batch_size=1, \
                                        shuffle=False, \
                                        num_workers=args.workers, \
                                        drop_last=False)

        # save current arguments
        self.save_args(args)

        ''' check stem-wise result '''
        if not self.args.do_not_separate:
            os.environ['MKL_THREADING_LAYER'] = 'GNU'
            separate_file_names = [args.input_file_name, args.reference_file_name]
            if self.args.interpolation:
                separate_file_names.append(args.reference_file_name_2interpolate)
            for cur_idx, cur_inf_dir in enumerate(sorted(glob(f"{args.target_dir}*/"))):
                for cur_file_name in separate_file_names:
                    cur_sep_file_path = os.path.join(cur_inf_dir, cur_file_name+'.wav')
                    cur_sep_output_dir = os.path.join(cur_inf_dir, args.stem_level_directory_name)
                    if os.path.exists(os.path.join(cur_sep_output_dir, self.args.separation_model, cur_file_name, 'drums.wav')):
                        print(f'\talready separated current file : {cur_sep_file_path}')
                    else:
                        cur_cmd_line = f"demucs {cur_sep_file_path} -n {self.args.separation_model} -d {self.args.separation_device} -o {cur_sep_output_dir}"
                        os.system(cur_cmd_line)


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


    # Inference whole song
    def inference(self, ):
        print("\n======= Start to inference music mixing style transfer =======")
        # normalized input
        output_name_tag = 'output' if self.args.normalize_input else 'output_notnormed'

        for step, (input_stems, reference_stems, dir_name) in enumerate(self.data_loader):
            print(f"---inference file name : {dir_name[0]}---")
            cur_out_dir = dir_name[0].replace(self.target_dir, self.output_dir)
            os.makedirs(cur_out_dir, exist_ok=True)
            ''' stem-level inference '''
            inst_outputs = []
            for cur_inst_idx, cur_inst_name in enumerate(self.args.instruments):
                print(f'\t{cur_inst_name}...')
                ''' segmentize whole songs into batch '''
                if len(input_stems[0][cur_inst_idx][0]) > self.args.segment_length:
                    cur_inst_input_stem = self.batchwise_segmentization(input_stems[0][cur_inst_idx], \
                                                                                dir_name[0], \
                                                                                segment_length=self.args.segment_length, \
                                                                                discard_last=False)
                else:
                    cur_inst_input_stem = [input_stems[:, cur_inst_idx]]
                if len(reference_stems[0][cur_inst_idx][0]) > self.args.segment_length*2:
                    cur_inst_reference_stem = self.batchwise_segmentization(reference_stems[0][cur_inst_idx], \
                                                                                dir_name[0], \
                                                                                segment_length=self.args.segment_length_ref, \
                                                                                discard_last=False)
                else:
                    cur_inst_reference_stem = [reference_stems[:, cur_inst_idx]]

                ''' inference '''
                # first extract reference style embedding
                infered_ref_data_list = []
                for cur_ref_data in cur_inst_reference_stem:
                    cur_ref_data = cur_ref_data.to(self.device)
                    # Effects Encoder inference
                    with torch.no_grad():
                        self.models["effects_encoder"].eval()
                        reference_feature = self.models["effects_encoder"](cur_ref_data)
                    infered_ref_data_list.append(reference_feature)
                # compute average value from the extracted exbeddings
                infered_ref_data = torch.stack(infered_ref_data_list)
                infered_ref_data_avg = torch.mean(infered_ref_data.reshape(infered_ref_data.shape[0]*infered_ref_data.shape[1], infered_ref_data.shape[2]), axis=0)

                # mixing style converter
                infered_data_list = []
                for cur_data in cur_inst_input_stem:
                    cur_data = cur_data.to(self.device)
                    with torch.no_grad():
                        self.models["mixing_converter"].eval()
                        infered_data = self.models["mixing_converter"](cur_data, infered_ref_data_avg.unsqueeze(0))
                    infered_data_list.append(infered_data.cpu().detach())

                # combine back to whole song
                for cur_idx, cur_batch_infered_data in enumerate(infered_data_list):
                    cur_infered_data_sequential = torch.cat(torch.unbind(cur_batch_infered_data, dim=0), dim=-1)
                    fin_data_out = cur_infered_data_sequential if cur_idx==0 else torch.cat((fin_data_out, cur_infered_data_sequential), dim=-1)
                # final output of current instrument
                fin_data_out_inst = fin_data_out[:, :input_stems[0][cur_inst_idx].shape[-1]].numpy()

                inst_outputs.append(fin_data_out_inst)
                # save output of each instrument
                if self.args.save_each_inst:
                    sf.write(os.path.join(cur_out_dir, f"{cur_inst_name}_{output_name_tag}.wav"), fin_data_out_inst.transpose(-1, -2), self.args.sample_rate, 'PCM_16')
            # remix
            fin_data_out_mix = sum(inst_outputs)
            sf.write(os.path.join(cur_out_dir, f"mixture_{output_name_tag}.wav"), fin_data_out_mix.transpose(-1, -2), self.args.sample_rate, 'PCM_16')


    # Inference whole song
    def inference_interpolation(self, ):
        print("\n======= Start to inference interpolation examples =======")
        # normalized input
        output_name_tag = 'output_interpolation' if self.args.normalize_input else 'output_notnormed_interpolation'

        for step, (input_stems, reference_stems_A, reference_stems_B, dir_name) in enumerate(self.data_loader):
            print(f"---inference file name : {dir_name[0]}---")
            cur_out_dir = dir_name[0].replace(self.target_dir, self.output_dir)
            os.makedirs(cur_out_dir, exist_ok=True)
            ''' stem-level inference '''
            inst_outputs = []
            for cur_inst_idx, cur_inst_name in enumerate(self.args.instruments):
                print(f'\t{cur_inst_name}...')
                ''' segmentize whole song '''
                # segmentize input according to number of interpolating segments
                interpolate_segment_length = input_stems[0][cur_inst_idx].shape[1] // self.args.interpolate_segments + 1
                cur_inst_input_stem = self.batchwise_segmentization(input_stems[0][cur_inst_idx], \
                                                                                dir_name[0], \
                                                                                segment_length=interpolate_segment_length, \
                                                                                discard_last=False)
                # batchwise segmentize 2 reference tracks
                if len(reference_stems_A[0][cur_inst_idx][0]) > self.args.segment_length_ref:
                    cur_inst_reference_stem_A = self.batchwise_segmentization(reference_stems_A[0][cur_inst_idx], \
                                                                                        dir_name[0], \
                                                                                        segment_length=self.args.segment_length_ref, \
                                                                                        discard_last=False)
                else:
                    cur_inst_reference_stem_A = [reference_stems_A[:, cur_inst_idx]]
                if len(reference_stems_B[0][cur_inst_idx][0]) > self.args.segment_length_ref:
                    cur_inst_reference_stem_B = self.batchwise_segmentization(reference_stems_B[0][cur_inst_idx], \
                                                                                        dir_name[0], \
                                                                                        segment_length=self.args.segment_length, \
                                                                                        discard_last=False)
                else:
                    cur_inst_reference_stem_B = [reference_stems_B[:, cur_inst_idx]]

                ''' inference '''
                # first extract reference style embeddings
                # reference A
                infered_ref_data_list = []
                for cur_ref_data in cur_inst_reference_stem_A:
                    cur_ref_data = cur_ref_data.to(self.device)
                    # Effects Encoder inference
                    with torch.no_grad():
                        self.models["effects_encoder"].eval()
                        reference_feature = self.models["effects_encoder"](cur_ref_data)
                    infered_ref_data_list.append(reference_feature)
                # compute average value from the extracted exbeddings
                infered_ref_data = torch.stack(infered_ref_data_list)
                infered_ref_data_avg_A = torch.mean(infered_ref_data.reshape(infered_ref_data.shape[0]*infered_ref_data.shape[1], infered_ref_data.shape[2]), axis=0)

                # reference B
                infered_ref_data_list = []
                for cur_ref_data in cur_inst_reference_stem_B:
                    cur_ref_data = cur_ref_data.to(self.device)
                    # Effects Encoder inference
                    with torch.no_grad():
                        self.models["effects_encoder"].eval()
                        reference_feature = self.models["effects_encoder"](cur_ref_data)
                    infered_ref_data_list.append(reference_feature)
                # compute average value from the extracted exbeddings
                infered_ref_data = torch.stack(infered_ref_data_list)
                infered_ref_data_avg_B = torch.mean(infered_ref_data.reshape(infered_ref_data.shape[0]*infered_ref_data.shape[1], infered_ref_data.shape[2]), axis=0)

                # mixing style converter
                infered_data_list = []
                for cur_idx, cur_data in enumerate(cur_inst_input_stem):
                    cur_data = cur_data.to(self.device)
                    # perform linear interpolation on embedding space
                    cur_weight = (self.args.interpolate_segments-1-cur_idx) / (self.args.interpolate_segments-1)
                    cur_ref_emb = cur_weight * infered_ref_data_avg_A +  (1-cur_weight) * infered_ref_data_avg_B
                    with torch.no_grad():
                        self.models["mixing_converter"].eval()
                        infered_data = self.models["mixing_converter"](cur_data, cur_ref_emb.unsqueeze(0))
                    infered_data_list.append(infered_data.cpu().detach())

                # combine back to whole song
                for cur_idx, cur_batch_infered_data in enumerate(infered_data_list):
                    cur_infered_data_sequential = torch.cat(torch.unbind(cur_batch_infered_data, dim=0), dim=-1)
                    fin_data_out = cur_infered_data_sequential if cur_idx==0 else torch.cat((fin_data_out, cur_infered_data_sequential), dim=-1)
                # final output of current instrument
                fin_data_out_inst = fin_data_out[:, :input_stems[0][cur_inst_idx].shape[-1]].numpy()
                inst_outputs.append(fin_data_out_inst)

                # save output of each instrument
                if self.args.save_each_inst:
                    sf.write(os.path.join(cur_out_dir, f"{cur_inst_name}_{output_name_tag}.wav"), fin_data_out_inst.transpose(-1, -2), self.args.sample_rate, 'PCM_16')
            # remix
            fin_data_out_mix = sum(inst_outputs)
            sf.write(os.path.join(cur_out_dir, f"mixture_{output_name_tag}.wav"), fin_data_out_mix.transpose(-1, -2), self.args.sample_rate, 'PCM_16')


    # function that segmentize an entire song into batch
    def batchwise_segmentization(self, target_song, song_name, segment_length, discard_last=False):
        assert target_song.shape[-1] >= self.args.segment_length, \
                f"Error : Insufficient duration!\n\t \
                Target song's length is shorter than segment length.\n\t \
                Song name : {song_name}\n\t \
                Consider changing the 'segment_length' or song with sufficient duration"

        # discard restovers (last segment)
        if discard_last:
            target_length = target_song.shape[-1] - target_song.shape[-1] % segment_length
            target_song = target_song[:, :target_length]
        # pad last segment
        else:
            pad_length = segment_length - target_song.shape[-1] % segment_length
            target_song = torch.cat((target_song, torch.zeros(2, pad_length)), axis=-1)

        # segmentize according to the given segment_length
        whole_batch_data = []
        batch_wise_data = []
        for cur_segment_idx in range(target_song.shape[-1]//segment_length):
            batch_wise_data.append(target_song[..., cur_segment_idx*segment_length:(cur_segment_idx+1)*segment_length])
            if len(batch_wise_data)==self.args.batch_size:
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
        record_path = f"{self.output_dir}style_transfer_inference_configurations.txt"
        f = open(record_path, 'w')
        np.savetxt(f, [info], delimiter=" ", fmt="%s")
        f.close()



if __name__ == '__main__':
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    os.environ['MASTER_PORT'] = '8888'

    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    ''' Configurations for music mixing style transfer '''
    currentdir = os.path.dirname(os.path.realpath(__file__))
    default_ckpt_path_enc = os.path.join(os.path.dirname(currentdir), 'weights', 'FXencoder_ps.pt')
    default_ckpt_path_conv = os.path.join(os.path.dirname(currentdir), 'weights', 'MixFXcloner_ps.pt')
    default_norm_feature_path = os.path.join(os.path.dirname(currentdir), 'weights', 'musdb18_fxfeatures_eqcompimagegain.npy')

    import argparse
    import yaml
    parser = argparse.ArgumentParser()

    directory_args = parser.add_argument_group('Directory args')
    # directory paths
    directory_args.add_argument('--target_dir', type=str, default='./samples/style_transfer/')
    directory_args.add_argument('--output_dir', type=str, default=None, help='if no output_dir is specified (None), the results will be saved inside the target_dir')
    directory_args.add_argument('--input_file_name', type=str, default='input')
    directory_args.add_argument('--reference_file_name', type=str, default='reference')
    directory_args.add_argument('--reference_file_name_2interpolate', type=str, default='reference_B')
    # saved weights
    directory_args.add_argument('--ckpt_path_enc', type=str, default=default_ckpt_path_enc)
    directory_args.add_argument('--ckpt_path_conv', type=str, default=default_ckpt_path_conv)
    directory_args.add_argument('--precomputed_normalization_feature', type=str, default=default_norm_feature_path)

    inference_args = parser.add_argument_group('Inference args')
    inference_args.add_argument('--sample_rate', type=int, default=44100)
    inference_args.add_argument('--segment_length', type=int, default=2**19)        # segmentize input according to this duration
    inference_args.add_argument('--segment_length_ref', type=int, default=2**19)    # segmentize reference according to this duration
    # stem-level instruments & separation
    inference_args.add_argument('--instruments', type=str2bool, default=["drums", "bass", "other", "vocals"], help='instrumental tracks to perform style transfer')
    inference_args.add_argument('--stem_level_directory_name', type=str, default='separated')
    inference_args.add_argument('--save_each_inst', type=str2bool, default=False)
    inference_args.add_argument('--do_not_separate', type=str2bool, default=False)
    inference_args.add_argument('--separation_model', type=str, default='mdx_extra')
    # FX normalization
    inference_args.add_argument('--normalize_input', type=str2bool, default=True)
    inference_args.add_argument('--normalization_order', type=str2bool, default=['eq', 'compression', 'imager', 'loudness']) # Effects to be normalized, order matters
    # interpolation
    inference_args.add_argument('--interpolation', type=str2bool, default=False)
    inference_args.add_argument('--interpolate_segments', type=int, default=30)

    device_args = parser.add_argument_group('Device args')
    device_args.add_argument('--workers', type=int, default=1)
    device_args.add_argument('--inference_device', type=str, default='gpu', help="if this option is not set to 'cpu', inference will happen on gpu only if there is a detected one")
    device_args.add_argument('--batch_size', type=int, default=1)   # for processing long audio
    device_args.add_argument('--separation_device', type=str, default='cpu', help="device for performing source separation using Demucs")

    args = parser.parse_args()

    # load network configurations
    with open(os.path.join(currentdir, 'configs.yaml'), 'r') as f:
        configs = yaml.full_load(f)
    args.cfg_encoder = configs['Effects_Encoder']['default']
    args.cfg_converter = configs['TCN']['default']


    # Perform music mixing style transfer
    inference_style_transfer = Mixing_Style_Transfer_Inference(args)
    if args.interpolation:
        inference_style_transfer.inference_interpolation()
    else:
        inference_style_transfer.inference()
    
    
    