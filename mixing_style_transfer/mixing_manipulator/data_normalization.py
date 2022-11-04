"""
    Implementation of the 'audio effects chain normalization'
"""
import numpy as np
import scipy

import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from utils_data_normalization import *
from normalization_imager import *


'''
    Audio Effects Chain Normalization
    process: normalizes input stems according to given precomputed features
'''
class Audio_Effects_Normalizer:
    def __init__(self, precomputed_feature_path, \
                    STEMS=['drums', 'bass', 'other', 'vocals'], \
                    EFFECTS=['eq', 'compression', 'imager', 'loudness']):
        self.STEMS = STEMS # Stems to be normalized
        self.EFFECTS = EFFECTS # Effects to be normalized, order matters

        # Audio settings
        self.SR = 44100
        self.SUBTYPE = 'PCM_16'

        # General Settings
        self.FFT_SIZE = 2**16
        self.HOP_LENGTH = self.FFT_SIZE//4

        # Loudness
        self.NTAPS = 1001
        self.LUFS = -30
        self.MIN_DB = -40 # Min amplitude to apply EQ matching

        # Compressor
        self.COMP_USE_EXPANDER = False
        self.COMP_PEAK_NORM = -10.0
        self.COMP_TRUE_PEAK = False
        self.COMP_PERCENTILE = 75 # features_mean (v1) was done with 25
        self.COMP_MIN_TH = -40
        self.COMP_MAX_RATIO = 20
        comp_settings = {key:{} for key in self.STEMS}
        for key in comp_settings:
            if key == 'vocals':
                comp_settings[key]['attack'] = 7.5
                comp_settings[key]['release'] = 400.0
                comp_settings[key]['ratio'] = 4
                comp_settings[key]['n_mels'] = 128
            elif key == 'drums':
                comp_settings[key]['attack'] = 10.0
                comp_settings[key]['release'] = 180.0
                comp_settings[key]['ratio'] = 6
                comp_settings[key]['n_mels'] = 128
            elif key == 'bass':
                comp_settings[key]['attack'] = 10.0
                comp_settings[key]['release'] = 500.0
                comp_settings[key]['ratio'] = 5
                comp_settings[key]['n_mels'] = 16
            elif key == 'other':
                comp_settings[key]['attack'] = 15.0
                comp_settings[key]['release'] = 666.0
                comp_settings[key]['ratio'] = 4
                comp_settings[key]['n_mels'] = 128
        self.comp_settings = comp_settings

        # Load Pre-computed Audio Effects Features
        features_mean = np.load(precomputed_feature_path, allow_pickle='TRUE')[()]
        self.features_mean = self.smooth_feature(features_mean)


    # normalize current audio input with the order of designed audio FX
    def normalize_audio(self, audio, src):
        assert src in self.STEMS

        normalized_audio = audio
        for cur_effect in self.EFFECTS:
            normalized_audio = self.normalize_audio_per_effect(normalized_audio, src=src, effect=cur_effect)

        return normalized_audio


    # normalize current audio input with current targeted audio FX
    def normalize_audio_per_effect(self, audio, src, effect):
        audio = audio.astype(dtype=np.float32)
        audio_track = np.pad(audio, ((self.FFT_SIZE, self.FFT_SIZE), (0, 0)), mode='constant')
        
        assert len(audio_track.shape) == 2  # Always expects two dimensions
        
        if audio_track.shape[1] == 1:    # Converts mono to stereo with repeated channels
            audio_track = np.repeat(audio_track, 2, axis=-1)
            
        output_audio = audio_track.copy()
        
        max_db = amp_to_db(np.max(np.abs(output_audio)))
        if max_db > self.MIN_DB:
        
            if effect == 'eq':
                # normalize each channel
                for ch in range(audio_track.shape[1]):
                    audio_eq_matched = get_eq_matching(output_audio[:, ch],
                                                        self.features_mean[effect][src],
                                                        sr=self.SR,
                                                        n_fft=self.FFT_SIZE,
                                                        hop_length=self.HOP_LENGTH,
                                                        min_db=self.MIN_DB,
                                                        ntaps=self.NTAPS,
                                                        lufs=self.LUFS)
                    

                    np.copyto(output_audio[:,ch], audio_eq_matched)

            elif effect == 'compression':
                assert(len(self.features_mean[effect][src])==2)
                # normalize each channel
                for ch in range(audio_track.shape[1]):
                    try:
                        audio_comp_matched = get_comp_matching(output_audio[:, ch],
                                                                self.features_mean[effect][src][0], 
                                                                self.features_mean[effect][src][1],
                                                                self.comp_settings[src]['ratio'],
                                                                self.comp_settings[src]['attack'],
                                                                self.comp_settings[src]['release'],
                                                                sr=self.SR,
                                                                min_db=self.MIN_DB,
                                                                min_th=self.COMP_MIN_TH, 
                                                                comp_peak_norm=self.COMP_PEAK_NORM,
                                                                max_ratio=self.COMP_MAX_RATIO,
                                                                n_mels=self.comp_settings[src]['n_mels'],
                                                                true_peak=self.COMP_TRUE_PEAK,
                                                                percentile=self.COMP_PERCENTILE, 
                                                                expander=self.COMP_USE_EXPANDER)

                        np.copyto(output_audio[:,ch], audio_comp_matched[:, 0])
                    except:
                        break

            elif effect == 'loudness':
                output_audio = fx_utils.lufs_normalize(output_audio, self.SR, self.features_mean[effect][src], log=False)
                
            elif effect == 'imager':
                # threshold of applying Haas effects
                mono_threshold = 0.99 if src=='bass' else 0.975
                audio_imager_matched = normalize_imager(output_audio, \
                                                        target_side_mid_bal=self.features_mean[effect][src], \
                                                        mono_threshold=mono_threshold, \
                                                        sr=self.SR)

                np.copyto(output_audio, audio_imager_matched)
        
        output_audio = output_audio[self.FFT_SIZE:self.FFT_SIZE+audio.shape[0]]
        return output_audio


    def smooth_feature(self, feature_dict_):
        
        for effect in self.EFFECTS:
            for key in self.STEMS:
                if effect == 'eq':
                    if key in ['other', 'vocals']:
                        f = 401
                    else:
                        f = 151
                    feature_dict_[effect][key] = scipy.signal.savgol_filter(feature_dict_[effect][key],
                                                                            f, 1, mode='mirror')
                elif effect == 'panning':
                    feature_dict_[effect][key] = scipy.signal.savgol_filter(feature_dict_[effect][key],
                                                                            501, 1, mode='mirror')
        return feature_dict_

