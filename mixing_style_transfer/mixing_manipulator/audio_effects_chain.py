"""
    Implementation of Audio Effects Chain Manipulation for the task 'Mixing Style Transfer'
"""
from glob import glob
import os
import sys

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
sys.path.append(os.path.dirname(currentdir))
from common_audioeffects import *
from common_dataprocessing import create_dataset



# create augmentation effects chain according to targeted effects with their applying probability
def create_effects_augmentation_chain(effects, \
                                        ir_dir_path=None, \
                                        sample_rate=44100, \
                                        shuffle=False, \
                                        parallel=False, \
                                        parallel_weight_factor=None):
    '''
        Args:
            effects (list of tuples or string) : First tuple element is string denoting the target effects. 
                                                    Second tuple element is probability of applying current effects.
            ir_dir_path (string) : directory path that contains directories of impulse responses organized according to RT60
            sample_rate (int) : using sampling rate
            shuffle (boolean) : shuffle FXs inside current FX chain
            parallel (boolean) : compute parallel FX computation (alpha * input + (1-alpha) * manipulated output)
            parallel_weight_factor : the value of alpha for parallel FX computation. default=None : random value in between (0.0, 0.5)
    '''
    fx_list = []
    apply_prob = []
    for cur_fx in effects:
        # store probability to apply current effects. default is to set as 100%
        if isinstance(cur_fx, tuple):
            apply_prob.append(cur_fx[1])
            cur_fx = cur_fx[0]
        else:
            apply_prob.append(1)

        # processors of each audio effects
        if isinstance(cur_fx, AugmentationChain) or isinstance(cur_fx, Processor):
            fx_list.append(cur_fx)
        elif cur_fx.lower()=='gain':
            fx_list.append(Gain())
        elif 'eq' in cur_fx.lower():
            fx_list.append(Equaliser(n_channels=2, sample_rate=sample_rate))
        elif 'comp' in cur_fx.lower():
            fx_list.append(Compressor(sample_rate=sample_rate))
        elif 'expand' in cur_fx.lower():
            fx_list.append(Expander(sample_rate=sample_rate))
        elif 'pan' in cur_fx.lower():
            fx_list.append(Panner())
        elif 'image'in cur_fx.lower():
            fx_list.append(MidSideImager())
        elif 'algorithmic' in cur_fx.lower():
            fx_list.append(AlgorithmicReverb(sample_rate=sample_rate))
        elif 'reverb' in cur_fx.lower():
            # apply algorithmic reverberation if ir_dir_path is not defined
            if ir_dir_path==None:
                fx_list.append(AlgorithmicReverb(sample_rate=sample_rate))
            # apply convolution reverberation
            else:
                IR_paths = glob(f"{ir_dir_path}*/RT60_avg/[!0-]*")
                IR_list = []
                IR_dict = {}
                for IR_path in IR_paths:
                    cur_rt = IR_path.split('/')[-1]
                    if cur_rt not in IR_dict:
                        IR_dict[cur_rt] = []
                    IR_dict[cur_rt].extend(create_dataset(path=IR_path, \
                                                            accepted_sampling_rates=[sample_rate], \
                                                            sources=['impulse_response'], \
                                                            mapped_sources={}, load_to_memory=True, debug=False)[0])
                long_ir_list = []
                for cur_rt in IR_dict:
                    cur_rt_len = int(cur_rt.split('-')[0])
                    if cur_rt_len < 3000:
                        IR_list.append(IR_dict[cur_rt])
                    else:
                        long_ir_list.extend(IR_dict[cur_rt])

                IR_list.append(long_ir_list)
                fx_list.append(ConvolutionalReverb(IR_list, sample_rate))
        else:
            raise ValueError(f"make sure the target effects are in the Augment FX chain : received fx called {cur_fx}")

    aug_chain_in = []
    for cur_i, cur_fx in enumerate(fx_list):
        normalize = False if isinstance(cur_fx, AugmentationChain) or cur_fx.name=='Gain' else True
        aug_chain_in.append((cur_fx, apply_prob[cur_i], normalize))

    return AugmentationChain(fxs=aug_chain_in, shuffle=shuffle, parallel=parallel, parallel_weight_factor=parallel_weight_factor)


# create audio FX-chain according to input instrument
def create_inst_effects_augmentation_chain(inst, \
                                            apply_prob_dict, \
                                            ir_dir_path=None, \
                                            algorithmic=False, \
                                            sample_rate=44100):
    '''
        Args:
            inst (string) : FXmanipulator for target instrument. Current version only distinguishes 'drums' for applying reverberation
            apply_prob_dict (dictionary of (FX name, probability)) : applying proababilities for each FX
            ir_dir_path (string) : directory path that contains directories of impulse responses organized according to RT60
            algorithmic (boolean) : rather to use algorithmic reverberation (True) or convolution reverberation (False)
            sample_rate (int) : using sampling rate
    '''
    reverb_type = 'algorithmic' if algorithmic else 'reverb'
    eq_comp_rand = create_effects_augmentation_chain([('eq', apply_prob_dict['eq']), ('comp', apply_prob_dict['comp'])], \
                                                            ir_dir_path=ir_dir_path, \
                                                            sample_rate=sample_rate, \
                                                            shuffle=True)
    pan_image_rand = create_effects_augmentation_chain([('pan', apply_prob_dict['pan']), ('imager', apply_prob_dict['imager'])], \
                                                            ir_dir_path=ir_dir_path, \
                                                            sample_rate=sample_rate, \
                                                            shuffle=True)
    if inst=='drums':
        # apply reverberation to low frequency with little probability
        low_pass_eq_params = ParameterList()
        low_pass_eq_params.add(Parameter('high_shelf_gain', -50.0, 'float', minimum=-50.0, maximum=-50.0))
        low_pass_eq_params.add(Parameter('high_shelf_freq', 100.0, 'float', minimum=100.0, maximum=100.0))
        low_pass_eq = Equaliser(n_channels=2, \
                                    sample_rate=sample_rate, \
                                    bands=['high_shelf'], \
                                    parameters=low_pass_eq_params)
        reverb_parallel_low = create_effects_augmentation_chain([low_pass_eq, (reverb_type, apply_prob_dict['reverb']*0.01)], \
                                                                ir_dir_path=ir_dir_path, \
                                                                sample_rate=sample_rate, \
                                                                parallel=True, \
                                                                parallel_weight_factor=0.8)
        # high pass eq for drums reverberation
        high_pass_eq_params = ParameterList()
        high_pass_eq_params.add(Parameter('low_shelf_gain', -50.0, 'float', minimum=-50.0, maximum=-50.0))
        high_pass_eq_params.add(Parameter('low_shelf_freq', 100.0, 'float', minimum=100.0, maximum=100.0))
        high_pass_eq = Equaliser(n_channels=2, \
                                    sample_rate=sample_rate, \
                                    bands=['low_shelf'], \
                                    parameters=high_pass_eq_params)
        reverb_parallel_high = create_effects_augmentation_chain([high_pass_eq, (reverb_type, apply_prob_dict['reverb'])], \
                                                                ir_dir_path=ir_dir_path, \
                                                                sample_rate=sample_rate, \
                                                                parallel=True, \
                                                                parallel_weight_factor=0.6)
        reverb_parallel = create_effects_augmentation_chain([reverb_parallel_low, reverb_parallel_high], \
                                                                ir_dir_path=ir_dir_path, \
                                                                sample_rate=sample_rate)
    else:
        reverb_parallel = create_effects_augmentation_chain([(reverb_type, apply_prob_dict['reverb'])], \
                                                                ir_dir_path=ir_dir_path, \
                                                                sample_rate=sample_rate, \
                                                                parallel=True)
    # full effects chain
    effects_chain = create_effects_augmentation_chain([eq_comp_rand, \
                                                            pan_image_rand, \
                                                            reverb_parallel, \
                                                            ('gain', apply_prob_dict['gain'])], \
                                                        ir_dir_path=ir_dir_path, \
                                                        sample_rate=sample_rate)

    return effects_chain

