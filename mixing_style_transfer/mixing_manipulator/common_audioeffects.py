"""
Audio effects for data augmentation.

Several audio effects can be combined into an augmentation chain.

Important note: We assume that the parallelization during training is done using
                multi-processing and not multi-threading. Hence, we do not need the
                `@sox.sox_context()` decorators as discussed in this
                [thread](https://github.com/pseeth/soxbindings/issues/4).

AI Music Technology Group, Sony Group Corporation
AI Speech and Sound Group, Sony Europe


This implementation originally belongs to Sony Group Corporation, 
    which has been introduced in the work "Automatic music mixing with deep learning and out-of-domain data".
    Original repo link: https://github.com/sony/FxNorm-automix
This work modifies a few implementations from the original repo to suit the task.
"""

from itertools import permutations
import logging
import numpy as np
import pymixconsole as pymc
from pymixconsole.parameter import Parameter
from pymixconsole.parameter_list import ParameterList
from pymixconsole.processor import Processor
from random import shuffle
from scipy.signal import oaconvolve
import soxbindings as sox
from typing import List, Optional, Tuple, Union
from numba import jit

# prevent pysox from logging warnings regarding non-opimal timestretch factors
logging.getLogger('sox').setLevel(logging.ERROR)


# Monkey-Patch `Processor` for convenience
# (a) Allow `None` as blocksize if processor can work on variable-length audio
def new_init(self, name, parameters, block_size, sample_rate, dtype='float32'):
    """
    Initialize processor.

    Args:
        self: Reference to object
        name (str): Name of processor.
        parameters (parameter_list): Parameters for this processor.
        block_size (int): Size of blocks for blockwise processing.
            Can also be `None` if full audio can be processed at once.
        sample_rate (int): Sample rate of input audio. Use `None` if effect is independent of this value.
        dtype (str): data type of samples
    """
    self.name = name
    self.parameters = parameters
    self.block_size = block_size
    self.sample_rate = sample_rate
    self.dtype = dtype


# (b) make code simpler
def new_update(self, parameter_name):
    """
    Update processor after randomization of parameters.

    Args:
        self: Reference to object.
        parameter_name (str): Parameter whose value has changed.
    """
    pass


# (c) representation for nice print
def new_repr(self):
    """
    Create human-readable representation.

    Args:
        self: Reference to object.

    Returns:
        string representation of object.
    """
    return f'Processor(name={self.name!r}, parameters={self.parameters!r}'


Processor.__init__ = new_init
Processor.__repr__ = new_repr
Processor.update = new_update


class AugmentationChain:
    """Basic audio Fx chain which is used for data augmentation."""

    def __init__(self,
                 fxs: Optional[List[Tuple[Union[Processor, 'AugmentationChain'], float, bool]]] = [],
                 shuffle: Optional[bool] = False,
                 parallel: Optional[bool] = False,
                 parallel_weight_factor = None,
                 randomize_param_value=True):
        """
        Create augmentation chain from the dictionary `fxs`.

        Args:
            fxs (list of tuples): First tuple element is an instances of `pymc.processor` or `AugmentationChain` that
                we want to use for data augmentation. Second element gives probability that effect should be applied.
                Third element defines, whether the processed signal is normalized by the RMS of the input.
            shuffle (bool): If `True` then order of Fx are changed whenever chain is applied.
        """
        self.fxs = fxs
        self.shuffle = shuffle
        self.parallel = parallel
        self.parallel_weight_factor = parallel_weight_factor
        self.randomize_param_value = randomize_param_value

    def apply_processor(self, x, processor: Processor, rms_normalize):
        """
        Pass audio in `x` through `processor` and output the respective processed audio.

        Args:
            x (Numpy array): Input audio of shape `n_samples` x `n_channels`.
            processor (Processor): Audio effect that we want to apply.
            rms_normalize (bool):  If `True`, the processed signal is normalized by the RMS of the signal.

        Returns:
            Numpy array: Processed audio of shape `n_samples` x `n_channels` (same size as `x')
        """

        n_samples_input = x.shape[0]

        if processor.block_size is None:
            y = processor.process(x)
        else:
            # make sure that n_samples is a multiple of `processor.block_size`
            if x.shape[0] % processor.block_size != 0:
                n_pad = processor.block_size - x.shape[0] % processor.block_size
                x = np.pad(x, ((0, n_pad), (0, 0)), mode='reflective')

            y = np.zeros_like(x)
            for idx in range(0, x.shape[0], processor.block_size):
                y[idx:idx+processor.block_size, :] = processor.process(x[idx:idx+processor.block_size, :])

        if rms_normalize:
            # normalize output energy such that it is the same as the input energy
            scale = np.sqrt(np.mean(np.square(x)) / np.maximum(1e-7, np.mean(np.square(y))))
            y *= scale

        # return audio of same length as x
        return y[:n_samples_input, :]

    def apply_same_processor(self, x_list, processor: Processor, rms_normalize):
        for i in range(len(x_list)):
            x_list[i] = self.apply_processor(x_list[i], processor, rms_normalize)
        
        return x_list

    def __call__(self, x_list):
        """
        Apply the same augmentation chain to audio tracks in list `x_list`.

        Args:
            x_list (list of Numpy array) : List of audio samples of shape `n_samples` x `n_channels`.

        Returns:
            y_list (list of Numpy array) : List of processed audio of same shape as `x_list` where the same effects have been applied.
        """
        # randomly shuffle effect order if `self.shuffle` is True
        if self.shuffle:
            shuffle(self.fxs)

        # apply effects with probabilities given in `self.fxs`
        y_list = x_list.copy()
        for fx, p, rms_normalize in self.fxs:
            if np.random.rand() < p:
                if isinstance(fx, Processor):
                    # randomize all effect parameters (also calls `update()` for each processor)
                    if self.randomize_param_value:
                        fx.randomize()
                    else:
                        fx.update(None)

                    # apply processor
                    y_list = self.apply_same_processor(y_list, fx, rms_normalize)
                else:
                    y_list = fx(y_list)
            
        if self.parallel:
            # weighting factor of input signal in the range of (0.0 ~ 0.5)
            weight_in = self.parallel_weight_factor if self.parallel_weight_factor else np.random.rand() / 2.
            for i in range(len(y_list)):
                y_list[i] = weight_in*x_list[i] + (1-weight_in)*y_list[i]

        return y_list

    def __repr__(self):
        """
        Human-readable representation.

        Returns:
            string representation of object.
        """
        return f'AugmentationChain(fxs={self.fxs!r}, shuffle={self.shuffle!r})'


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DISTORTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def hard_clip(x, threshold_dB, drive):
    """
    Hard clip distortion.

    Args:
        x: input audio
        threshold_dB: threshold
        drive: drive

    Returns:
        (Numpy array): distorted audio
    """
    drive_linear = np.power(10., drive / 20.).astype(np.float32)
    threshold_linear = 10. ** (threshold_dB / 20.)
    return np.clip(x * drive_linear, -threshold_linear, threshold_linear)


def overdrive(x, drive, colour, sample_rate):
    """
    Overdrive distortion.

    Args:
        x: input audio
        drive: Controls the amount of distortion (dB).
        colour: Controls the amount of even harmonic content in the output(dB)
        sample_rate: sampling rate

    Returns:
        (Numpy array): distorted audio
    """
    scale = np.max(np.abs(x))
    if scale > 0.9:
        clips = True
        x = x * (0.9 / scale)
    else:
        clips = False

    tfm = sox.Transformer()
    tfm.overdrive(gain_db=drive, colour=colour)
    y = tfm.build_array(input_array=x, sample_rate_in=sample_rate).astype(np.float32)

    if clips:
        y *= scale / 0.9  # rescale output to original scale
    return y


def hyperbolic_tangent(x, drive):
    """
    Hyperbolic Tanh distortion.

    Args:
        x: input audio
        drive: drive

    Returns:
        (Numpy array): distorted audio
    """
    drive_linear = np.power(10., drive / 20.).astype(np.float32)
    return np.tanh(2. * x * drive_linear)


def soft_sine(x, drive):
    """
    Soft sine distortion.

    Args:
        x: input audio
        drive: drive

    Returns:
        (Numpy array): distorted audio
    """
    drive_linear = np.power(10., drive / 20.).astype(np.float32)
    y = np.clip(x * drive_linear, -np.pi/4.0, np.pi/4.0)
    return np.sin(2. * y)


def bit_crusher(x, bits):
    """
    Bit crusher distortion.

    Args:
        x: input audio
        bits: bits

    Returns:
        (Numpy array): distorted audio
    """
    return np.rint(x * (2 ** bits)) / (2 ** bits)


class Distortion(Processor):
    """
    Distortion processor.

    Processor parameters:
        mode (str): Currently supports the following five modes: hard_clip, waveshaper, soft_sine, tanh, bit_crusher.
            Each mode has different parameters such as threshold, factor, or bits.
        threshold (float): threshold
        drive (float): drive
        factor (float): factor
        limit_range (float): limit range
        bits (int): bits
    """

    def __init__(self, sample_rate, name='Distortion', parameters=None):
        """
        Initialize processor.

        Args:
            sample_rate (int): sample rate.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name, None, block_size=None, sample_rate=sample_rate)
        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('mode', 'hard_clip', 'string',
                                          options=['hard_clip',
                                                   'overdrive',
                                                   'soft_sine',
                                                   'tanh',
                                                   'bit_crusher']))
            self.parameters.add(Parameter('threshold', 0.0, 'float',
                                          units='dB', maximum=0.0, minimum=-20.0))
            self.parameters.add(Parameter('drive', 0.0, 'float',
                                          units='dB', maximum=20.0, minimum=0.0))
            self.parameters.add(Parameter('colour', 20.0, 'float',
                                          maximum=100.0, minimum=0.0))
            self.parameters.add(Parameter('bits', 12, 'int',
                                          maximum=12, minimum=8))

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): distorted audio of size `n_samples x n_channels`.
        """
        if self.parameters.mode.value == 'hard_clip':
            y = hard_clip(x, self.parameters.threshold.value, self.parameters.drive.value)
        elif self.parameters.mode.value == 'overdrive':
            y = overdrive(x, self.parameters.drive.value,
                          self.parameters.colour.value, self.sample_rate)
        elif self.parameters.mode.value == 'soft_sine':
            y = soft_sine(x, self.parameters.drive.value)
        elif self.parameters.mode.value == 'tanh':
            y = hyperbolic_tangent(x, self.parameters.drive.value)
        elif self.parameters.mode.value == 'bit_crusher':
            y = bit_crusher(x, self.parameters.bits.value)

        # If the output has low amplitude, (some distortion settigns can "crush" down the amplitude)
        # Then it`s normalised to the input's amplitude
        x_max = np.max(np.abs(x)) + 1e-8
        o_max = np.max(np.abs(y)) + 1e-8
        if x_max > o_max:
            y = y*(x_max/o_max)

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% EQUALISER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Equaliser(Processor):
    """
    Five band parametric equaliser (two shelves and three central bands).

    All gains are set in dB values and range from `MIN_GAIN` dB to `MAX_GAIN` dB.
    This processor is implemented as cascade of five biquad IIR filters
    that are implemented using the infamous cookbook formulae from RBJ.

    Processor parameters:
        low_shelf_gain (float), low_shelf_freq (float)
        first_band_gain (float), first_band_freq (float), first_band_q (float)
        second_band_gain (float), second_band_freq (float), second_band_q (float)
        third_band_gain (float), third_band_freq (float), third_band_q (float)

    original from https://github.com/csteinmetz1/pymixconsole/blob/master/pymixconsole/processors/equaliser.py
    """

    def __init__(self, n_channels,
                 sample_rate,
                 gain_range=(-15.0, 15.0),
                 q_range=(0.1, 2.0),
                 bands=['low_shelf', 'first_band', 'second_band', 'third_band', 'high_shelf'],
                 hard_clip=False,
                 name='Equaliser', parameters=None):
        """
        Initialize processor.

        Args:
            n_channels (int): Number of audio channels.
            sample_rate (int): Sample rate of audio.
            gain_range (tuple of floats): minimum and maximum gain that can be used.
            q_range (tuple of floats): minimum and maximum q value.
            hard_clip (bool): Whether we clip to [-1, 1.] after processing.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name, parameters=parameters, block_size=None, sample_rate=sample_rate)

        self.n_channels = n_channels

        MIN_GAIN, MAX_GAIN = gain_range
        MIN_Q, MAX_Q = q_range

        if not parameters:
            self.parameters = ParameterList()
            # low shelf parameters -------
            self.parameters.add(Parameter('low_shelf_gain', 0.0, 'float', minimum=MIN_GAIN, maximum=MAX_GAIN))
            self.parameters.add(Parameter('low_shelf_freq', 80.0, 'float', minimum=30.0, maximum=200.0))
            # first band parameters ------
            self.parameters.add(Parameter('first_band_gain', 0.0, 'float', minimum=MIN_GAIN, maximum=MAX_GAIN))
            self.parameters.add(Parameter('first_band_freq', 400.0, 'float', minimum=200.0, maximum=1000.0))
            self.parameters.add(Parameter('first_band_q', 0.7, 'float', minimum=MIN_Q, maximum=MAX_Q))
            # second band parameters -----
            self.parameters.add(Parameter('second_band_gain', 0.0, 'float', minimum=MIN_GAIN, maximum=MAX_GAIN))
            self.parameters.add(Parameter('second_band_freq', 2000.0, 'float', minimum=1000.0, maximum=3000.0))
            self.parameters.add(Parameter('second_band_q', 0.7, 'float', minimum=MIN_Q, maximum=MAX_Q))
            # third band parameters ------
            self.parameters.add(Parameter('third_band_gain', 0.0, 'float', minimum=MIN_GAIN, maximum=MAX_GAIN))
            self.parameters.add(Parameter('third_band_freq', 4000.0, 'float', minimum=3000.0, maximum=8000.0))
            self.parameters.add(Parameter('third_band_q', 0.7, 'float', minimum=MIN_Q, maximum=MAX_Q))
            # high shelf parameters ------
            self.parameters.add(Parameter('high_shelf_gain', 0.0, 'float', minimum=MIN_GAIN, maximum=MAX_GAIN))
            self.parameters.add(Parameter('high_shelf_freq', 8000.0, 'float', minimum=5000.0, maximum=10000.0))

        self.bands = bands
        self.filters = self.setup_filters()
        self.hard_clip = hard_clip

    def setup_filters(self):
        """
        Create IIR filters.

        Returns:
            IIR filters
        """
        filters = {}

        for band in self.bands:

            G = getattr(self.parameters, band + '_gain').value
            fc = getattr(self.parameters, band + '_freq').value
            rate = self.sample_rate

            if band in ['low_shelf', 'high_shelf']:
                Q = 0.707
                filter_type = band
            else:
                Q = getattr(self.parameters, band + '_q').value
                filter_type = 'peaking'

            filters[band] = pymc.components.iirfilter.IIRfilter(G, Q, fc, rate, filter_type, n_channels=self.n_channels)

        return filters

    def update_filter(self, band):
        """
        Update filters.

        Args:
            band (str): Band that should be updated.
        """
        self.filters[band].G = getattr(self.parameters, band + '_gain').value
        self.filters[band].fc = getattr(self.parameters, band + '_freq').value
        self.filters[band].rate = self.sample_rate

        if band in ['first_band', 'second_band', 'third_band']:
            self.filters[band].Q = getattr(self.parameters, band + '_q').value

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        if parameter_name is not None:
            bands = ['_'.join(parameter_name.split('_')[:2])]
        else:
            bands = self.bands

        for band in bands:
            self.update_filter(band)

        for _band, iirfilter in self.filters.items():
            iirfilter.reset_state()

    def reset_state(self):
        """Reset state."""
        for _band, iirfilter in self.filters.items():
            iirfilter.reset_state()

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): equalized audio of size `n_samples x n_channels`.
        """
        for _band, iirfilter in self.filters.items():
            iirfilter.reset_state()
            x = iirfilter.apply_filter(x)

        if self.hard_clip:
            x = np.clip(x, -1.0, 1.0)

        # make sure that we have float32 as IIR filtering returns float64
        x = x.astype(np.float32)

        # make sure that we have two dimensions (if `n_channels == 1`)
        if x.ndim == 1:
            x = x[:, np.newaxis]

        return x


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPRESSOR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
@jit(nopython=True)
def compressor_process(x, threshold, attack_time, release_time, ratio, makeup_gain, sample_rate, yL_prev):
    """
    Apply compressor.

    Args:
        x (Numpy array): audio data.
        threshold: threshold in dB.
        attack_time: attack_time in ms.
        release_time: release_time in ms.
        ratio: ratio.
        makeup_gain: makeup_gain.
        sample_rate: sample rate.
        yL_prev: internal state of the envelop gain.

    Returns:
        compressed audio.
    """
    M = x.shape[0]
    x_g = np.zeros(M)
    x_l = np.zeros(M)
    y_g = np.zeros(M)
    y_l = np.zeros(M)
    c = np.zeros(M)
    yL_prev = 0.
    
    alpha_attack = np.exp(-1/(0.001 * sample_rate * attack_time))
    alpha_release = np.exp(-1/(0.001 * sample_rate * release_time))

    for i in np.arange(M):
        if np.abs(x[i]) < 0.000001:
            x_g[i] = -120.0
        else:
            x_g[i] = 20 * np.log10(np.abs(x[i]))
        
        if ratio > 1:
            if x_g[i] >= threshold:
                y_g[i] = threshold + (x_g[i] - threshold) / ratio
            else:
                y_g[i] = x_g[i]
        elif ratio < 1:
            if x_g[i] <= threshold:
                y_g[i] = threshold + (x_g[i] - threshold) / (1/ratio)
            else:
                y_g[i] = x_g[i]

        x_l[i] = x_g[i] - y_g[i]

        if x_l[i] > yL_prev:
            y_l[i] = alpha_attack * yL_prev + (1 - alpha_attack) * x_l[i]
        else:
            y_l[i] = alpha_release * yL_prev + (1 - alpha_release) * x_l[i]

        c[i] = np.power(10.0, (makeup_gain - y_l[i]) / 20.0)
        yL_prev = y_l[i]

    y = x * c

    return y, yL_prev


class Compressor(Processor):
    """
    Single band stereo dynamic range compressor.

    Processor parameters:
        threshold (float)
        attack_time (float)
        release_time (float)
        ratio (float)
        makeup_gain (float)
    """

    def __init__(self, sample_rate, name='Compressor', parameters=None):
        """
        Initialize processor.

        Args:
            sample_rate (int): Sample rate of input audio.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=sample_rate)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('threshold', -20.0, 'float', units='dB', minimum=-80.0, maximum=-5.0))
            self.parameters.add(Parameter('attack_time', 2.0, 'float', units='ms', minimum=1., maximum=20.0))
            self.parameters.add(Parameter('release_time', 100.0, 'float', units='ms', minimum=50.0, maximum=500.0))
            self.parameters.add(Parameter('ratio', 4.0, 'float', minimum=4., maximum=40.0))
            # we remove makeup_gain parameter inside the Compressor

        # store internal state (for block-wise processing)
        self.yL_prev = None

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): compressed audio of size `n_samples x n_channels`.
        """
        if self.yL_prev is None:
            self.yL_prev = [0.] * x.shape[1]

        if not self.parameters.threshold.value == 0.0 or not self.parameters.ratio.value == 1.0:
            y = np.zeros_like(x)

            for ch in range(x.shape[1]):
                y[:, ch], self.yL_prev[ch] = compressor_process(x[:, ch],
                                                                self.parameters.threshold.value,
                                                                self.parameters.attack_time.value,
                                                                self.parameters.release_time.value,
                                                                self.parameters.ratio.value,
                                                                0.0, # makeup_gain = 0
                                                                self.sample_rate,
                                                                self.yL_prev[ch])
        else:
            y = x

        return y

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        self.yL_prev = None


# %%%%%%%%%%%%%%%%%%%%%%%%%% CONVOLUTIONAL REVERB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class ConvolutionalReverb(Processor):
    """
    Convolutional Reverb.

    Processor parameters:
        wet_dry (float): Wet/dry ratio.
        decay (float): Applies a fade out to the impulse response.
        pre_delay (float): Value in ms. Shifts the IR in time and allows.
            A positive value produces a traditional delay between the dry signal and the wet.
            A negative delay is, in reality, zero delay, but effectively trims off the start of IR,
            so the reverb response begins at a point further in.
    """

    def __init__(self, impulse_responses, sample_rate, name='ConvolutionalReverb', parameters=None):
        """
        Initialize processor.

        Args:
            impulse_responses (list): List with impulse responses created by `common_dataprocessing.create_dataset`
            sample_rate (int): Sample rate that we should assume (used for fade-out computation)
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.

        Raises:
            ValueError: if no impulse responses are provided.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=sample_rate)

        if impulse_responses is None:
            raise ValueError('List of impulse responses must be provided for ConvolutionalReverb processor.')
        self.impulse_responses = impulse_responses

        if not parameters:
            self.parameters = ParameterList()
            self.max_ir_num = len(max(impulse_responses, key=len))
            self.parameters.add(Parameter('index', 0, 'int', minimum=0, maximum=len(impulse_responses)))
            self.parameters.add(Parameter('index_ir', 0, 'int', minimum=0, maximum=self.max_ir_num))
            self.parameters.add(Parameter('wet', 1.0, 'float', minimum=1.0, maximum=1.0))
            self.parameters.add(Parameter('dry', 0.0, 'float', minimum=0.0, maximum=0.0))
            self.parameters.add(Parameter('decay', 1.0, 'float', minimum=1.0, maximum=1.0))
            self.parameters.add(Parameter('pre_delay', 0, 'int', units='ms', minimum=0, maximum=0))

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        # we sample IR with a uniform random distribution according to RT60 values
        chosen_ir_duration = self.impulse_responses[self.parameters.index.value]
        chosen_ir_idx = self.parameters.index_ir.value % len(chosen_ir_duration)
        self.h = np.copy(chosen_ir_duration[chosen_ir_idx]['impulse_response']())

        # fade out the impulse based on the decay setting (starting from peak value)
        if self.parameters.decay.value < 1.:
            idx_peak = np.argmax(np.max(np.abs(self.h), axis=1), axis=0)
            fstart = np.minimum(self.h.shape[0],
                                idx_peak + int(self.parameters.decay.value * (self.h.shape[0] - idx_peak)))
            fstop = np.minimum(self.h.shape[0], fstart + int(0.020*self.sample_rate))  # constant 20 ms fade out
            flen = fstop - fstart

            fade = np.arange(1, flen+1, dtype=self.dtype)/flen
            fade = np.power(0.1, fade * 5)
            self.h[fstart:fstop, :] *= fade[:, np.newaxis]
            self.h = self.h[:fstop]

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): reverbed audio of size `n_samples x n_channels`.
        """
        # reshape IR to the correct size
        n_channels = x.shape[1]
        if self.h.shape[1] == 1 and n_channels > 1:
            self.h = np.hstack([self.h] * n_channels)  # repeat mono IR for multi-channel input
        if self.h.shape[1] > 1 and n_channels == 1:
            self.h = self.h[:, np.random.randint(self.h.shape[1]), np.newaxis]  # randomly choose one IR channel

        if self.parameters.wet.value == 0.0:
            return x
        else:
            # perform convolution to get wet signal
            y = oaconvolve(x, self.h, mode='full', axes=0)

            # cut out wet signal (compensating for the delay that the IR is introducing + predelay)
            idx = np.argmax(np.max(np.abs(self.h), axis=1), axis=0)
            idx += int(0.001 * np.abs(self.parameters.pre_delay.value) * self.sample_rate)

            idx = np.clip(idx, 0, self.h.shape[0]-1)

            y = y[idx:idx+x.shape[0], :]

            # return weighted sum of dry and wet signal
            return self.parameters.dry.value * x + self.parameters.wet.value * y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%% HAAS EFFECT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
def haas_process(x, delay, feedback, wet_channel):
    """
    Add Haas effect to audio.

    Args:
        x (Numpy array): input audio.
        delay: Delay that we apply to one of the channels (in samples).
        feedback: Feedback value.
        wet_channel: Which channel we process (`left` or `right`).

    Returns:
        (Numpy array): Audio with Haas effect.
    """
    y = np.copy(x)
    if wet_channel == 'left':
        y[:, 0] += feedback * np.roll(x[:, 0], delay)
    elif wet_channel == 'right':
        y[:, 1] += feedback * np.roll(x[:, 1], delay)

    return y


class Haas(Processor):
    """
    Haas Effect Processor.

    Randomly selects one channel and applies a short delay to it.

    Processor parameters:
        delay (int)
        feedback (float)
        wet_channel (string)
    """

    def __init__(self, sample_rate, delay_range=(-0.040, 0.040), name='Haas', parameters=None,
                 ):
        """
        Initialize processor.

        Args:
            sample_rate (int): Sample rate of input audio.
            delay_range (tuple of floats): minimum/maximum delay for Haas effect.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=sample_rate)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('delay', int(delay_range[1] * sample_rate), 'int', units='samples',
                                          minimum=int(delay_range[0] * sample_rate),
                                          maximum=int(delay_range[1] * sample_rate)))
            self.parameters.add(Parameter('feedback', 0.35, 'float', minimum=0.33, maximum=0.66))
            self.parameters.add(Parameter('wet_channel', 'left', 'string', options=['left', 'right']))

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): audio with Haas effect of size `n_samples x n_channels`.
        """
        assert x.shape[1] == 1 or x.shape[1] == 2, 'Haas effect only works with monaural or stereo audio.'

        if x.shape[1] < 2:
            x = np.repeat(x, 2, axis=1)

        y = haas_process(x, self.parameters.delay.value,
                         self.parameters.feedback.value, self.parameters.wet_channel.value)

        return y

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        self.reset_state()

    def reset_state(self):
        """Reset state."""
        self.read_idx = 0
        self.write_idx = self.parameters.delay.value
        self.buffer = np.zeros((65536, 2))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PANNER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Panner(Processor):
    """
    Simple stereo panner.

    If input is mono, output is stereo.
    Original edited from https://github.com/csteinmetz1/pymixconsole/blob/master/pymixconsole/processors/panner.py
    """

    def __init__(self, name='Panner', parameters=None):
        """
        Initialize processor.

        Args:
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        # default processor class constructor
        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=None)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('pan', 0.5, 'float', minimum=0., maximum=1.))
            self.parameters.add(Parameter('pan_law', '-4.5dB', 'string',
                                          options=['-4.5dB', 'linear', 'constant_power']))

        # setup the coefficents based on default params
        self.update()

    def _calculate_pan_coefficents(self):
        """
        Calculate panning coefficients from the chosen pan law.

        Based on the set pan law determine the gain value
        to apply for the left and right channel to achieve panning effect.
        This operates on the assumption that the input channel is mono.
        The output data will be stereo at the moment, but could be expanded
        to a higher channel count format.
        The panning value is in the range [0, 1], where
        0 means the signal is panned completely to the left, and
        1 means the signal is apanned copletely to the right.

        Raises:
            ValueError: `self.parameters.pan_law` is not supported.
        """
        self.gains = np.zeros(2, dtype=self.dtype)

        # first scale the linear [0, 1] to [0, pi/2]
        theta = self.parameters.pan.value * (np.pi/2)

        if self.parameters.pan_law.value == 'linear':
            self.gains[0] = ((np.pi/2) - theta) * (2/np.pi)
            self.gains[1] = theta * (2/np.pi)
        elif self.parameters.pan_law.value == 'constant_power':
            self.gains[0] = np.cos(theta)
            self.gains[1] = np.sin(theta)
        elif self.parameters.pan_law.value == '-4.5dB':
            self.gains[0] = np.sqrt(((np.pi/2) - theta) * (2/np.pi) * np.cos(theta))
            self.gains[1] = np.sqrt(theta * (2/np.pi) * np.sin(theta))
        else:
            raise ValueError(f'Invalid pan_law {self.parameters.pan_law.value}.')


    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): panned audio of size `n_samples x n_channels`.
        """
        assert x.shape[1] == 1 or x.shape[1] == 2, 'Panner only works with monaural or stereo audio.'

        if x.shape[1] < 2:
            x = np.repeat(x, 2, axis=1)


        return x * self.gains

    def update(self, parameter_name=None):
        """
        Update processor after randomization of parameters.

        Args:
            parameter_name (str): Parameter whose value has changed.
        """
        self._calculate_pan_coefficents()

    def reset_state(self):
        """Reset state."""
        self._output_buffer = np.empty([self.block_size, 2])
        self.update()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% STEREO IMAGER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class MidSideImager(Processor):
    def __init__(self, name='IMAGER', parameters=None):
        super().__init__(name, parameters=parameters, block_size=None, sample_rate=None)

        if not parameters:
            self.parameters = ParameterList()
            # values of 0.0~1.0 indicate making the signal more centered while 1.0~2.0 means making the signal more wider
            self.parameters.add(Parameter("bal",  0.0,    "float", processor=self, minimum=0.0,    maximum=2.0))

    def process(self, data):
        """
        # input shape : [signal length, 2]
        ### note! stereo imager won't work if the input signal is a mono signal (left==right)
        ### if you want to apply stereo imager to a mono signal, first stereoize it with Haas effects
        """

        # to mid-side channels
        mid, side = self.lr_to_ms(data[:,0], data[:,1])
        # apply mid-side weights according to energy
        mid_e, side_e = np.sum(mid**2), np.sum(side**2)
        total_e = mid_e + side_e
        # apply weights
        max_side_multiplier = np.sqrt(total_e / (side_e + 1e-3))
        # compute current multiply factor
        cur_bal = round(getattr(self.parameters, "bal").value, 3)
        side_gain = cur_bal if cur_bal <= 1. else max_side_multiplier * (cur_bal-1)
        # multiply weighting factor
        new_side = side * side_gain
        new_side_e = side_e * (side_gain ** 2)
        left_mid_e = total_e - new_side_e
        mid_gain = np.sqrt(left_mid_e / (mid_e + 1e-3))
        new_mid = mid * mid_gain
        # convert back to left-right channels
        left, right = self.ms_to_lr(new_mid, new_side)
        imaged = np.stack([left, right], 1)

        return imaged

    # left-right channeled signal to mid-side signal
    def lr_to_ms(self, left, right):
        mid = left + right
        side = left - right
        return mid, side

    # mid-side channeled signal to left-right signal
    def ms_to_lr(self, mid, side):
        left = (mid + side) / 2
        right = (mid - side) / 2
        return left, right

    def update(self, parameter_name=None):
        return parameter_name


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% GAIN %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Gain(Processor):
    """
    Gain Processor.

    Applies gain in dB and can also randomly inverts polarity.

    Processor parameters:
        gain (float): Gain that should be applied (dB scale).
        invert (bool): If True, then we also invert the waveform.
    """

    def __init__(self, name='Gain', parameters=None):
        """
        Initialize processor.

        Args:
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name, parameters=parameters, block_size=None, sample_rate=None)

        if not parameters:
            self.parameters = ParameterList()
            # self.parameters.add(Parameter('gain', 1.0, 'float', units='dB', minimum=-12.0, maximum=6.0))
            self.parameters.add(Parameter('gain', 1.0, 'float', units='dB', minimum=-6.0, maximum=9.0))
            self.parameters.add(Parameter('invert', False, 'bool'))

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): gain-augmented audio of size `n_samples x n_channels`.
        """
        gain = 10 ** (self.parameters.gain.value / 20.)
        if self.parameters.invert.value:
            gain = -gain
        return gain * x


# %%%%%%%%%%%%%%%%%%%%%%% SIMPLE CHANNEL SWAP %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class SwapChannels(Processor):
    """
    Swap channels in multi-channel audio.

    Processor parameters:
        index (int) Selects the permutation that we are using.
            Please note that "no permutation" is one of the permutations in `self.permutations` at index `0`.
    """

    def __init__(self, n_channels, name='SwapChannels', parameters=None):
        """
        Initialize processor.

        Args:
            n_channels (int): Number of channels in audio that we want to process.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=None)

        self.permutations = tuple(permutations(range(n_channels), n_channels))

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('index', 0, 'int', minimum=0, maximum=len(self.permutations)))

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): channel-swapped audio of size `n_samples x n_channels`.
        """
        return x[:, self.permutations[self.parameters.index.value]]


# %%%%%%%%%%%%%%%%%%%%%%% Monauralize %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Monauralize(Processor):
    """
    Monauralizes audio (i.e., removes spatial information).

    Process parameters:
        seed_channel (int): channel that we use for overwriting the others.
    """

    def __init__(self, n_channels, name='Monauralize', parameters=None):
        """
        Initialize processor.

        Args:
            n_channels (int): Number of channels in audio that we want to process.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=None)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('seed_channel', 0, 'int', minimum=0, maximum=n_channels))

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): monauralized audio of size `n_samples x n_channels`.
        """
        return np.tile(x[:, [self.parameters.seed_channel.value]], (1, x.shape[1]))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PITCH SHIFT %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class PitchShift(Processor):
    """
    Simple pitch shifter using SoX and soxbindings (https://github.com/pseeth/soxbindings).

    Processor parameters:
        steps (float): Pitch shift as positive/negative semitones
        quick (bool): If True, this effect will run faster but with lower sound quality.
    """

    def __init__(self, sample_rate, fix_length=True, name='PitchShift', parameters=None):
        """
        Initialize processor.

        Args:
            sample_rate (int): Sample rate of input audio.
            fix_length (bool): If True, then output has same length as input.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=sample_rate)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('steps', 0.0, 'float', minimum=-6., maximum=6.))
            self.parameters.add(Parameter('quick', False, 'bool'))

        self.fix_length = fix_length
        self.clips = False

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): pitch-shifted audio of size `n_samples x n_channels`.
        """
        if self.parameters.steps.value == 0.0:
            y = x
        else:
            scale = np.max(np.abs(x))
            if scale > 0.9:
                clips = True
                x = x * (0.9 / scale)
            else:
                clips = False

            tfm = sox.Transformer()
            tfm.pitch(self.parameters.steps.value, quick=bool(self.parameters.quick.value))
            y = tfm.build_array(input_array=x, sample_rate_in=self.sample_rate).astype(np.float32)

            if clips:
                y *= scale / 0.9  # rescale output to original scale

        if self.fix_length:
            n_samples_input = x.shape[0]
            n_samples_output = y.shape[0]
            if n_samples_input < n_samples_output:
                idx1 = (n_samples_output - n_samples_input) // 2
                idx2 = idx1 + n_samples_input
                y = y[idx1:idx2]
            elif n_samples_input > n_samples_output:
                n_pad = n_samples_input - n_samples_output
                y = np.pad(y, ((n_pad//2, n_pad - n_pad//2), (0, 0)))

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TIME STRETCH %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class TimeStretch(Processor):
    """
    Simple time stretcher using SoX and soxbindings (https://github.com/pseeth/soxbindings).

    Processor parameters:
        factor (float): Time stretch factor.
        quick (bool): If True, this effect will run faster but with lower sound quality.
        stretch_type (str): Algorithm used for stretching (`tempo` or `stretch`).
        audio_type (str): Sets which time segments are most optmial when finding
            the best overlapping points for time stretching.
    """

    def __init__(self, sample_rate, fix_length=True, name='TimeStretch', parameters=None):
        """
        Initialize processor.

        Args:
            sample_rate (int): Sample rate of input audio.
            fix_length (bool): If True, then output has same length as input.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=sample_rate)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('factor', 1.0, 'float', minimum=1/1.33, maximum=1.33))
            self.parameters.add(Parameter('quick', False, 'bool'))
            self.parameters.add(Parameter('stretch_type', 'tempo', 'string', options=['tempo', 'stretch']))
            self.parameters.add(Parameter('audio_type', 'l', 'string', options=['m', 's', 'l']))

        self.fix_length = fix_length

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): time-stretched audio of size `n_samples x n_channels`.
        """
        if self.parameters.factor.value == 1.0:
            y = x
        else:
            scale = np.max(np.abs(x))
            if scale > 0.9:
                clips = True
                x = x * (0.9 / scale)
            else:
                clips = False

            tfm = sox.Transformer()
            if self.parameters.stretch_type.value == 'stretch':
                tfm.stretch(self.parameters.factor.value)
            elif self.parameters.stretch_type.value == 'tempo':
                tfm.tempo(self.parameters.factor.value,
                          audio_type=self.parameters.audio_type.value,
                          quick=bool(self.parameters.quick.value))
            y = tfm.build_array(input_array=x, sample_rate_in=self.sample_rate).astype(np.float32)

            if clips:
                y *= scale / 0.9  # rescale output to original scale

        if self.fix_length:
            n_samples_input = x.shape[0]
            n_samples_output = y.shape[0]
            if n_samples_input < n_samples_output:
                idx1 = (n_samples_output - n_samples_input) // 2
                idx2 = idx1 + n_samples_input
                y = y[idx1:idx2]
            elif n_samples_input > n_samples_output:
                n_pad = n_samples_input - n_samples_output
                y = np.pad(y, ((n_pad//2, n_pad - n_pad//2), (0, 0)))

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PLAYBACK SPEED %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class PlaybackSpeed(Processor):
    """
    Simple playback speed effect using SoX and soxbindings (https://github.com/pseeth/soxbindings).

    Processor parameters:
        factor (float): Playback speed factor.
    """

    def __init__(self, sample_rate, fix_length=True, name='PlaybackSpeed', parameters=None):
        """
        Initialize processor.

        Args:
            sample_rate (int): Sample rate of input audio.
            fix_length (bool): If True, then output has same length as input.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=sample_rate)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('factor', 1.0, 'float', minimum=1./1.33, maximum=1.33))

        self.fix_length = fix_length

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): resampled audio of size `n_samples x n_channels`.
        """
        if self.parameters.factor.value == 1.0:
            y = x
        else:
            scale = np.max(np.abs(x))
            if scale > 0.9:
                clips = True
                x = x * (0.9 / scale)
            else:
                clips = False

            tfm = sox.Transformer()
            tfm.speed(self.parameters.factor.value)
            y = tfm.build_array(input_array=x, sample_rate_in=self.sample_rate).astype(np.float32)

            if clips:
                y *= scale / 0.9  # rescale output to original scale

        if self.fix_length:
            n_samples_input = x.shape[0]
            n_samples_output = y.shape[0]
            if n_samples_input < n_samples_output:
                idx1 = (n_samples_output - n_samples_input) // 2
                idx2 = idx1 + n_samples_input
                y = y[idx1:idx2]
            elif n_samples_input > n_samples_output:
                n_pad = n_samples_input - n_samples_output
                y = np.pad(y, ((n_pad//2, n_pad - n_pad//2), (0, 0)))

        return y


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BEND %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class Bend(Processor):
    """
    Simple bend effect using SoX and soxbindings (https://github.com/pseeth/soxbindings).

    Processor parameters:
        n_bends (int): Number of segments or intervals to pitch shift
    """

    def __init__(self, sample_rate, pitch_range=(-600, 600), fix_length=True, name='Bend', parameters=None):
        """
        Initialize processor.

        Args:
            sample_rate (int): Sample rate of input audio.
            pitch_range (tuple of ints): min and max pitch bending ranges in cents
            fix_length (bool): If True, then output has same length as input.
            name (str): Name of processor.
            parameters (parameter_list): Parameters for this processor.
        """
        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=sample_rate)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter('n_bends', 2, 'int', minimum=2, maximum=10))
        self.pitch_range_min, self.pitch_range_max = pitch_range

    def process(self, x):
        """
        Process audio.

        Args:
            x (Numpy array): input audio of size `n_samples x n_channels`.

        Returns:
            (Numpy array): pitch-bended audio of size `n_samples x n_channels`.
        """
        n_bends = self.parameters.n_bends.value
        max_length = x.shape[0] / self.sample_rate

        # Generates random non-overlapping segments
        delta = 1. / self.sample_rate
        boundaries = np.sort(delta + np.random.rand(n_bends-1) * (max_length - delta))

        start, end = np.zeros(n_bends), np.zeros(n_bends)
        start[0] = delta
        for i, b in enumerate(boundaries):
            end[i] = b
            start[i+1] = b
        end[-1] = max_length

        # randomly sample pitch-shifts in cents
        cents = np.random.randint(self.pitch_range_min, self.pitch_range_max+1, n_bends)

        # remove segment if cent value is zero or start == end (as SoX does not allow such values)
        idx_keep = np.logical_and(cents != 0, start != end)
        n_bends, start, end, cents = sum(idx_keep), start[idx_keep], end[idx_keep], cents[idx_keep]

        scale = np.max(np.abs(x))
        if scale > 0.9:
            clips = True
            x = x * (0.9 / scale)
        else:
            clips = False

        tfm = sox.Transformer()
        tfm.bend(n_bends=int(n_bends), start_times=list(start), end_times=list(end), cents=list(cents))
        y = tfm.build_array(input_array=x, sample_rate_in=self.sample_rate).astype(np.float32)

        if clips:
            y *= scale / 0.9  # rescale output to original scale

        return y

    
    
    
    
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% ALGORITHMIC REVERB %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
class AlgorithmicReverb(Processor):
    def __init__(self, name="algoreverb", parameters=None, sample_rate=44100, **kwargs):

        super().__init__(name=name, parameters=parameters, block_size=None, sample_rate=sample_rate, **kwargs)

        if not parameters:
            self.parameters = ParameterList()
            self.parameters.add(Parameter("room_size",   0.5, "float", minimum=0.05, maximum=0.85))
            self.parameters.add(Parameter("damping",     0.1, "float", minimum=0.0,  maximum=1.0))
            self.parameters.add(Parameter("dry_mix",     0.9, "float", minimum=0.0,  maximum=1.0))
            self.parameters.add(Parameter("wet_mix",     0.1, "float", minimum=0.0,  maximum=1.0))
            self.parameters.add(Parameter("width",       0.7, "float", minimum=0.0,  maximum=1.0))
        
        # Tuning
        self.stereospread = 23
        self.scalegain    = 0.2


    def process(self, data):

        if data.ndim >= 2:
            dataL = data[:,0]
            if data.shape[1] == 2:
                dataR = data[:,1]
            else:
                dataR = data[:,0]
        else:
            dataL = data
            dataR = data

        output = np.zeros((data.shape[0], 2))

        xL, xR = self.process_filters(dataL.copy(), dataR.copy())

        wet1_g = self.parameters.wet_mix.value * ((self.parameters.width.value/2) + 0.5)
        wet2_g = self.parameters.wet_mix.value * ((1-self.parameters.width.value)/2)
        dry_g  = self.parameters.dry_mix.value

        output[:,0] = (wet1_g * xL) + (wet2_g * xR) + (dry_g * dataL)
        output[:,1] = (wet1_g * xR) + (wet2_g * xL) + (dry_g * dataR)

        return output

    def process_filters(self, dataL, dataR):

        xL  = self.combL1.process(dataL.copy() * self.scalegain)
        xL += self.combL2.process(dataL.copy() * self.scalegain)
        xL += self.combL3.process(dataL.copy() * self.scalegain)
        xL += self.combL4.process(dataL.copy() * self.scalegain)
        xL  = self.combL5.process(dataL.copy() * self.scalegain)
        xL += self.combL6.process(dataL.copy() * self.scalegain)
        xL += self.combL7.process(dataL.copy() * self.scalegain)
        xL += self.combL8.process(dataL.copy() * self.scalegain)

        xR  = self.combR1.process(dataR.copy() * self.scalegain)
        xR += self.combR2.process(dataR.copy() * self.scalegain)
        xR += self.combR3.process(dataR.copy() * self.scalegain)
        xR += self.combR4.process(dataR.copy() * self.scalegain)
        xR  = self.combR5.process(dataR.copy() * self.scalegain)
        xR += self.combR6.process(dataR.copy() * self.scalegain)
        xR += self.combR7.process(dataR.copy() * self.scalegain)
        xR += self.combR8.process(dataR.copy() * self.scalegain)

        yL1 = self.allpassL1.process(xL)
        yL2 = self.allpassL2.process(yL1)
        yL3 = self.allpassL3.process(yL2)
        yL4 = self.allpassL4.process(yL3)

        yR1 = self.allpassR1.process(xR)
        yR2 = self.allpassR2.process(yR1)
        yR3 = self.allpassR3.process(yR2)
        yR4 = self.allpassR4.process(yR3)

        return yL4, yR4

    def update(self, parameter_name):

        rs = self.parameters.room_size.value
        dp = self.parameters.damping.value
        ss = self.stereospread

        # initialize allpass and feedback comb-filters
        # (with coefficients optimized for fs=44.1kHz)
        self.allpassL1 = pymc.components.allpass.Allpass(556,    rs, self.block_size)
        self.allpassR1 = pymc.components.allpass.Allpass(556+ss, rs, self.block_size)
        self.allpassL2 = pymc.components.allpass.Allpass(441,    rs, self.block_size)
        self.allpassR2 = pymc.components.allpass.Allpass(441+ss, rs, self.block_size)
        self.allpassL3 = pymc.components.allpass.Allpass(341,    rs, self.block_size)
        self.allpassR3 = pymc.components.allpass.Allpass(341+ss, rs, self.block_size)
        self.allpassL4 = pymc.components.allpass.Allpass(225,    rs, self.block_size)
        self.allpassR4 = pymc.components.allpass.Allpass(255+ss, rs, self.block_size)    

        self.combL1 = pymc.components.comb.Comb(1116,    dp, rs, self.block_size)
        self.combR1 = pymc.components.comb.Comb(1116+ss, dp, rs, self.block_size)
        self.combL2 = pymc.components.comb.Comb(1188,    dp, rs, self.block_size)
        self.combR2 = pymc.components.comb.Comb(1188+ss, dp, rs, self.block_size)
        self.combL3 = pymc.components.comb.Comb(1277,    dp, rs, self.block_size)
        self.combR3 = pymc.components.comb.Comb(1277+ss, dp, rs, self.block_size)
        self.combL4 = pymc.components.comb.Comb(1356,    dp, rs, self.block_size)
        self.combR4 = pymc.components.comb.Comb(1356+ss, dp, rs, self.block_size)
        self.combL5 = pymc.components.comb.Comb(1422,    dp, rs, self.block_size)
        self.combR5 = pymc.components.comb.Comb(1422+ss, dp, rs, self.block_size)
        self.combL6 = pymc.components.comb.Comb(1491,    dp, rs, self.block_size)
        self.combR6 = pymc.components.comb.Comb(1491+ss, dp, rs, self.block_size)
        self.combL7 = pymc.components.comb.Comb(1557,    dp, rs, self.block_size)
        self.combR7 = pymc.components.comb.Comb(1557+ss, dp, rs, self.block_size)
        self.combL8 = pymc.components.comb.Comb(1617,    dp, rs, self.block_size)
        self.combR8 = pymc.components.comb.Comb(1617+ss, dp, rs, self.block_size)

