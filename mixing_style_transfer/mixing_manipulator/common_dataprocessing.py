"""
Module with common functions for loading training data and preparing minibatches.

AI Music Technology Group, Sony Group Corporation
AI Speech and Sound Group, Sony Europe

This implementation originally belongs to Sony Group Corporation, 
    which has been introduced in the work "Automatic music mixing with deep learning and out-of-domain data".
    Original repo link: https://github.com/sony/FxNorm-automix
"""

import numpy as np
import os
import sys
import functools
import scipy.io.wavfile as wav
import soundfile as sf
from typing import Tuple

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from common_audioeffects import AugmentationChain
from common_miscellaneous import uprint


def load_wav(file_path, mmap=False, convert_float=False):
    """
    Load a WAV file in C_CONTIGUOUS format.

    Args:
        file_path: Path to WAV file (16bit, 24bit or 32bit PCM supported)
        mmap: If `True`, then we do not load the WAV data into memory but use a memory-mapped representation

    Returns:
        fs: Sample rate
        samples: Numpy array (np.int16 or np.int32) with audio [n_samples x n_channels]
    """
    fs, samples = wav.read(file_path, mmap=mmap)

    # ensure that we have a 2d array (monaural files are just loaded as vectors)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    # make sure that we have loaded an integer PCM WAV file as we assume this later
    # when we scale the amplitude
    assert(samples.dtype == np.int16 or samples.dtype == np.int32)
    
    if convert_float:
        conversion_scale = 1. / (1. + np.iinfo(samples.dtype).max)
        samples = samples.astype(dtype=np.float32) * conversion_scale
        
    return fs, samples


def save_wav(file_path, fs, samples, subtype='PCM_16'):
    """
    Save a WAV file (16bit or 32bit PCM).

    Important note: We save here using the same conversion as is used in
                    `generate_data`, i.e., we multiply by `1 + np.iinfo(np.int16).max`
                    or `1 + np.iinfo(np.int32).max` which is a different behavior
                    than `libsndfile` as described here:
                    http://www.mega-nerd.com/libsndfile/FAQ.html#Q010

    Args:
        file_path: Path where to store the WAV file
        fs: Sample rate
        samples: Numpy array (float32 with values in [-1, 1) and shape [n_samples x n_channels])
        subtype: Either `PCM_16` or `PCM_24` or `PCM_32` in order to store as 16bit, 24bit or 32bit PCM file
    """
    assert subtype in ['PCM_16', 'PCM_24', 'PCM_32'], subtype
    
    if subtype == 'PCM_16':
        dtype = np.int16
    else:
        dtype = np.int32

    # convert to int16 (check for clipping)
    
    samples = samples * (1 + np.iinfo(dtype).max)
    if np.min(samples) < np.iinfo(dtype).min or np.max(samples) > np.iinfo(dtype).max:
        uprint(f'WARNING: Clipping occurs for {file_path}.')
        samples_ = samples / (1 + np.iinfo(dtype).max)
        print('max value ', np.max(np.abs(samples_)))
    samples = np.clip(samples, np.iinfo(dtype).min, np.iinfo(dtype).max)
    samples = samples.astype(dtype)

    # store WAV file
    sf.write(file_path, samples, fs, subtype=subtype)


def load_files_lists(path):
    """
    Auxiliary function to find the paths for all mixtures in a database.

    Args:
        path: path to the folder containing the files to list

    Returns:
        list_of_directories: list of directories (= list of songs) in `path`
    """
    # get directories in `path`
    list_of_directories = []
    for folder in os.listdir(path):
        list_of_directories.append(folder)

    return list_of_directories


def create_dataset(path, accepted_sampling_rates, sources, mapped_sources, n_channels=-1, load_to_memory=False,
                   debug=False, verbose=False):
    """
    Prepare data in `path` for training/validation/test set generation.

    Args:
        path: path to the dataset
        accepted_sampling_rates: list of accepted sampling rates
        sources: list of sources
        mapped_sources: list of mapped sources
        n_channels: number of channels
        load_to_memory: whether to load to main memory
        debug: if `True`, then we load only `NUM_SAMPLES_SMALL_DATASET`

    Raises:
        ValueError: mapping of sources not possible is data is not loaded into memory

    Returns:
        data: list of dictionaries with function handles (to load the data)
        directories: list of directories
    """
    NUM_SAMPLES_SMALL_DATASET = 16

    # source mapping currently only works if we load everything into the memory
    if mapped_sources and not load_to_memory:
        raise ValueError('Mapping of sources only supported if data is loaded into the memory.')

    # get directories for dataset
    directories = load_files_lists(path)

    # load all songs for dataset
    if debug:
        data = [dict() for _x in range(np.minimum(NUM_SAMPLES_SMALL_DATASET, len(directories)))]
    else:
        data = [dict() for _x in range(len(directories))]

    material_length = {}  # in seconds
    for i, d in enumerate(directories):
        if verbose:
            uprint(f'Processing mixture ({i+1} of {len(directories)}): {d}')

        # add names of all files in this folder
        files = os.listdir(os.path.join(path, d))
        for f in files:
            src_name = os.path.splitext(f)[0]
            if ((src_name not in sources
                 and src_name not in mapped_sources)):
                 if verbose:
                    uprint(f'\tIgnoring unknown source from file {f}')
            else:
                if src_name not in sources:
                    src_name = mapped_sources[src_name]
                if verbose:
                    uprint(f'\tAdding function handle for "{src_name}" from file {f}')

                _data = load_wav(os.path.join(path, d, f), mmap=not load_to_memory)

                # determine properties from loaded data
                _samplingrate = _data[0]
                _n_channels = _data[1].shape[1]
                _duration = _data[1].shape[0] / _samplingrate

                # collect statistics about data for each source
                if src_name in material_length:
                    material_length[src_name] += _duration
                else:
                    material_length[src_name] = _duration

                # make sure that sample rate and number of channels matches
                if n_channels != -1 and _n_channels != n_channels:
                    raise ValueError(f'File has {_n_channels} '
                                     f'channels but expected {n_channels}.')

                if _samplingrate not in accepted_sampling_rates:
                    raise ValueError(f'File has fs = {_samplingrate}Hz '
                                     f'but expected {accepted_sampling_rates}Hz.')

                # if we already loaded data for this source then append data
                if src_name in data[i]:
                    _data = (_data[0], np.vstack((_data[1],
                                                  data[i][src_name].keywords['file_path_or_data'][1])))
                data[i][src_name] = functools.partial(generate_data,
                                                      file_path_or_data=_data)

        if debug and i == NUM_SAMPLES_SMALL_DATASET-1:
            # load only first `NUM_SAMPLES_SMALL_DATASET` songs
            break

    # delete all entries where we did not find an source file
    idx_empty = [_ for _ in range(len(data)) if len(data[_]) == 0]
    for idx in sorted(idx_empty, reverse=True):
        del data[idx]

    return data, directories

def create_dataset_mixing(path, accepted_sampling_rates, sources, mapped_sources, n_channels=-1, load_to_memory=False,
                   debug=False, pad_wrap_samples=None):
    """
    Prepare data in `path` for training/validation/test set generation.

    Args:
        path: path to the dataset
        accepted_sampling_rates: list of accepted sampling rates
        sources: list of sources
        mapped_sources: list of mapped sources
        n_channels: number of channels
        load_to_memory: whether to load to main memory
        debug: if `True`, then we load only `NUM_SAMPLES_SMALL_DATASET`

    Raises:
        ValueError: mapping of sources not possible is data is not loaded into memory

    Returns:
        data: list of dictionaries with function handles (to load the data)
        directories: list of directories
    """
    NUM_SAMPLES_SMALL_DATASET = 16

    # source mapping currently only works if we load everything into the memory
    if mapped_sources and not load_to_memory:
        raise ValueError('Mapping of sources only supported if data is loaded into the memory.')

    # get directories for dataset
    directories = load_files_lists(path)
    directories.sort()

    # load all songs for dataset
    uprint(f'\nCreating dataset for path={path} ...')

    if debug:
        data = [dict() for _x in range(np.minimum(NUM_SAMPLES_SMALL_DATASET, len(directories)))]
    else:
        data = [dict() for _x in range(len(directories))]

    material_length = {}  # in seconds
    for i, d in enumerate(directories):
        uprint(f'Processing mixture ({i+1} of {len(directories)}): {d}')

        # add names of all files in this folder
        files = os.listdir(os.path.join(path, d))
        _data_mix = []
        _stems_name = []
        for f in files:
            src_name = os.path.splitext(f)[0]
            if ((src_name not in sources
                 and src_name not in mapped_sources)):
                uprint(f'\tIgnoring unknown source from file {f}')
            else:
                if src_name not in sources:
                    src_name = mapped_sources[src_name]
                uprint(f'\tAdding function handle for "{src_name}" from file {f}')

                _data = load_wav(os.path.join(path, d, f), mmap=not load_to_memory)
                
                if pad_wrap_samples:
                    _data = (_data[0], np.pad(_data[1], [(pad_wrap_samples, 0), (0,0)], 'wrap'))

                # determine properties from loaded data
                _samplingrate = _data[0]
                _n_channels = _data[1].shape[1]
                _duration = _data[1].shape[0] / _samplingrate

                # collect statistics about data for each source
                if src_name in material_length:
                    material_length[src_name] += _duration
                else:
                    material_length[src_name] = _duration

                # make sure that sample rate and number of channels matches
                if n_channels != -1 and _n_channels != n_channels:
                    if _n_channels == 1:    # Converts mono to stereo with repeated channels
                        _data = (_data[0], np.repeat(_data[1], 2, axis=-1)) 
                        print("Converted file to stereo by repeating mono channel")
                    else:
                        raise ValueError(f'File has {_n_channels} '
                                         f'channels but expected {n_channels}.')

                if _samplingrate not in accepted_sampling_rates:
                    raise ValueError(f'File has fs = {_samplingrate}Hz '
                                     f'but expected {accepted_sampling_rates}Hz.')

                # if we already loaded data for this source then append data
                if src_name in data[i]:
                    _data = (_data[0], np.vstack((_data[1],
                                                  data[i][src_name].keywords['file_path_or_data'][1])))
                    
                _data_mix.append(_data)
                _stems_name.append(src_name)
                
        data[i]["-".join(_stems_name)] = functools.partial(generate_data,
                                                           file_path_or_data=_data_mix)

        if debug and i == NUM_SAMPLES_SMALL_DATASET-1:
            # load only first `NUM_SAMPLES_SMALL_DATASET` songs
            break

    # delete all entries where we did not find an source file
    idx_empty = [_ for _ in range(len(data)) if len(data[_]) == 0]
    for idx in sorted(idx_empty, reverse=True):
        del data[idx]

    uprint(f'Finished preparation of dataset. '
           f'Found in total the following material (in {len(data)} directories):')
    for src in material_length:
        uprint(f'\t{src}: {material_length[src] / 60.0 / 60.0:.2f} hours')
    return data, directories


def generate_data(file_path_or_data, random_sample_size=None):
    """
    Load one stem/several stems specified by `file_path_or_data`.

    Alternatively, can also be the result of `wav.read()` if the data has already been loaded previously.

    If `file_path_or_data` is a tuple/list, then we load several files and will return also a tuple/list.
    This is useful for cases where we want to make sure to have the same random chunk for several stems.

    If `random_sample_chunk_size` is not None, then only `random_sample_chunk_size` samples are randomly selected.

    Args:
        file_path_or_data: either path to data or the data itself
        random_sample_size: if `random_sample_size` is not None, only `random_sample_size` samples are randomly selected

    Returns:
        samples: data with size `num_samples x num_channels` or a list of samples
    """
    needs_wrapping = False
    if isinstance(file_path_or_data, str):
        needs_wrapping = True  # single file path -> wrap
    if ((type(file_path_or_data[0]) is not list
         and type(file_path_or_data[0]) is not tuple)):
        needs_wrapping = True  # single data -> wrap
    if needs_wrapping:
        file_path_or_data = (file_path_or_data,)

    # create list where we store all samples
    samples = [None] * len(file_path_or_data)

    # load samples from wav file
    for i, fpod in enumerate(file_path_or_data):
        if isinstance(fpod, str):
            _fs, samples[i] = load_wav(fpod)
        else:
            _fs, samples[i] = fpod

    # if `random_sample_chunk_size` is not None, then only select subset
    if random_sample_size is not None:
        # get maximum length of all stems (at least `random_sample_chunk_size`)
        max_length = random_sample_size
        for s in samples:
            max_length = np.maximum(max_length, s.shape[0])

        # make sure that we can select enough audio and that all have the same length `max_length`
        # (for short loops, `random_sample_chunk_size` can be larger than `s.shape[0]`)
        for i, s in enumerate(samples):
            if s.shape[0] < max_length:
                required_padding = max_length - s.shape[0]
                zeros = np.zeros((required_padding // 2 + 1, s.shape[1]),
                                 dtype=s.dtype, order='F')
                samples[i] = np.concatenate([zeros, s, zeros])
                
        # select random part of audio
        idx_start = np.random.randint(max_length)
     
        for i, s in enumerate(samples):
            if idx_start + random_sample_size < s.shape[0]:
                samples[i] = s[idx_start:idx_start + random_sample_size]
            else:
                samples[i] = np.concatenate([s[idx_start:],
                                             s[:random_sample_size - (s.shape[0] - idx_start)]])

    # convert from `int16/int32` to `float32` precision (this will also make a copy)
    for i, s in enumerate(samples):
        conversion_scale = 1. / (1. + np.iinfo(s.dtype).max)
        samples[i] = s.astype(dtype=np.float32) * conversion_scale

    if len(samples) == 1:
        return samples[0]
    else:
        return samples


def create_minibatch(data: list, sources: list, 
                     present_prob: dict, overlap_prob: dict,
                     augmenter: AugmentationChain, augmenter_padding: Tuple[int],
                     batch_size: int, n_samples: int, n_channels: int, idx_songs: dict):
    """
    Create a minibatch.

    This function also handles the case that we do not have a source in one mixture.
    This can, e.g., happen for instrumental pieces that do not have vocals.

    Args:
        data (list): data to create the minibatch from.
        sources (list): list of sources.
        present_prob (dict): probability of a source to be present.
        overlap_prob (dict): probability of overlap.
        augmenter (AugmentationChain): audio effect chain that we want to apply for data augmentation
        augmenter_padding (tuple of ints): padding that we should apply to left/right side of data to avoid
            boundary effects of `augmenter`.
        batch_size (int): number of training samples in one minibatch.
        n_samples (int): number of time samples.
        n_channels (int): number of channels.
        idx_songs (dict): index of songs.

    Returns:
        inp (Numpy array): minibatch, input to the network (i.e. the mixture) of size
            `batch_size x n_samples x n_channels`
        tar (dict with Numpy arrays): dictionary which contains for each source the targets,
            each of the `c_contiguous` ndarrays is `batch_size x n_samples x n_channels`
    """
    # initialize numpy arrays which keep input/targets
    shp = (batch_size, n_samples, n_channels)
    inp = np.zeros(shape=shp, dtype=np.float32, order='C')
    tar = {src: np.zeros(shape=shp, dtype=np.float32, order='C') for src in sources}
    
    # use padding to avoid boundary effects of augmenter
    pad_left = None if augmenter_padding[0] == 0 else augmenter_padding[0]
    pad_right = None if augmenter_padding[1] == 0 else -augmenter_padding[1]

    def augm(i, s, n):
        return augmenter(data[i][s](random_sample_size=n+sum(augmenter_padding)))[pad_left:pad_right]

    # create mini-batch
    for src in sources:
        
        for j in range(batch_size):
            # get song index for this source
            _idx_song = idx_songs[src][j]

            # determine whether this source is present/whether we overlap
            is_present = src not in present_prob or np.random.rand() < present_prob[src]
            is_overlap = src in overlap_prob and np.random.rand() < overlap_prob[src]

            # if song contains source, then add it to input/targetg]
            if src in data[_idx_song] and is_present:
                tar[src][j, ...] = augm(_idx_song, src, n_samples)

                # overlap source with same source from randomly choosen other song
                if is_overlap:
                    idx_overlap_ = np.random.randint(len(data))
                    if idx_overlap_ != _idx_song and src in data[idx_overlap_]:
                        tar[src][j, ...] += augm(idx_overlap_, src, n_samples)

        # compute input
        inp += tar[src]

    # make sure that all have not too large amplitude (check only mixture)
    maxabs_amp = np.maximum(1.0, 1e-6 + np.max(np.abs(inp), axis=(1, 2), keepdims=True))
    inp /= maxabs_amp
    for src in sources:
        tar[src] /= maxabs_amp

    return inp, tar

def create_minibatch_mixing(data: list, sources: list, inputs: list, outputs: list,
                     present_prob: dict, overlap_prob: dict,
                     augmenter: AugmentationChain, augmenter_padding: Tuple[int], augmenter_sources: list,
                     batch_size: int, n_samples: int, n_channels: int, idx_songs: dict):
    """
    Create a minibatch.

    This function also handles the case that we do not have a source in one mixture.
    This can, e.g., happen for instrumental pieces that do not have vocals.

    Args:
        data (list): data to create the minibatch from.
        sources (list): list of sources.
        present_prob (dict): probability of a source to be present.
        overlap_prob (dict): probability of overlap.
        augmenter (AugmentationChain): audio effect chain that we want to apply for data augmentation
        augmenter_padding (tuple of ints): padding that we should apply to left/right side of data to avoid
            boundary effects of `augmenter`.
        augmenter_sources (list): list of sources to augment
        batch_size (int): number of training samples in one minibatch.
        n_samples (int): number of time samples.
        n_channels (int): number of channels.
        idx_songs (dict): index of songs.

    Returns:
        inp (Numpy array): minibatch, input to the network (i.e. the mixture) of size
            `batch_size x n_samples x n_channels`
        tar (dict with Numpy arrays): dictionary which contains for each source the targets,
            each of the `c_contiguous` ndarrays is `batch_size x n_samples x n_channels`
    """
    # initialize numpy arrays which keep input/targets
    shp = (batch_size, n_samples, n_channels)
    stems = {src: np.zeros(shape=shp, dtype=np.float32, order='C') for src in inputs}
    mix = {src: np.zeros(shape=shp, dtype=np.float32, order='C') for src in outputs}
    
    # use padding to avoid boundary effects of augmenter
    pad_left = None if augmenter_padding[0] == 0 else augmenter_padding[0]
    pad_right = None if augmenter_padding[1] == 0 else -augmenter_padding[1]

    def augm(i, n):
        s = list(data[i])[0]
        input_multitracks = data[i][s](random_sample_size=n+sum(augmenter_padding))
        audio_tags = list(data[i])[0].split("-")
        
        # Only applies augmentation to inputs, not output.
        for k, tag in enumerate(audio_tags):
            if tag in augmenter_sources:
                input_multitracks[k] = augmenter(input_multitracks[k])[pad_left:pad_right]
            else:
                input_multitracks[k] = input_multitracks[k][pad_left:pad_right]
        return input_multitracks

    # create mini-batch
    for src in outputs:
        
        for j in range(batch_size):
            # get song index for this source
            _idx_song = idx_songs[src][j]

            multitrack_audio = augm(_idx_song, n_samples)
            
            audio_tags = list(data[_idx_song])[0].split("-")

            for i, tag in enumerate(audio_tags):
                if tag in inputs:
                    stems[tag][j, ...] = multitrack_audio[i]
                if tag in outputs:
                    mix[tag][j, ...] = multitrack_audio[i]

    return stems, mix

