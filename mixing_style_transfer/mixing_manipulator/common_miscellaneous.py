"""
Common miscellaneous functions.

AI Music Technology Group, Sony Group Corporation
AI Speech and Sound Group, Sony Europe

This implementation originally belongs to Sony Group Corporation, 
    which has been introduced in the work "Automatic music mixing with deep learning and out-of-domain data".
    Original repo link: https://github.com/sony/FxNorm-automix
"""
import os
import psutil
import sys
import numpy as np
import librosa
import torch
import math


def uprint(s):
    """
    Unbuffered print to stdout.

    We also flush stderr to have the log-file in sync.

    Args:
        s: string to print
    """
    print(s)
    sys.stdout.flush()
    sys.stderr.flush()


def recursive_getattr(obj, attr):
    """
    Run `getattr` recursively (e.g., for `fc1.weight`).

    Args:
        obj: object
        attr: attribute to get

    Returns:
        object
    """
    for a in attr.split('.'):
        obj = getattr(obj, a)
    return obj


def compute_stft(samples, hop_length, fft_size, stft_window):
    """
    Compute the STFT of `samples` applying a Hann window of size `FFT_SIZE`, shifted for each frame by `hop_length`.

    Args:
        samples: num samples x channels
        hop_length: window shift in samples
        fft_size: FFT size which is also the window size
        stft_window: STFT analysis window

    Returns:
        stft: frames x channels x freqbins
    """
    n_channels = samples.shape[1]
    n_frames = 1+int((samples.shape[0] - fft_size)/hop_length)
    stft = np.empty((n_frames, n_channels, fft_size//2+1), dtype=np.complex64)

    # convert into f_contiguous (such that [:,n] slicing is c_contiguous)
    samples = np.asfortranarray(samples)

    for n in range(n_channels):
        # compute STFT (output has size `n_frames x N_BINS`)
        stft[:, n, :] = librosa.stft(samples[:, n],
                                     n_fft=fft_size,
                                     hop_length=hop_length,
                                     window=stft_window,
                                     center=False).transpose()
    return stft


def compute_istft(stft, hop_length, stft_window):
    """
    Compute the inverse STFT of `stft`.

    Args:
        stft: frames x channels x freqbins
        hop_length: window shift in samples
        stft_window: STFT synthesis window

    Returns:
        samples: num samples x channels
    """
    for n in range(stft.shape[1]):
        s = librosa.istft(stft[:, n, :].transpose(),
                          hop_length=hop_length, window=stft_window, center=False)
        if n == 0:
            samples = s
        else:
            samples = np.column_stack((samples, s))

    # ensure that we have a 2d array (monaural files are just loaded as vectors)
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]

    return samples


def get_size(obj):
    """
    Recursively find size of objects (in bytes).

    Args:
        obj: object

    Returns:
        size of object
    """
    size = sys.getsizeof(obj)

    import functools

    if isinstance(obj, dict):
        size += sum([get_size(v) for v in obj.values()])
        size += sum([get_size(k) for k in obj.keys()])
    elif isinstance(obj, functools.partial):
        size += sum([get_size(v) for v in obj.keywords.values()])
        size += sum([get_size(k) for k in obj.keywords.keys()])
    elif isinstance(obj, list):
        size += sum([get_size(i) for i in obj])
    elif isinstance(obj, tuple):
        size += sum([get_size(i) for i in obj])
    return size


def get_process_memory():
    """
    Return memory consumption in GBytes.

    Returns:
        memory used by the process
    """
    return psutil.Process(os.getpid()).memory_info()[0] / (2 ** 30)


def check_complete_convolution(input_size, kernel_size, stride=1,
                               padding=0, dilation=1, note=''):
    """
    Check where the convolution is complete.

    Returns true if no time steps left over in a Conv1d

    Args:
        input_size: size of input
        kernel_size: size of kernel
        stride: stride
        padding: padding
        dilation: dilation
        note: string for additional notes
    """
    is_complete = ((input_size + 2*padding - dilation * (kernel_size - 1) - 1)
                   / stride + 1).is_integer()
    uprint(f'{note} {is_complete}')


def pad_to_shape(x: torch.Tensor, y: int) -> torch.Tensor:
    """
    Right-pad or right-trim first argument last dimension to have same size as second argument.

    Args:
        x: Tensor to be padded.
        y: Size to pad/trim x last dimension to

    Returns:
        `x` padded to match `y`'s dimension.
    """
    inp_len = y
    output_len = x.shape[-1]
    return torch.nn.functional.pad(x, [0, inp_len - output_len])


def valid_length(input_size, kernel_size, stride=1, padding=0, dilation=1):
    """
    Return the nearest valid upper length to use with the model so that there is no time steps left over in a 1DConv.

    For all layers, size of the (input - kernel_size) % stride = 0.
    Here valid means that there is no left over frame neglected and discarded.

    Args:
        input_size: size of input
        kernel_size: size of kernel
        stride: stride
        padding: padding
        dilation: dilation

    Returns:
        valid length for convolution
    """
    length = math.ceil((input_size + 2*padding - dilation * (kernel_size - 1) - 1)/stride) + 1
    length = (length - 1) * stride - 2*padding + dilation * (kernel_size - 1) + 1

    return int(length)


def td_length_from_fd(fd_length: int, fft_size: int, fft_hop: int) -> int:
    """
    Return the length in time domain, given the length in frequency domain.

    Return the necessary length in the time domain of a signal to be transformed into
    a signal of length `fd_length` in time-frequency domain with the given STFT
    parameters `fft_size` and `fft_hop`. No padding is assumed.

    Args:
        fd_length: length in frequency domain
        fft_size: size of FFT
        fft_hop: hop length

    Returns:
        length in time domain
    """
    return (fd_length - 1) * fft_hop + fft_size
