import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import scipy
import math
import librosa
import librosa.display
import fnmatch
import os
from functools import partial
import pyloudnorm
from scipy.signal import lfilter
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import paired_distances


import matplotlib.pyplot as plt 

def db(x):
    """Computes the decible energy of a signal"""
    return 20*np.log10(np.sqrt(np.mean(np.square(x))))

def melspectrogram(y, mirror_pad=False):
    """Compute melspectrogram feature extraction
    
    Keyword arguments:
    signal -- input audio as a signal in a numpy object
    inputnorm -- normalization of output
    mirror_pad -- pre and post-pend mirror signals 
    
    Returns freq x time
               
    
    Assumes the input sampling rate is 22050Hz
    """
    
    # Extract mel.
    fftsize = 1024
    window = 1024
    hop = 512
    melBin = 128
    sr = 22050

    # mirror pad signal
    # first embedding centered on time 0 
    # last embedding centered on end of signal
    if mirror_pad:
        y = np.insert(y, 0, y[0:int(half_frame_length_sec * sr)][::-1])
        y = np.insert(y, len(y), y[-int(half_frame_length_sec * sr):][::-1])
    
    S = librosa.core.stft(y,n_fft=fftsize,hop_length=hop,win_length=window)
    X = np.abs(S)
    mel_basis = librosa.filters.mel(sr,n_fft=fftsize,n_mels=melBin)
    mel_S = np.dot(mel_basis,X)

    # value log compression
    mel_S = np.log10(1+10*mel_S)
    mel_S = mel_S.astype(np.float32)
    

    return mel_S


def getFilesPath(directory, extension):
    
    n_path=[]
    for path, subdirs, files in os.walk(directory):
        for name in files:
            if fnmatch.fnmatch(name, extension):
                n_path.append(os.path.join(path,name))
    n_path.sort()
                
    return n_path



def getRandomTrim(x, length, pad=0, start=None):
    
    length = length+pad
    if x.shape[0] <= length:
        x_ = x
        while(x.shape[0] <= length):
            x_ = np.concatenate((x_,x_))
    else:
        if start is None:
            start = np.random.randint(0, x.shape[0]-length, size=None)
        end = length+start
        if end > x.shape[0]:
            x_ = x[start:]
            x_ = np.concatenate((x_, x[:length-x.shape[0]]))
        else:
            x_ = x[start:length+start]
            
    return x_[:length]

def fadeIn(x, length=128):
    
    w = scipy.signal.hann(length*2, sym=True)
    w1 = w[0:length]
    ones = np.ones(int(x.shape[0]-length))
    w = np.append(w1, ones)
    
    return x*w

def fadeOut(x, length=128):
    
    w = scipy.signal.hann(length*2, sym=True)
    w2 = w[length:length*2]
    ones = np.ones(int(x.shape[0]-length))
    w = np.append(ones, w2)
    
    return x*w


def plotTimeFreq(audio, sr, n_fft=512, hop_length=128, ylabels=None):
    
    n = len(audio)
#     plt.figure(figsize=(14, 4*n))
    colors = list(plt.cm.viridis(np.linspace(0,1,n)))
    
    X = []
    X_db = []
    maxs = np.zeros((n,))
    mins = np.zeros((n,))
    maxs_t = np.zeros((n,))
    for i, x in enumerate(audio):
        
        if x.ndim == 2 and x.shape[-1] == 2:
            x = librosa.core.to_mono(x.T)
        X_ = librosa.stft(x, n_fft=n_fft, hop_length=hop_length)
        X_db_ = librosa.amplitude_to_db(abs(X_))
        X.append(X_)
        X_db.append(X_db_)
        maxs[i] = np.max(X_db_)
        mins[i] = np.min(X_db_)
        maxs_t[i] = np.max(np.abs(x))
    vmax = np.max(maxs)
    vmin = np.min(mins)
    tmax = np.max(maxs_t)
    for i, x in enumerate(audio):
        
        if x.ndim == 2 and x.shape[-1] == 2:
            x = librosa.core.to_mono(x.T)
            
        plt.subplot(n, 2, 2*i+1)
        librosa.display.waveplot(x, sr=sr, color=colors[i])
        if ylabels:
            plt.ylabel(ylabels[i])
            
        plt.ylim(-tmax,tmax)
        plt.subplot(n, 2, 2*i+2)
        librosa.display.specshow(X_db[i], sr=sr, x_axis='time', y_axis='log',
                                 hop_length=hop_length, cmap='GnBu', vmax=vmax, vmin=vmin)
#         plt.colorbar(format='%+2.0f dB')








def slicing(x, win_length, hop_length, center = True, windowing = False, pad = 0):
    # Pad the time series so that frames are centered
    if center:
#         x = np.pad(x, int((win_length-hop_length+pad) // 2), mode='constant')
        x = np.pad(x, ((int((win_length-hop_length+pad)//2), int((win_length+hop_length+pad)//2)),), mode='constant')
        
    # Window the time series.
    y_frames = librosa.util.frame(x, frame_length=win_length, hop_length=hop_length)
    if windowing:
        window = scipy.signal.hann(win_length, sym=False)
    else:
        window = 1.0 
    f = []
    for i in range(len(y_frames.T)):
        f.append(y_frames.T[i]*window)
    return np.float32(np.asarray(f)) 


def overlap(x, x_len, win_length, hop_length, windowing = True, rate = 1): 
    x = x.reshape(x.shape[0],x.shape[1]).T
    if windowing:
        window = scipy.signal.hann(win_length, sym=False)
        rate = rate*hop_length/win_length
    else:
        window = 1
        rate = 1
    n_frames = x_len / hop_length
    expected_signal_len = int(win_length + hop_length * (n_frames))
    y = np.zeros(expected_signal_len)
    for i in range(int(n_frames)):
            sample = i * hop_length 
            w = x[:, i]
            y[sample:(sample + win_length)] = y[sample:(sample + win_length)] + w*window
    y = y[int(win_length // 2):-int(win_length // 2)]
    return np.float32(y*rate)   




            


def highpassFiltering(x_list, f0, sr):

    b1, a1 = scipy.signal.butter(4, f0/(sr/2),'highpass')
    x_f = []
    for x in x_list:
        x_f_ = scipy.signal.filtfilt(b1, a1, x).copy(order='F')
        x_f.append(x_f_)
    return x_f

def lineartodB(x):
    return 20*np.log10(x) 
def dBtoLinear(x):
    return np.power(10,x/20)

def lufs_normalize(x, sr, lufs, log=True):

    # measure the loudness first 
    meter = pyloudnorm.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(x+1e-10)
    if log:
        print("original loudness: ", loudness," max value: ", np.max(np.abs(x)))

    loudness_normalized_audio = pyloudnorm.normalize.loudness(x, loudness, lufs)
    
    maxabs_amp = np.maximum(1.0, 1e-6 + np.max(np.abs(loudness_normalized_audio)))
    loudness_normalized_audio /= maxabs_amp
    
    loudness = meter.integrated_loudness(loudness_normalized_audio)
    if log:
        print("new loudness: ", loudness," max value: ", np.max(np.abs(loudness_normalized_audio)))

    
    return loudness_normalized_audio

import soxbindings as sox

def lufs_normalize_compand(x, sr, lufs):
    
    tfm = sox.Transformer()
    tfm.compand(attack_time = 0.001,
                decay_time = 0.01,
                soft_knee_db = 1.0,
                tf_points = [(-70, -70), (-0.1, -20), (0, 0)])
    
    x = tfm.build_array(input_array=x, sample_rate_in=sr).astype(np.float32)

    # measure the loudness first 
    meter = pyloudnorm.Meter(sr) # create BS.1770 meter
    loudness = meter.integrated_loudness(x)
    print("original loudness: ", loudness," max value: ", np.max(np.abs(x)))

    loudness_normalized_audio = pyloudnorm.normalize.loudness(x, loudness, lufs)
    
    maxabs_amp = np.maximum(1.0, 1e-6 + np.max(np.abs(loudness_normalized_audio)))
    loudness_normalized_audio /= maxabs_amp
    
    loudness = meter.integrated_loudness(loudness_normalized_audio)
    print("new loudness: ", loudness," max value: ", np.max(np.abs(loudness_normalized_audio)))
    
    
   


    
    return loudness_normalized_audio


    


def getDistances(x,y):

    distances = {}
    distances['mae'] = mean_absolute_error(x, y)
    distances['mse'] = mean_squared_error(x, y)
    distances['euclidean'] = np.mean(paired_distances(x, y, metric='euclidean'))
    distances['manhattan'] = np.mean(paired_distances(x, y, metric='manhattan'))
    distances['cosine'] = np.mean(paired_distances(x, y, metric='cosine'))
   
    distances['mae'] = round(distances['mae'], 5)
    distances['mse'] = round(distances['mse'], 5)
    distances['euclidean'] = round(distances['euclidean'], 5)
    distances['manhattan'] = round(distances['manhattan'], 5)
    distances['cosine'] = round(distances['cosine'], 5)
    
    return distances

def getMFCC(x, sr, mels=128, mfcc=13, mean_norm=False):
    
    melspec = librosa.feature.melspectrogram(y=x, sr=sr, S=None,
                                     n_fft=1024, hop_length=256,
                                     n_mels=mels, power=2.0)
    melspec_dB = librosa.power_to_db(melspec, ref=np.max)
    mfcc = librosa.feature.mfcc(S=melspec_dB, sr=sr, n_mfcc=mfcc)
    if mean_norm:
        mfcc -= (np.mean(mfcc, axis=0))
    return mfcc

        
def getMSE_MFCC(y_true, y_pred, sr, mels=128, mfcc=13, mean_norm=False):
    
    ratio = np.mean(np.abs(y_true))/np.mean(np.abs(y_pred))
    y_pred =  ratio*y_pred
    
    y_mfcc = getMFCC(y_true, sr, mels=mels, mfcc=mfcc, mean_norm=mean_norm)
    z_mfcc = getMFCC(y_pred, sr, mels=mels, mfcc=mfcc, mean_norm=mean_norm)
    
    return getDistances(y_mfcc[:,:], z_mfcc[:,:]) 