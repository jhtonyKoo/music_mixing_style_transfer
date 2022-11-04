import os

import sys
import time
import numpy as np
import scipy
import librosa
import pyloudnorm as pyln

sys.setrecursionlimit(int(1e6))

import sklearn

currentdir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(currentdir)
from common_miscellaneous import compute_stft, compute_istft
from common_audioeffects import Panner, Compressor, AugmentationChain, ConvolutionalReverb, Equaliser, AlgorithmicReverb
import fx_utils

import soundfile as sf
import aubio

import time

import warnings

# Functions

def print_dict(dict_):
    for i in dict_:
        print(i)
        for j in dict_[i]:
            print('\t', j)

def amp_to_db(x):
    return 20*np.log10(x + 1e-30)

def db_to_amp(x):
    return 10**(x/20)

def get_running_stats(x, features, N=20):
    mean = []
    std = []
    for i in range(len(features)):
        mean_, std_ = running_mean_std(x[:,i], N)
        mean.append(mean_)
        std.append(std_)
    mean = np.asarray(mean)
    std = np.asarray(std)
    
    return mean, std

def running_mean_std(x, N):
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        cumsum = np.cumsum(np.insert(x, 0, 0)) 
        cumsum2 = np.cumsum(np.insert(x**2, 0, 0)) 
        mean = (cumsum[N:] - cumsum[:-N]) / float(N)

        std = np.sqrt(((cumsum2[N:] - cumsum2[:-N]) / N) - (mean * mean))

    return mean, std

def get_eq_matching(audio_t, ref_spec, sr=44100, n_fft=65536, hop_length=16384,
                    min_db=-50, ntaps=101, lufs=-30):
    
    audio_t = np.copy(audio_t)
    max_db = amp_to_db(np.max(np.abs(audio_t)))
    if max_db > min_db:

        audio_t = fx_utils.lufs_normalize(audio_t, sr, lufs, log=False)
        audio_D = compute_stft(np.expand_dims(audio_t, 1),
                         hop_length,
                         n_fft,
                         np.sqrt(np.hanning(n_fft+1)[:-1]))
        audio_D = np.abs(audio_D)
        audio_D_avg = np.mean(audio_D, axis=0)[0]

        m = ref_spec.shape[0]

        Ts = 1.0/sr # sampling interval
        n = m # length of the signal
        kk = np.arange(n)
        T = n/sr
        frq = kk/T # two sides frequency range
        frq /=2

        diff_eq = amp_to_db(ref_spec)-amp_to_db(audio_D_avg)
        diff_eq = db_to_amp(diff_eq)
        diff_eq = np.sqrt(diff_eq)

        diff_filter = scipy.signal.firwin2(ntaps,
                                           frq/np.max(frq),
                                           diff_eq,
                                           nfreqs=None, window='hamming',
                                           nyq=None, antisymmetric=False)


        output = scipy.signal.filtfilt(diff_filter, 1, audio_t,
                                       axis=-1, padtype='odd', padlen=None,
                                       method='pad', irlen=None)

    else:
        output = audio_t
        
    return output

def get_SPS(x, n_fft=2048, hop_length=1024, smooth=False, frames=False):
    
    x = np.copy(x)
    eps = 1e-20
        
    audio_D = compute_stft(x,
                 hop_length,
                 n_fft,
                 np.sqrt(np.hanning(n_fft+1)[:-1]))
    
    audio_D_l = np.abs(audio_D[:, 0, :] + eps)
    audio_D_r = np.abs(audio_D[:, 1, :] + eps)
    
    phi = 2 * (np.abs(audio_D_l*np.conj(audio_D_r)))/(np.abs(audio_D_l)**2+np.abs(audio_D_r)**2)
    
    phi_l = np.abs(audio_D_l*np.conj(audio_D_r))/(np.abs(audio_D_l)**2)
    phi_r = np.abs(audio_D_r*np.conj(audio_D_l))/(np.abs(audio_D_r)**2)
    delta = phi_l - phi_r
    delta_ = np.sign(delta)
    SPS = (1-phi)*delta_
    
    phi_mean = np.mean(phi, axis=0)
    if smooth:
        phi_mean = scipy.signal.savgol_filter(phi_mean, 501, 1, mode='mirror')
    
    SPS_mean = np.mean(SPS, axis=0)
    if smooth:
        SPS_mean = scipy.signal.savgol_filter(SPS_mean, 501, 1, mode='mirror')
        

    return SPS_mean, phi_mean, SPS, phi


def get_mean_side(sps, freqs=[50,2500], sr=44100, n_fft=2048):
    
    sign = np.sign(sps+ 1e-10)
    
    idx1 = freqs[0]
    idx2 = freqs[1]

    f1 = int(np.floor(idx1*n_fft/sr))
    f2 = int(np.floor(idx2*n_fft/sr))

    sign_mean = np.mean(sign[f1:f2])/np.abs(np.mean(sign[f1:f2]))
    sign_mean
    
    return sign_mean

def get_panning_param_values(phi, side):

    p = np.zeros_like(phi)
    
    g = (np.clip(phi+1e-30, 0, 1))/2

    for i, g_ in enumerate(g):

        if side > 0:
            p[i] = 1 - g_

        elif side < 0:
            p[i] = g_

        else:
            p[i] = 0.5

    g_l = 1-p
    g_r = p

    return p, [g_l, g_r]

def get_panning_matching(audio, ref_phi,
                         sr=44100, n_fft=2048, hop_length=1024,
                         min_db_f=-10, max_freq_pan=16000, frames=True):
    
    eps = 1e-20
    window = np.sqrt(np.hanning(n_fft+1)[:-1])
    audio = np.copy(audio)
    audio_t = np.pad(audio, ((n_fft, n_fft), (0, 0)), mode='constant')

    sps_mean_, phi_mean_, _, _ = get_SPS(audio_t, n_fft=n_fft, hop_length=hop_length, smooth=True)

    side = get_mean_side(sps_mean_, sr=sr, n_fft=n_fft)

    if side > 0:
        alpha = 0.7
    else:
        alpha = 0.3
        
    processor = Panner()
    processor.parameters.pan.value = alpha
    processor.parameters.pan_law.value = 'linear'
    processor.update()
    audio_t_ = processor.process(audio_t)

    sps_mean_, phi_mean, sps_frames, phi_frames = get_SPS(audio_t_, n_fft=n_fft,
                                                          hop_length=hop_length,
                                                          smooth=True, frames=frames)

    if frames:

        p_i_ = []
        g_i_ = []
        p_ref = []
        g_ref = []
        for i in range(len(sps_frames)):
            sps_ = sps_frames[i]
            phi_ = phi_frames[i]
            p_, g_ = get_panning_param_values(phi_, side)
            p_i_.append(p_)
            g_i_.append(g_)
            p_, g_ = get_panning_param_values(ref_phi, side)
            p_ref.append(p_)
            g_ref.append(g_) 
        ratio = (np.asarray(g_ref)/(np.asarray(g_i_)+eps))
        g_l = ratio[:,0,:]
        g_r = ratio[:,1,:]

     
    else:
        p, g = get_panning_param_values(ref_phi, side)
        p_i, g_i = get_panning_param_values(phi_mean, side)
        ratio = (np.asarray(g)/np.asarray(g_i))
        g_l = ratio[0]
        g_r = ratio[1]
        
    audio_new_D = compute_stft(audio_t_,
                               hop_length,
                               n_fft,
                               window)

    audio_new_D_mono = audio_new_D.copy()
    audio_new_D_mono = audio_new_D_mono[:, 0, :] + audio_new_D_mono[:, 1, :]
    audio_new_D_mono = np.abs(audio_new_D_mono)
    
    audio_new_D_phase = np.angle(audio_new_D)
    audio_new_D = np.abs(audio_new_D)

    audio_new_D_l = audio_new_D[:, 0, :]
    audio_new_D_r = audio_new_D[:, 1, :]
    
    if frames:
        for i, frame in enumerate(audio_new_D_mono):            
            max_db = amp_to_db(np.max(np.abs(frame)))         
            if max_db < min_db_f:
                g_r[i] = np.ones_like(frame)
                g_l[i] = np.ones_like(frame)
                
        idx1 = max_freq_pan
        f1 = int(np.floor(idx1*n_fft/sr))
        ones = np.ones_like(g_l)
        g_l[f1:] = ones[f1:]
        g_r[f1:] = ones[f1:]

    audio_new_D_l = audio_new_D_l*g_l
    audio_new_D_r = audio_new_D_r*g_r

    audio_new_D_l = np.expand_dims(audio_new_D_l, 0)
    audio_new_D_r = np.expand_dims(audio_new_D_r, 0)

    audio_new_D_ = np.concatenate((audio_new_D_l,audio_new_D_r))

    audio_new_D_ = np.moveaxis(audio_new_D_, 0, 1)

    audio_new_D_ = audio_new_D_ * (np.cos(audio_new_D_phase) + np.sin(audio_new_D_phase)*1j)

    audio_new_t = compute_istft(audio_new_D_,
                                hop_length,
                                window)

    audio_new_t = audio_new_t[n_fft:n_fft+audio.shape[0]]

    return audio_new_t



def get_mean_peak(audio, sr=44100, true_peak=False, n_mels=128, percentile=75):
    
#     Returns mean peak value in dB after the 1Q is removed.
#     Input should be in the shape samples x channel
    
    audio_ = audio
    window_size = 2**10 # FFT size
    hop_size = window_size
    
    peak = []
    std = []
    for ch in range(audio_.shape[-1]):
        x = np.ascontiguousarray(audio_[:, ch])

        if true_peak:
            x = librosa.resample(x, sr, 4*sr)
            sr = 4*sr
            window_size = 4*window_size
            hop_size = 4*hop_size
            
        onset_func = aubio.onset('hfc', buf_size=window_size, hop_size=hop_size, samplerate=sr)

        frames = np.float32(librosa.util.frame(x, frame_length=window_size, hop_length=hop_size))
        
        onset_times = []
        for frame in frames.T:
            
            if onset_func(frame):
                
                onset_time = onset_func.get_last()
                onset_times.append(onset_time) 
                
        samples=[]
        if onset_times:
            for i, p in enumerate(onset_times[:-1]):
                samples.append(onset_times[i]+np.argmax(np.abs(x[onset_times[i]:onset_times[i+1]])))
            samples.append(onset_times[-1]+np.argmax(np.abs(x[onset_times[-1]:])))

        p_value = []
        for p in samples:
            p_ = amp_to_db(np.abs(x[p]))
            p_value.append(p_)
        p_value_=[]
        for p in p_value:
            if p > np.percentile(p_value, percentile):
                p_value_.append(p)
        if p_value_:
            peak.append(np.mean(p_value_))
            std.append(np.std(p_value_))
        elif p_value:
            peak.append(np.mean(p_value))
            std.append(np.std(p_value))
        else:
            return None
    return [np.mean(peak), np.mean(std)]

def compress(processor, audio, sr, th, ratio, attack, release):
    
    eps = 1e-20  
    x = audio

    processor.parameters.threshold.value = th
    processor.parameters.ratio.value = ratio
    processor.parameters.attack_time.value = attack
    processor.parameters.release_time.value = release
    processor.update()
    output = processor.process(x)
    
    if np.max(np.abs(output)) >= 1.0:
        output = np.clip(output, -1.0, 1.0)
    
    return output

def get_comp_matching(audio,
                      ref_peak, ref_std,
                      ratio, attack, release, sr=44100,
                      min_db=-50, comp_peak_norm=-10.0,
                      min_th=-40, max_ratio=20, n_mels=128,
                      true_peak=False, percentile=75, expander=True):
    
    x = audio.copy()
    
    if x.ndim < 2:
        x = np.expand_dims(x, 1)
        
    max_db = amp_to_db(np.max(np.abs(x)))
    if max_db > min_db:
        
        x = pyln.normalize.peak(x, comp_peak_norm)

        peak, std = get_mean_peak(x, sr,
                                  n_mels=n_mels,
                                  true_peak=true_peak,
                                  percentile=percentile)

        if peak > (ref_peak - ref_std) and peak < (ref_peak + ref_std):
            return x

    #     DownwardCompress
        elif peak > (ref_peak - ref_std): 
            processor = Compressor(sample_rate=sr)
            # print('compress')
            ratios = np.linspace(ratio, max_ratio, max_ratio-ratio+1)
            ths = np.linspace(-1-9, min_th, 2*np.abs(min_th)-1-18)
            for rt in ratios:
                for th in ths:
                    y = compress(processor, x, sr, th, rt, attack, release)
                    peak, std = get_mean_peak(y, sr,
                                              n_mels=n_mels,
                                              true_peak=true_peak,
                                              percentile=percentile)
                    if peak < (ref_peak + ref_std):
                        break
                else:
                    continue
                break

            return y

    #      Upward Expand        
        elif peak < (ref_peak + ref_std):
            
            if expander:
                processor = Compressor(sample_rate=sr)
                ratios = np.linspace(ratio, max_ratio, max_ratio-ratio+1)
                ths = np.linspace(-1, min_th, 2*np.abs(min_th)-1)[::-1]

                for rt in ratios:
                    for th in ths:
                        y = compress(processor, x, sr, th, 1/rt, attack, release)
                        peak, std = get_mean_peak(y, sr,
                                                  n_mels=n_mels,
                                                  true_peak=true_peak,
                                                  percentile=percentile)
                        if peak > (ref_peak - ref_std):
                            break
                    else:
                        continue
                    break

                return y
            
            else:
                return x
    else:
        return x
    
    
    
# REVERB


def get_reverb_send(audio, eq_parameters, rv_parameters, impulse_responses=None,
                    eq_prob=1.0, rv_prob=1.0, parallel=True, shuffle=False, sr=44100, bands=['low_shelf', 'high_shelf']):
    
    x = audio.copy()
    
    if x.ndim < 2:
        x = np.expand_dims(x, 1)
        
    channels = x.shape[-1]
    eq_gain = eq_parameters.low_shelf_gain.value
        
        
    eq = Equaliser(n_channels=channels,
               sample_rate=sr,
               gain_range=(eq_gain, eq_gain),
               bands=bands,
               hard_clip=False,
                name='Equaliser', parameters=eq_parameters)
    eq.randomize()

    if impulse_responses:
    
        reverb = ConvolutionalReverb(impulse_responses=impulse_responses,
                                 sample_rate=sr, 
                                 parameters=rv_parameters)
        
    else:
        
        reverb = AlgorithmicReverb(sample_rate=sr, 
                             parameters=rv_parameters)
    
    reverb.randomize()
    
    fxchain = AugmentationChain([
                             (eq, rv_prob, False),
                            (reverb, eq_prob, False)
                             ],
                            shuffle=shuffle, parallel=parallel)
    
    output = fxchain(x)
    
    return output


    
# FUNCTIONS TO COMPUTE FEATURES

def compute_loudness_features(args_):
    
    audio_out_ = args_[0]
    audio_tar_ = args_[1]
    idx = args_[2]
    sr = args_[3]
   
    loudness_ = {key:[] for key in ['d_lufs', 'd_peak',]}

    peak_tar = np.max(np.abs(audio_tar_))
    peak_tar_db = 20.0 * np.log10(peak_tar)

    peak_out = np.max(np.abs(audio_out_))
    peak_out_db = 20.0 * np.log10(peak_out)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        meter = pyln.Meter(sr) # create BS.1770 meter
        loudness_tar = meter.integrated_loudness(audio_tar_)
        loudness_out = meter.integrated_loudness(audio_out_)
    
    loudness_['d_lufs'].append(sklearn.metrics.mean_absolute_percentage_error([loudness_tar], [loudness_out]))
    loudness_['d_peak'].append(sklearn.metrics.mean_absolute_percentage_error([peak_tar_db], [peak_out_db]))

    return loudness_

def compute_spectral_features(args_):
    
    audio_out_ = args_[0]
    audio_tar_ = args_[1]
    idx = args_[2]
    sr = args_[3]
    fft_size = args_[4]
    hop_length = args_[5]
    channels = args_[6]
    
    audio_out_ = pyln.normalize.peak(audio_out_, -1.0)
    audio_tar_ = pyln.normalize.peak(audio_tar_, -1.0)
    
    spec_out_ = compute_stft(audio_out_,
                         hop_length,
                         fft_size,
                         np.sqrt(np.hanning(fft_size+1)[:-1]))
    spec_out_ = np.transpose(spec_out_, axes=[1, -1, 0])
    spec_out_ = np.abs(spec_out_)
    
    spec_tar_ = compute_stft(audio_tar_,
                             hop_length,
                             fft_size,
                             np.sqrt(np.hanning(fft_size+1)[:-1]))
    spec_tar_ = np.transpose(spec_tar_, axes=[1, -1, 0])
    spec_tar_ = np.abs(spec_tar_)
   
    spectral_ = {key:[] for key in ['centroid_mean',
                                    'bandwidth_mean',
                                    'contrast_l_mean',
                                    'contrast_m_mean',
                                    'contrast_h_mean',
                                    'rolloff_mean',
                                    'flatness_mean',
                                    'mape_mean',
                                   ]}
    
    centroid_mean_ = []
    centroid_std_ = []
    bandwidth_mean_ = []
    bandwidth_std_ = []
    contrast_l_mean_ = []
    contrast_l_std_ = []
    contrast_m_mean_ = []
    contrast_m_std_ = []
    contrast_h_mean_ = []
    contrast_h_std_ = []
    rolloff_mean_ = []
    rolloff_std_ = []
    flatness_mean_ = []

    for ch in range(channels):
        tar = spec_tar_[ch]
        out = spec_out_[ch]

        tar_sc = librosa.feature.spectral_centroid(y=None, sr=sr, S=tar,
                             n_fft=fft_size, hop_length=hop_length)

        out_sc = librosa.feature.spectral_centroid(y=None, sr=sr, S=out,
                             n_fft=fft_size, hop_length=hop_length)

        tar_bw = librosa.feature.spectral_bandwidth(y=None, sr=sr, S=tar,
                                                    n_fft=fft_size, hop_length=hop_length, 
                                                    centroid=tar_sc, norm=True, p=2)

        out_bw = librosa.feature.spectral_bandwidth(y=None, sr=sr, S=out,
                                                    n_fft=fft_size, hop_length=hop_length, 
                                                    centroid=out_sc, norm=True, p=2)
        # l = 0-250, m = 1-2-3 = 250 - 2000, h = 2000 - SR/2
        tar_ct = librosa.feature.spectral_contrast(y=None, sr=sr, S=tar,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   fmin=250.0, n_bands=4, quantile=0.02, linear=False)

        out_ct = librosa.feature.spectral_contrast(y=None, sr=sr, S=out,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   fmin=250.0, n_bands=4, quantile=0.02, linear=False)

        tar_ro = librosa.feature.spectral_rolloff(y=None, sr=sr, S=tar,
                                                  n_fft=fft_size, hop_length=hop_length, 
                                                  roll_percent=0.85)

        out_ro = librosa.feature.spectral_rolloff(y=None, sr=sr, S=out,
                                                  n_fft=fft_size, hop_length=hop_length, 
                                                  roll_percent=0.85)
        
        tar_ft = librosa.feature.spectral_flatness(y=None, S=tar,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   amin=1e-10, power=2.0)

        out_ft = librosa.feature.spectral_flatness(y=None, S=out,
                                                   n_fft=fft_size, hop_length=hop_length, 
                                                   amin=1e-10, power=2.0)

        
        eps = 1e-0
        N = 40
        mean_sc_tar, std_sc_tar = get_running_stats(tar_sc.T+eps, [0], N=N)
        mean_sc_out, std_sc_out = get_running_stats(out_sc.T+eps, [0], N=N)
        
        assert np.isnan(mean_sc_tar).any() == False, f'NAN values mean_sc_tar {idx}'
        assert np.isnan(mean_sc_out).any() == False, f'NAN values mean_sc_out {idx}'
        

        mean_bw_tar, std_bw_tar = get_running_stats(tar_bw.T+eps, [0], N=N)
        mean_bw_out, std_bw_out = get_running_stats(out_bw.T+eps, [0], N=N)
        
        assert np.isnan(mean_bw_tar).any() == False, f'NAN values tar mean bw {idx}'
        assert np.isnan(mean_bw_out).any() == False, f'NAN values out mean bw {idx}'
        
        mean_ct_tar, std_ct_tar = get_running_stats(tar_ct.T, list(range(tar_ct.shape[0])), N=N)
        mean_ct_out, std_ct_out = get_running_stats(out_ct.T, list(range(out_ct.shape[0])), N=N)
        
        assert np.isnan(mean_ct_tar).any() == False, f'NAN values tar mean ct {idx}'
        assert np.isnan(mean_ct_out).any() == False, f'NAN values out mean ct {idx}'
        
        mean_ro_tar, std_ro_tar = get_running_stats(tar_ro.T+eps, [0], N=N)
        mean_ro_out, std_ro_out = get_running_stats(out_ro.T+eps, [0], N=N)
        
        assert np.isnan(mean_ro_tar).any() == False, f'NAN values tar mean ro {idx}'
        assert np.isnan(mean_ro_out).any() == False, f'NAN values out mean ro {idx}'
        
        mean_ft_tar, std_ft_tar = get_running_stats(tar_ft.T, [0], N=800) # gives very high numbers due to N (80) value
        mean_ft_out, std_ft_out = get_running_stats(out_ft.T, [0], N=800)
        
        mape_mean_sc = sklearn.metrics.mean_absolute_percentage_error(mean_sc_tar[0], mean_sc_out[0])
        
        mape_mean_bw = sklearn.metrics.mean_absolute_percentage_error(mean_bw_tar[0], mean_bw_out[0])
        
        mape_mean_ct_l = sklearn.metrics.mean_absolute_percentage_error(mean_ct_tar[0], mean_ct_out[0])
        
        mape_mean_ct_m = sklearn.metrics.mean_absolute_percentage_error(np.mean(mean_ct_tar[1:4], axis=0),
                                                                        np.mean(mean_ct_out[1:4], axis=0))

        mape_mean_ct_h = sklearn.metrics.mean_absolute_percentage_error(mean_ct_tar[-1], mean_ct_out[-1])
   
        mape_mean_ro = sklearn.metrics.mean_absolute_percentage_error(mean_ro_tar[0], mean_ro_out[0])
        
        mape_mean_ft = sklearn.metrics.mean_absolute_percentage_error(mean_ft_tar[0], mean_ft_out[0])
        
        centroid_mean_.append(mape_mean_sc)
        bandwidth_mean_.append(mape_mean_bw)
        contrast_l_mean_.append(mape_mean_ct_l)
        contrast_m_mean_.append(mape_mean_ct_m)
        contrast_h_mean_.append(mape_mean_ct_h)
        rolloff_mean_.append(mape_mean_ro)
        flatness_mean_.append(mape_mean_ft)

    spectral_['centroid_mean'].append(np.mean(centroid_mean_))
    
    spectral_['bandwidth_mean'].append(np.mean(bandwidth_mean_))
    
    spectral_['contrast_l_mean'].append(np.mean(contrast_l_mean_))
    
    spectral_['contrast_m_mean'].append(np.mean(contrast_m_mean_))
    
    spectral_['contrast_h_mean'].append(np.mean(contrast_h_mean_))
    
    spectral_['rolloff_mean'].append(np.mean(rolloff_mean_))
    
    spectral_['flatness_mean'].append(np.mean(flatness_mean_))
    
    spectral_['mape_mean'].append(np.mean([np.mean(centroid_mean_),
                                      np.mean(bandwidth_mean_),
                                      np.mean(contrast_l_mean_),
                                      np.mean(contrast_m_mean_),
                                      np.mean(contrast_h_mean_),
                                      np.mean(rolloff_mean_),
                                      np.mean(flatness_mean_),
                                     ]))
    
    return spectral_

# PANNING 
def get_panning_rms_frame(sps_frame, freqs=[0,22050], sr=44100, n_fft=2048):
    
    idx1 = freqs[0]
    idx2 = freqs[1]

    f1 = int(np.floor(idx1*n_fft/sr))
    f2 = int(np.floor(idx2*n_fft/sr))
    
    p_rms = np.sqrt((1/(f2-f1)) * np.sum(sps_frame[f1:f2]**2))
    
    return p_rms
def get_panning_rms(sps, freqs=[[0, 22050]], sr=44100, n_fft=2048):
    
    p_rms = []
    for frame in sps:
        p_rms_ = []
        for f in freqs:
            rms = get_panning_rms_frame(frame, freqs=f, sr=sr, n_fft=n_fft)
            p_rms_.append(rms)
        p_rms.append(p_rms_)
    
    return np.asarray(p_rms)



def compute_panning_features(args_):
    
    audio_out_ = args_[0]
    audio_tar_ = args_[1]
    idx = args_[2]
    sr = args_[3]
    fft_size = args_[4]
    hop_length = args_[5]
    
    audio_out_ = pyln.normalize.peak(audio_out_, -1.0)
    audio_tar_ = pyln.normalize.peak(audio_tar_, -1.0)
    
    panning_ = {}
                               
    freqs=[[0, sr//2], [0, 250], [250, 2500], [2500, sr//2]]  
    
    _, _, sps_frames_tar, _ = get_SPS(audio_tar_, n_fft=fft_size,
                                  hop_length=hop_length,
                                  smooth=True, frames=True)
    
    _, _, sps_frames_out, _ = get_SPS(audio_out_, n_fft=fft_size,
                                      hop_length=hop_length,
                                      smooth=True, frames=True)


    p_rms_tar = get_panning_rms(sps_frames_tar,
                    freqs=freqs,
                    sr=sr,
                    n_fft=fft_size)
    
    p_rms_out = get_panning_rms(sps_frames_out,
                    freqs=freqs,
                    sr=sr,
                    n_fft=fft_size)
    
    # to avoid num instability, deletes frames with zero rms from target
    if np.min(p_rms_tar) == 0.0:
        id_zeros = np.where(p_rms_tar.T[0] == 0)
        p_rms_tar_ = []
        p_rms_out_ = []
        for i in range(len(freqs)):
            temp_tar = np.delete(p_rms_tar.T[i], id_zeros)
            temp_out = np.delete(p_rms_out.T[i], id_zeros)
            p_rms_tar_.append(temp_tar)
            p_rms_out_.append(temp_out)
        p_rms_tar_ = np.asarray(p_rms_tar_)
        p_rms_tar = p_rms_tar_.T
        p_rms_out_ = np.asarray(p_rms_out_)
        p_rms_out = p_rms_out_.T
    
    N = 40 
    
    mean_tar, std_tar = get_running_stats(p_rms_tar, freqs, N=N)
    mean_out, std_out = get_running_stats(p_rms_out, freqs, N=N)
    
    panning_['P_t_mean'] = [sklearn.metrics.mean_absolute_percentage_error(mean_tar[0], mean_out[0])]
    panning_['P_l_mean'] = [sklearn.metrics.mean_absolute_percentage_error(mean_tar[1], mean_out[1])]
    panning_['P_m_mean'] = [sklearn.metrics.mean_absolute_percentage_error(mean_tar[2], mean_out[2])]
    panning_['P_h_mean'] = [sklearn.metrics.mean_absolute_percentage_error(mean_tar[3], mean_out[3])]

    panning_['mape_mean'] = [np.mean([panning_['P_t_mean'],
                                      panning_['P_l_mean'],
                                      panning_['P_m_mean'],
                                      panning_['P_h_mean'],
                                     ])]
    
    return panning_

# DYNAMIC

def get_rms_dynamic_crest(x, frame_length, hop_length):
    
    rms = []
    dynamic_spread = []
    crest = []
    for ch in range(x.shape[-1]):
        frames = librosa.util.frame(x[:, ch], frame_length=frame_length, hop_length=hop_length)
        rms_ = []
        dynamic_spread_ = []
        crest_ = []
        for i in frames.T:
            x_rms = amp_to_db(np.sqrt(np.sum(i**2)/frame_length))   
            x_d = np.sum(amp_to_db(np.abs(i)) - x_rms)/frame_length
            x_c = amp_to_db(np.max(np.abs(i)))/x_rms
            
            rms_.append(x_rms)
            dynamic_spread_.append(x_d)
            crest_.append(x_c)
        rms.append(rms_)
        dynamic_spread.append(dynamic_spread_)
        crest.append(crest_)
        
    rms = np.asarray(rms)
    dynamic_spread = np.asarray(dynamic_spread)
    crest = np.asarray(crest)  
    
    rms = np.mean(rms, axis=0)
    dynamic_spread = np.mean(dynamic_spread, axis=0)
    crest = np.mean(crest, axis=0)
    
    rms = np.expand_dims(rms, axis=0)
    dynamic_spread = np.expand_dims(dynamic_spread, axis=0)
    crest = np.expand_dims(crest, axis=0)
    
    return rms, dynamic_spread, crest

def lowpassFiltering(x, f0, sr):

    b1, a1 = scipy.signal.butter(4, f0/(sr/2),'lowpass')
    x_f = []
    for ch in range(x.shape[-1]):
        x_f_ = scipy.signal.filtfilt(b1, a1, x[:, ch]).copy(order='F')
        x_f.append(x_f_)
    return np.asarray(x_f).T  


def get_low_freq_weighting(x, sr, n_fft, hop_length, f0 = 1000):
    
    x_low = lowpassFiltering(x, f0, sr)
    
    X_low = compute_stft(x_low,
                         hop_length,
                         n_fft,
                         np.sqrt(np.hanning(n_fft+1)[:-1]))
    X_low = np.transpose(X_low, axes=[1, -1, 0])
    X_low = np.abs(X_low)

    X = compute_stft(x,
                         hop_length,
                         n_fft,
                         np.sqrt(np.hanning(n_fft+1)[:-1]))
    X = np.transpose(X, axes=[1, -1, 0])
    X = np.abs(X)

    eps = 1e-5
    ratio = (X_low)/(X+eps)
    ratio = np.sum(ratio, axis = 1)
    ratio = np.mean(ratio, axis = 0)
    
    return np.expand_dims(ratio, axis=0)

def compute_dynamic_features(args_):
    
    audio_out_ = args_[0]
    audio_tar_ = args_[1]
    idx = args_[2]
    sr = args_[3]
    fft_size = args_[4]
    hop_length = args_[5]
    
    audio_out_ = pyln.normalize.peak(audio_out_, -1.0)
    audio_tar_ = pyln.normalize.peak(audio_tar_, -1.0)
    
    dynamic_ = {}
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
    
        rms_tar, dyn_tar, crest_tar = get_rms_dynamic_crest(audio_tar_, fft_size, hop_length)
        rms_out, dyn_out, crest_out = get_rms_dynamic_crest(audio_out_, fft_size, hop_length)

        low_ratio_tar = get_low_freq_weighting(audio_tar_, sr, fft_size, hop_length, f0=1000)
        
        low_ratio_out = get_low_freq_weighting(audio_out_, sr, fft_size, hop_length, f0=1000)
        
    N = 40
    
    eps = 1e-10

    rms_tar = (-1*rms_tar) + 1.0
    rms_out = (-1*rms_out) + 1.0
    dyn_tar = (-1*dyn_tar) + 1.0
    dyn_out = (-1*dyn_out) + 1.0
    
    mean_rms_tar, std_rms_tar = get_running_stats(rms_tar.T, [0], N=N)
    mean_rms_out, std_rms_out = get_running_stats(rms_out.T, [0], N=N)
    
    mean_dyn_tar, std_dyn_tar = get_running_stats(dyn_tar.T, [0], N=N)
    mean_dyn_out, std_dyn_out = get_running_stats(dyn_out.T, [0], N=N)
    
    mean_crest_tar, std_crest_tar = get_running_stats(crest_tar.T, [0], N=N)
    mean_crest_out, std_crest_out = get_running_stats(crest_out.T, [0], N=N)
    
    mean_low_ratio_tar, std_low_ratio_tar = get_running_stats(low_ratio_tar.T, [0], N=N)
    mean_low_ratio_out, std_low_ratio_out = get_running_stats(low_ratio_out.T, [0], N=N)
        
    dynamic_['rms_mean'] = [sklearn.metrics.mean_absolute_percentage_error(mean_rms_tar, mean_rms_out)]
    dynamic_['dyn_mean'] = [sklearn.metrics.mean_absolute_percentage_error(mean_dyn_tar, mean_dyn_out)]
    dynamic_['crest_mean'] = [sklearn.metrics.mean_absolute_percentage_error(mean_crest_tar, mean_crest_out)]
    
    dynamic_['l_ratio_mean_mape'] = [sklearn.metrics.mean_absolute_percentage_error(mean_low_ratio_tar, mean_low_ratio_out)]
    dynamic_['l_ratio_mean_l2'] = [sklearn.metrics.mean_squared_error(mean_low_ratio_tar, mean_low_ratio_out)]

    dynamic_['mape_mean'] = [np.mean([dynamic_['rms_mean'],
                                      dynamic_['dyn_mean'],
                                      dynamic_['crest_mean'],
                                     ])]
    
    return dynamic_
                                                                      