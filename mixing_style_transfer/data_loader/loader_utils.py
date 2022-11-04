""" Utility file for loaders """

import numpy as np
import soundfile as sf
import wave



# Function to convert frame level audio into atomic time
def frames_to_time(total_length, sr=44100):
    in_time = total_length / sr
    hour = int(in_time / 3600)
    minute = int((in_time - hour*3600) / 60)
    second = int(in_time - hour*3600 - minute*60)
    return f"{hour:02d}:{minute:02d}:{second:02d}"


# Function to convert atomic labeled time into frames or seconds
def time_to_frames(input_time, to_frames=True, sr=44100):
    hour, minute, second = input_time.split(':')
    total_seconds = int(hour)*3600 + int(minute)*60 + int(second)
    return total_seconds*sr if to_frames else total_seconds


# Function to convert seconds to atomic labeled time
def sec_to_time(input_time):
    return frames_to_time(input_time, sr=1)


# Function to load total trainable raw audio lengths
def get_total_audio_length(audio_paths):
    total_length = 0
    for cur_audio_path in audio_paths:
        cur_wav = wave.open(cur_audio_path, 'r')
        total_length += cur_wav.getnframes()    # here, length = # of frames
    return total_length


# Function to load length of an input wav audio
def load_wav_length(audio_path):
    pt_wav = wave.open(audio_path, 'r')
    length = pt_wav.getnframes()
    return length


# Function to load only selected 16 bit, stereo wav audio segment from an input wav audio
def load_wav_segment(audio_path, start_point=None, duration=None, axis=1, sample_rate=44100):
    start_point = 0 if start_point==None else start_point
    duration = load_wav_length(audio_path) if duration==None else duration
    pt_wav = wave.open(audio_path, 'r')

    if pt_wav.getframerate()!=sample_rate:
        raise ValueError(f"ValueError: input audio's sample rate should be {sample_rate}")
    pt_wav.setpos(start_point)
    x = pt_wav.readframes(duration)
    if pt_wav.getsampwidth()==2:
        x = np.frombuffer(x, dtype=np.int16)
        X = x / float(2**15)    # needs to be 16 bit format 
    elif pt_wav.getsampwidth()==4:
        x = np.frombuffer(x, dtype=np.int32)
        X = x / float(2**31)    # needs to be 32 bit format 
    else:
        raise ValueError("ValueError: input audio's bit depth should be 16 or 32-bit")

    # exception for stereo channels 
    if pt_wav.getnchannels()==2:
        X_l = np.expand_dims(X[::2], axis=axis)
        X_r = np.expand_dims(X[1::2], axis=axis)
        X = np.concatenate((X_l, X_r), axis=axis)
    return X

