"""
    @Author: Meet Sable
    @ID: 201901442
"""

import os
import numpy as np
import librosa
from librosa.util import normalize
from scipy.signal import fftconvolve
import soundfile as sf
import random
from tqdm import tqdm

from scipy.signal import stft, istft
import tensorflow as tf
import torch
from torch import nn

NFFT=512

def wave_to_frames(wave, frame_len, hop_len):
    """
        Split audio waves to frames

        Parameters
        ----------
        wave : source audio
        frame_len : length of each frame
        hop_len : length of the hops
    """
    return librosa.util.frame(wave, frame_length=frame_len, hop_length=hop_len, axis=0)

def frame_to_wave(frames, hop_len):
    """
        Stacks the frame into complete audio

        Parameters
        ----------
        frames : Array of audio frames
        hop_len : Length of the hop
    """
    return tf.signal.overlap_and_add(frames, frame_step=hop_len).numpy()


def wave_to_stft(wave, sr, win_len, hop_len, n_fft=NFFT):
    """
        Convert signal wave to stft magnitude spectrum in db

        Parameters
        ----------
        wave : signal array
        sr : samplerate
        win_len : window length
        hop_len : offset between two windows
    """
    overlap_len = win_len - hop_len
    f, t, spec = stft(wave, fs=sr, nperseg=win_len, noverlap=overlap_len, nfft=n_fft)
    # spec = librosa.amplitude_to_db(spec, ref=np.max)
    return spec

def stft_to_wave(spec, sr, win_len, hop_len, n_fft=NFFT):
    """
        Convert stft to time domain signal

        Parameters
        ----------
        spec : STFT spectrum
        sr : samplerate
        win_len : window length
        hop_len : hop length
    """
    overlap_len = win_len - hop_len
    # spec = librosa.db_to_amplitude(spec, ref=np.max)
    t, sig = istft(spec, fs=sr, nperseg=win_len, noverlap=overlap_len, nfft=n_fft)
    # sig = librosa.util.normalize(sig)
    return sig

def wave_to_stft_frames(wave, sr, frame_len, frame_hop_len, win_len, hop_len, n_fft=NFFT):
    """
        Convert speech signal to stft frames of fixed length

        Parameters
        ----------
        wave : speech signal
        sr : samplerate
        frame_len : length of 1 frame
        frame_hop_len : hop length between two frames
        win_len : window length for stft computation
        hop_len : hop length for stft computation
    """
    frames = wave_to_frames(wave, frame_len=frame_len, hop_len=frame_hop_len)
    stft_frames = [ _ for _ in range(len(frames))]
    for i, frame in enumerate(frames, 0):
        stft_frames[i] = wave_to_stft(frame, sr=sr, win_len=win_len, hop_len=hop_len, n_fft=n_fft)
    return stft_frames

def stft_frames_to_wave(stft_frames, sr, frame_hop_len, win_len, hop_len, n_fft=NFFT):
    """
        Convert stft frames to complete signal

        Parameters
        ----------
        stft_frames : stft frames of signal
        sr : samplerate
        frame_hop_len : length of hop between frames
        win_len : window length
        hop_len : hop length
    """
    frames = [ _ for _ in range(len(stft_frames))]
    for i, frame in enumerate(stft_frames, 0):
        frames[i] = stft_to_wave(frame, sr=sr, win_len=win_len, hop_len=hop_len, n_fft=n_fft)
    sig = frame_to_wave(frames, hop_len=frame_hop_len)
    return sig

def concate_conv_output(inputs):
    output_l = []
    for batch in inputs:
        output = batch[0]
        for i in range(1, len(batch)):
            output = torch.cat((output, batch[i]), dim=1)
        output_l.append(output.T)
    output = nn.utils.rnn.pad_sequence(output_l, batch_first=True)
    return output

def maxpool1(arr):
    """
        Get maximum element from 2d tensor

        Parameters
        ----------
        arr : 2d tensor
    """
    maxx = []
    for r in arr:
        maxx.append(torch.max(r))
    
    return torch.max(torch.tensor(maxx))

def convolve_ir(clean_speech, rir_file: str):
    """
        Adds the provided impulse response to the audio file

        Parameters
        ----------
        clean_speech : Clean audio or path to clean audio file
        rir_file : path to impulse response
    """
    if type(clean_speech) == str:
        clean_speech, sr_c = sf.read(clean_speech)

    rir, sr_r = sf.read(rir_file)
    clean_speech = normalize(clean_speech)
    rir = normalize(rir)

    reverbed_speech = normalize(fftconvolve(clean_speech, rir, mode='full'))
    return reverbed_speech

def convolve_and_save(clean_speech_file: str, rir_file: str, save_path: str, save_id: int):
    """
        Convolves the clean speech audio with room impulse response

        Parameters
        ----------
        clean_speech_file : Clean audio file location
        rir_file : Room impulse response file location
        save_path : Location where to savet the convoluted audio
        save_id : sample id
    """
    # Reading clean speech and room impulse response
    clean_speech, fs_c = sf.read(clean_speech_file)
    rir, fs_r = sf.read(rir_file)

    clean_speech = normalize(clean_speech)
    rir = normalize(rir)

    reverbed_speech = normalize(fftconvolve(clean_speech, rir, mode='same'))
    filename = f'{save_id}.wav'
    sf.write(os.path.join(save_path, filename), reverbed_speech, samplerate=fs_c)


def generate(clean_speech_paths: str, room_impulse_response_paths: str, destination_path: str, num_of_samples: int):
    """
        Dataset will be generated by convolving each clean speech audio with a random room impulse response.

        Parameters
        ----------
        clean_speech_paths : list of path to clean audio recordings.
        room_impulse_response_paths : list of RIRs wav files.
        destination_path : path to the destination folder were generated files will be saved.
        num_of_samples : Number of Datasamples to be created
    """
    # check wether the directory for destination exists
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)
    
    dataset = []

    num_of_samples_of_clean_speech = len(clean_speech_paths)
    # list to keep track of filepath to clean and it's corresponding reverberated audio
    for i in tqdm(range(num_of_samples)):
        ind = i % int(num_of_samples_of_clean_speech)
        rnd_rir = random.choice(room_impulse_response_paths)
        clean_speech = clean_speech_paths[ind]
        convolve_and_save(clean_speech, rnd_rir, destination_path, i+1)
        dataset.append([i+1, clean_speech])
    np.savetxt(os.path.join(destination_path, 'dataset_info'), dataset, fmt='%s, %s')

def listfiles(dirPath: str) -> list:
    """
        Returns the list of all audio files from a directory

        Parameters
        ----------
        dirPath : directory path
    """
    obj = os.scandir(dirPath)
    ls = []
    for path, subdirs, files in os.walk(dirPath):
        for name in files:
            if name.endswith(('.wav','.flac')):
                ls.append(os.path.join(path, name))
    return ls


# if __name__ == "__main__":
#     speech_filepaths = listfiles('.\\LibriSpeech')
#     rir_paths = listfiles('.\\rirs_noises\\simulated_rirs')
#     destination = '.\\generated_dataset'
#     num_of_samples=10_000
#     generate(speech_filepaths, rir_paths, destination, num_of_samples)