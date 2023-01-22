"""
    Author : Meet Sable
    ID : 201901442
"""

import torch
from torch import nn

from librosa.display import waveshow, specshow

import matplotlib.pyplot as plt
import soundfile as sf
import random
import numpy as np

from helper_funcs import concate_conv_output, wave_to_stft, stft_to_wave, maxpool1, convolve_ir

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = nn.Conv2d(1, 4, (10, 10), padding=(4,4))
        self.conv2 = nn.Conv2d(4, 4, (5, 5), padding=(2,2))
        self.conv3 = nn.Conv2d(4, 8, (7, 7), padding=(3,3))
        self.conv4 = nn.Conv2d(8, 8, (5, 5), padding=(2,2))
        self.conv5 = nn.Conv2d(8, 8, (3, 3), padding=(1))

        # 257 -> 256 after conv layers

        self.bilstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, bidirectional=True)

        self.fc1 = nn.Linear(4*256, 257)

    def forward(self, inputs):

        inputs.unsqueeze_(dim=0)
        inputs.unsqueeze_(dim=0)
        c_outs = self.conv1(inputs)
        inputs.squeeze_()
        c_outs = self.conv2(c_outs)
        c_outs = self.conv3(c_outs)
        c_outs = self.conv4(c_outs)
        c_outs = self.conv5(c_outs)

        output = concate_conv_output(c_outs)

        output = torch.cat((inputs.squeeze()[:-1].T, output.squeeze())) 

        lstm_out, (h, c) = self.bilstm(output)
        # print(h.size())
        fc_in = h.flatten()

        out = self.fc1(fc_in)
        return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.bilstm = nn.LSTM(input_size=257, hidden_size=257, num_layers=2, bidirectional=True)
        
        self.conv1 = nn.Conv2d(1, 4, (5,5))
        self.conv2 = nn.Conv2d(1, 4, (3,3))
        self.conv3 = nn.Conv2d(1, 4, (1,1))

        self.fc1 = nn.Linear(12, 1)

    def forward(self, true, filtered):
        # print(true.size())
        # print(filtered.size())
        x = torch.cat([true.T, filtered])
        
        x.unsqueeze_(dim=0)
        lstm_out, (h, c) = self.bilstm(x)
        convs = [self.conv1(lstm_out), self.conv2(lstm_out), self.conv3(lstm_out)]

        fc_in = [0 for _ in range(12)]
        for i, conv in enumerate(convs, 0):
            for j, ch in enumerate(conv, 0):
                fc_in[i*4+j] = maxpool1(ch)
        fc_in = Tensor(fc_in)
        out = self.fc1(fc_in)

        return out

def train_gan(clean_speech_paths: list, rir_paths : list, num_of_iters : int, generator : Generator, discriminator : Discriminator, update_delay = 500):
    loss_func = nn.MSELoss()
    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()
        loss_func.cuda()

    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002)


    for itr in range(num_of_iters):
        clean_speech_filepath = random.choice(clean_speech_paths)
        rir_filepath = random.choice(rir_paths)

        # Preparing audio files
        clean_speech, sr = sf.read(clean_speech_filepath)
        rev_speech = convolve_ir(clean_speech, rir_filepath)
        win_len = int(0.032 * sr)
        hop_len = int(0.016 * sr)

        clean_speech_stft = wave_to_stft(clean_speech, sr, win_len, hop_len)
        rev_speech_stft = wave_to_stft(rev_speech, sr, win_len, hop_len)
    
        clean_speech_t = Tensor(np.abs(clean_speech_stft))
        rev_speech_t = Tensor(np.abs(rev_speech_stft))

        # training generator
        optimizer_g.zero_grad()
        gen_inv_impulse = generator(rev_speech_t)

        filtered_speech = rev_speech_t.T * gen_inv_impulse
    
        g_loss = loss_func(discriminator(clean_speech_t.clone().detach(), filtered_speech), Tensor(1).fill_(1.0))

        g_loss.backward()
        optimizer_g.step()

        # training Discriminator
        optimizer_d.zero_grad()

        real_loss = loss_func(discriminator(clean_speech_t.clone(), clean_speech_t.clone().T), Tensor(1).fill_(1.0))
        filtered_loss = loss_func(discriminator(clean_speech_t.clone(), filtered_speech), Tensor(1).fill_(0.0))

        d_loss = 0.5 * (real_loss + filtered_loss)
        d_loss.backward()
        optimizer_d.step()

        if itr % update_delay-1 == 0:
            print(f"[iteration:{itr}] \t [D loss:{d_loss}] \t [G loss:{g_loss}]")
        
    return generator, discriminator



# def reconstruction_from_stft_test():
#     clean_speech, sr = sf.read("LibriSpeech\dev-clean\\652\\129742\\652-129742-0001.flac")
#     wv = waveshow(clean_speech)
    
#     win_len = int(0.025*sr)
#     hop_len = int(0.001*sr)
#     frame_len = int(0.32*sr)
#     frame_hop_len = int(0.16*sr)

#     plt.figure()
#     sp = specshow(wave_to_stft(clean_speech, sr, win_len, hop_len))
    
#     frames = wave_to_stft_frames(clean_speech, sr, frame_len, frame_hop_len, win_len, hop_len)
#     print(frames[0].shape)
#     sig = stft_frames_to_wave(frames, sr, frame_hop_len, win_len, hop_len)
    
#     plt.figure()
#     wv2 = waveshow(sig)

#     plt.show()
#     return frames



# test()
