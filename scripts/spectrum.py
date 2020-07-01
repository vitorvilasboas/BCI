# -*- coding: utf-8 -*-
import os
import math
import time
import pickle
import pyOpenBCI
import collections
import numpy as np
import pandas as pd
import pywt # pacote Py Wavelets
from scipy.io import loadmat
from scipy.linalg import eigh
from datetime import datetime
from scipy.signal import welch, butter, lfilter, filtfilt, iirfilter, stft, morlet, cwt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scripts.bci_utils import extractEpochs, nanCleaner, Filter

from scipy.fftpack import fft

# data, events, info = labeling(path='/mnt/dados/eeg_data/IV2a/', ds='IV2a', session='T', subj=1, channels=None, save=False)
data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True)
Fs = 250


## Filtro A
# b, a = butter(5, [8/125, 30/125], btype='bandpass')
b, a = iirfilter(5, [8/125, 30/125], btype='band')
# X_ = lfilter(b, a, X) #filtfilt

plt.ion()
fig = plt.figure(figsize=(15, 4), facecolor='mintcream')

gridspec.GridSpec(2,2)
axes = fig.add_subplot(111)
T, tx = 4, 0.1
# t0, tn = 1/Fs, (1+T)/Fs
t0, tn = tx, T
for i in range(0, 15000, int(Fs*tx)):
    # plt.figure(figsize=(15, 4), facecolor='mintcream')
    # axes.ylim((-120, 120))
    plt.subplot2grid((2,2), (0,0), colspan=2)
    # plt.subplot(2, 1, 1)
    y = data[13, i:i+(T*Fs)]*1e6 # data[13, i:i+1000]*1e6
    # print(i)
    x = np.linspace(t0, tn, T*Fs)
    plt.gca().cla() # optionally clear axes # plt.clf(), plt.cla(), plt.close()
    plt.plot(x,y)
    plt.title('Sinal bruto')
    plt.ylim((-60, 60))
    plt.ylabel(r'$\mu$V')
    plt.xlabel('Tempo (s)')
    plt.draw()
    # i += 0.1
    
    # plt.subplot(2, 1, 2)
    plt.subplot2grid((2,2), (1,0))
    y_ = lfilter(b, a, y) #filtfilt
    freq, plh = welch(y_, fs=Fs, nfft=(y_.shape[-1]-1))
    plt.gca().cla() # optionally clear axes # plt.clf(), plt.cla(), plt.close()
    plt.plot(freq, plh*1e13)
    plt.subplots_adjust(hspace = .6)
    plt.title('Welch')
    plt.xlim((6, 32))
    plt.ylim((0, 20*1e13))
    plt.ylabel(r'$\mu$V')
    plt.xlabel('Frequência (Hz)')
    plt.draw()
    
    t0 += (tx)
    tn += (tx)
    plt.pause(tx)

fig.tight_layout()
plt.show(block=True)



# plt.ion()
# fig = plt.figure(figsize=(10, 4), facecolor='mintcream')
# axes = fig.add_subplot(111)
# T, tx = 4, 0.5
# # t0, tn = 1/Fs, (1+T)/Fs
# t0, tn = tx, T
# for i in range(0, 15000, int(Fs*tx)):
#     y = data[13, i:i+(T*Fs)]*1e6 # data[13, i:i+1000]*1e6
#     print(i)
#     # x = np.linspace(t0, tn, T*Fs)
#     y_ = lfilter(b, a, y) #filtfilt
#     freq, plh = welch(y_, fs=Fs, nfft=(y_.shape[-1]-1))
#     plt.gca().cla() # optionally clear axes # plt.clf(), plt.cla(), plt.close()
#     plt.plot(freq, plh*1e13)
#     plt.xlim((5, 35))
#     plt.draw()
#     # i += 0.1
#     t0 += (tx)
#     tn += (tx)
#     plt.pause(tx)
# plt.show(block=True)



# class_ids = [1,2]
# smin = math.floor(0.5 * info['fs'])
# smax = math.floor(2.5 * info['fs'])
# buffer_len = smax - smin
# epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
# epochs = nanCleaner(epochs)

# X = [ epochs[np.where(labels == i)] for i in class_ids ]
# X = np.vstack(X)


## Filtro A
# b, a = butter(5, [8/125, 30/125], btype='bandpass')
# b, a = iirfilter(5, [8/125, 30/125], btype='band')
# X_ = lfilter(b, a, X) #filtfilt

# lh = X_[0]
# rh = X_[100]

# # Filtro B
# # D = np.eye(22,22) - np.ones((22,22))/22
# # lh = D @ lh
# # rh = D @ rh

# ch = 13 # 7,13 = hemisf esquerdo (atenua RH) 
# # ch = 17 # 11,17 = hemisf direito (atenua LH)
# lado = 'hemif. esquerdo' if ch in [7,13] else 'hemif. direito'

# ## Welch
# # sinais de duas trials de classes diferentes com o mesmo canal localizado em um dos lados do cérebro
# freq, plh = welch(lh[ch], fs=info['fs'], nfft=(lh.shape[-1]-1))
# _   , prh = welch(rh[ch], fs=info['fs'], nfft=(rh.shape[-1]-1)) 

# plt.plot(freq, plh*1e13, label='LH') 
# plt.plot(freq, prh*1e13, label='RH')
# plt.xlim((0,40))
# # plt.ylim((-35, -25))
# plt.title(f'Welch C3 ({lado})')
# plt.ylabel(r'$\mu$V')
# plt.xlabel('Frequência (Hz)')
# plt.legend()

# ## FFT
# T = 1/info['fs']
# # freq = np.linspace(0.0, 1.0/(2.0*T), lh.shape[-1]//2)
# freq = np.fft.fftfreq(lh.shape[-1], T)
# mask = freq>0
# freq = freq[mask]

# flh = np.abs(np.fft.fft(lh[ch]))[mask]
# frh = np.abs(np.fft.fft(rh[ch]))[mask]

# # flh = (2.0 * np.abs( fft(lh[ch]) / lh.shape[-1]))[mask]
# # frh = (2.0 * np.abs( fft(rh[ch]) / rh.shape[-1]))[mask]

# plt.figure()
# plt.plot(freq, flh*1e5, label='LH')
# plt.plot(freq, frh*1e5, label='RH')
# plt.xlim((0,40))
# plt.title(f'FFT C3 ({lado})')
# plt.ylabel(r'$\mu$V')
# plt.xlabel('Frequência (Hz)')
# plt.legend()

# # freq, tempo, Zxx = stft(lh[ch], info['fs'], nperseg=lh.shape[-1])
# # plt.figure()
# # plt.pcolormesh(tempo, freq, np.abs(Zxx)) # , vmin=0, vmax=amp
# # plt.title('STFT Magnitude')
# # plt.ylabel('Frequency [Hz]')
# # plt.xlabel('Time [sec]')
# # plt.legend()

# # (cA, cD) = pywt.dwt(lh, 'db1')


# ###########################################################
# X = [ epochs[np.where(labels == i)] for i in class_ids ]
# Xa = X[0] # all epochs LH
# Xb = X[1] # all epochs RH

# D = np.eye(22,22) - np.ones((22,22))/22
# Ra = np.asarray([D @ Xa[i] for i in range(len(Xa))])
# Rb = np.asarray([D @ Xb[i] for i in range(len(Xb))])

# xa = Ra[:,ch] # all epochs, 1 channel, all samples LH
# xb = Rb[:,ch] # all epochs, 1 channel, all samples RH
# # xa[np.isnan(xa)] = np.zeros(xa[np.isnan(xa)].shape)
# # xb[np.isnan(xb)] = np.zeros(xb[np.isnan(xb)].shape)

# ## Welch
# freq, pa = welch(xa, fs=250, nfft=(xa.shape[-1]-1)) # nfft=499 para q=500
# _   , pb = welch(xb, fs=250, nfft=(xb.shape[-1]-1)) 
# pa, pb = np.real(pa), np.real(pb)
# ma, mb = np.mean(pa,0), np.mean(pb,0)

# plt.figure(3, facecolor='mintcream')
# plt.subplots(figsize=(10, 12), facecolor='mintcream')
# plt.subplot(2, 1, 1)
# # plt.semilogy(freq, np.mean(p1,0), label='LH')
# # plt.semilogy(freq, np.mean(p2,0), label='RH')
# plt.plot(freq, ma*1e13, label='LH')  # np.log10(ma)
# plt.plot(freq, mb*1e13, label='RH')
# plt.xlim((0,40))
# # plt.ylim((-14, -11.5))
# plt.title(f'Welch C3 ({lado})')
# plt.ylabel(r'$\mu$V')
# plt.xlabel('Frequência (Hz)')
# plt.legend()

# ## FFT
# T = 1/info['fs']
# # freq = np.linspace(0.0, 1.0/(2.0*T), xa.shape[-1]//2)
# freq = np.fft.fftfreq(xa.shape[-1], T)
# mask = freq>0
# freq = freq[mask]

# fa = np.abs(np.fft.fft(xa))[:, mask]
# fb = np.abs(np.fft.fft(xb))[:, mask]
# # fa = (2.0 * np.abs( fft(xa) / xa.shape[-1]))[:, mask]
# # fb = (2.0 * np.abs( fft(xb) / xb.shape[-1]))[:, mask]

# ma, mb = np.mean(fa,0), np.mean(fb,0)

# plt.subplot(2, 1, 2)
# plt.plot(freq, ma*1e5, label='LH')
# plt.plot(freq, mb*1e5, label='RH')
# plt.xlim((0,40))
# plt.title(f'FFT C3 ({lado})')
# plt.ylabel(r'$\mu$V')
# plt.xlabel('Frequência (Hz)')
# plt.legend()





# plt.ion()
# fig = plt.figure(figsize=(15, 4), facecolor='mintcream')
# # axes = fig.add_subplot(111)
# T, tx = 1, 0.1
# # t0, tn = 1/Fs, (1+T)/Fs
# t0, tn = tx, T
# for i in range(0, 15000, int(Fs*tx)):
#     # plt.figure(figsize=(15, 4), facecolor='mintcream')
#     plt.ylim((-120, 120))
#     y = data[13, i:i+(T*Fs)]*1e6 # data[13, i:i+1000]*1e6
#     print(i)
#     x = np.linspace(t0, tn, T*Fs)
#     plt.gca().cla() # optionally clear axes # plt.clf(), plt.cla(), plt.close()
#     plt.plot(x,y)
#     plt.draw()
#     # i += 0.1
#     t0 += tx
#     tn += tx
#     plt.pause(tx)
# plt.show(block=True)