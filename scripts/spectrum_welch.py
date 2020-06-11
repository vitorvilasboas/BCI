# -*- coding: utf-8 -*-
import os
import math
import time
import pickle
import pyOpenBCI
import collections
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.linalg import eigh
from datetime import datetime
from scipy.signal import welch, butter, lfilter, filtfilt, iirfilter
import matplotlib.pyplot as plt
from scripts.bci_utils import extractEpochs, nanCleaner, Filter, labeling

from scipy.fftpack import fft

# data, events, info = labeling(path='/mnt/dados/eeg_data/IV2a/', ds='IV2a', session='T', subj=1, channels=None, save=False)
data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True)
class_ids = [1,2]
smin = math.floor(0.5 * info['fs'])
smax = math.floor(2.5 * info['fs'])
buffer_len = smax - smin
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
epochs = nanCleaner(epochs)

X = [ epochs[np.where(labels == i)] for i in class_ids ]
X = np.vstack(X)

## Filtro A
# b, a = butter(5, [8/125, 30/125], btype='bandpass')
b, a = iirfilter(5, [8/125, 30/125], btype='band')
X = lfilter(b, a, X) #filtfilt

xa = X[0]
xb = X[100] 

# Filtro B
# D = np.eye(22,22) - np.ones((22,22))/22
# xa = D @ xa
# xb = D @ xb

ch = 13 # 7,13 = hemisf esquerdo (atenua RH - xa) 
# ch = 17 # 11,17 = hemisf direito (atenua LH - xb)

## Welch
freq, pa = welch(xa[ch], fs=info['fs'], nfft=(xa.shape[-1]-1))
_   , pb = welch(xb[ch], fs=info['fs'], nfft=(xb.shape[-1]-1)) 

plt.plot(freq, pa, label='LH') 
plt.plot(freq, pb, label='RH')
plt.xlim((0,40))
# plt.ylim((-35, -25))
plt.title(f'Welch [{ch+1}]')
plt.legend()


## FFT
T = 1/info['fs']
# freq = np.linspace(0.0, 1.0/(2.0*T), xa.shape[-1]//2)
freq = np.fft.fftfreq(xa.shape[-1], T)
mask = freq>0
freq = freq[mask]

fa = np.abs(np.fft.fft(xa[ch]))[mask]
fb = np.abs( np.fft.fft(xb[ch]))[mask]

# fa = (2.0 * np.abs( fft(xa[ch]) / xa.shape[-1]))[mask]
# fb = (2.0 * np.abs( fft(xb[ch]) / xb.shape[-1]))[mask]

plt.figure()
plt.plot(freq, fa, label='LH')
plt.plot(freq, fb, label='RH')
plt.xlim((0,40))
plt.title(f'FFT [{ch+1}]')
plt.legend()


###########################################################

# X = [ epochs[np.where(labels == i)] for i in class_ids ]

# X1 = X[0] # LH
# X2 = X[1] # RH

# e, c, s = X1.shape
# R1 = np.zeros((e, c, s))
# R2 = np.zeros((e, c, s))
# D = np.eye(c,c) - np.ones((c,c))/c
# for i in range(len(R1)):
#     R1[i,:,:] = (X1[i,:,:].T @ D).T
#     R2[i,:,:] = (X2[i,:,:].T @ D).T



# # y1 =  Y1[:,canal,:]
# # y2 =  Y2[:,canal,:]

# y1 = R1[:,canal,:]
# y2 = R2[:,canal,:]

# index1 = y1[np.isnan(y1)]
# index2 = y2[np.isnan(y2)]

# y1[np.isnan(y1)] = np.zeros(index1.shape)
# y2[np.isnan(y2)] = np.zeros(index2.shape)

# freq, p1 = welch(y1, fs=250, nfft=(y1.shape[-1]-1)) # nfft=499 para q=500
# _   , p2 = welch(y2, fs=250, nfft=(y2.shape[-1]-1)) 

# p1r = np.real(p1)
# p2r = np.real(p2)

# m1 = np.mean(p1r,0)
# m2 = np.mean(p2r,0)

# # plt.semilogy(freq, np.mean(p1,0), label='LH')
# # plt.semilogy(freq, np.mean(p2,0), label='RH')
# plt.plot(freq, np.log10(m1), label='LH') 
# plt.plot(freq, np.log10(m2), label='RH')
# plt.xlim((0,40))
# plt.ylim((-14, -11.5))
# plt.legend()
# plt.title(str(canal))