#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 08:24:12 2020
@author: vboas
"""
import math
import itertools
import numpy as np
from scipy.linalg import eigh
from scipy.fftpack import fft
from bci_utils import extractEpochs
from scipy.signal import lfilter, butter, iirfilter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class_ids = [1, 2]

data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True) ## >>> PUT HERE THE DATA SET PATH
smin, smax = math.floor(0.5 * info['fs']), math.floor(2.5 * info['fs'])
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
ZT = [epochs[np.where(labels==i)] for i in class_ids]
ZT = np.r_[ZT[0],ZT[1]]
tT = np.r_[class_ids[0]*np.ones(int(len(ZT)/2)), class_ids[1]*np.ones(int(len(ZT)/2))]

Fs = info['fs']
f_low, f_high = 8, 30
DFT = 1

#%% Filtering
if DFT:
    buffer_len = smax - smin
    dft_res_freq = Fs/buffer_len # resolução em frequência fft
    dft_size_band = round(2/dft_res_freq) # 2 representa sen e cos que foram separados do componente complexo da fft intercalados
    
    data_out = fft(ZT)
    REAL = np.transpose(np.real(data_out), (2, 0, 1))
    IMAG = np.transpose(np.imag(data_out), (2, 0, 1))
    data_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    XT_FFT = np.transpose(data_out, (1, 2, 0))

    bmin = f_low * dft_size_band
    bmax = f_high * dft_size_band
    # print(bmin, bmax)
    XT = XT_FFT[:, :, bmin:bmax]

else: # IIR Filtering
    nyq = 0.5 * Fs
    low = f_low / nyq
    high = f_high / nyq
    if high >= 1: high = 0.99
    b, a = butter(5, [low, high], btype='bandpass')
    # b, a = iirfilter(5, [low,high], btype='band')
    # XT = lfilter(b, a, ZT) 
    XT = filtfilt(b, a, ZT)

#%% CSP
ncomp = 8
e, c, s = XT.shape
classes = np.unique(tT)   
Xa = XT[classes[0] == tT,:,:]
Xb = XT[classes[1] == tT,:,:]

Sa = np.zeros((c, c)) 
Sb = np.zeros((c, c))
for i in range(int(e/2)):
    # Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T)
    # Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T)
    Sa += np.dot(Xa[i,:,:], Xa[i,:,:].T) / Xa[i].shape[-1] # sum((Xa * Xa.T)/q)
    Sb += np.dot(Xb[i,:,:], Xb[i,:,:].T) / Xb[i].shape[-1] # sum((Xb * Xb.T)/q)
Sa /= len(Xa)
Sb /= len(Xb)

[D, W] = eigh(Sa, Sa + Sb)
ind = np.empty(c, dtype=int)
ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
ind[1::2] = np.arange(0, c // 2)
W = W[:, ind]
Wf = W.T[:ncomp]
       
YT = np.asarray([np.dot(Wf, ep) for ep in XT])

#%% Feature extraction
XT_CSP = np.log(np.mean(YT ** 2, axis=2))

#%% LDA Classifier
clf = LDA()
clf.fit(XT_CSP, tT)

############################################################
#%% ################# EVALUATE UNIQUE EPOCH
############################################################

data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01E.npy', allow_pickle=True) ## >>> PUT HERE THE DATA SET PATH
smin, smax = math.floor(0.5 * info['fs']), math.floor(2.5 * info['fs'])
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)

for ep in range(1):
    idx = int(np.random.choice(epochs.shape[0], 1)) # escolhe uma época de teste
    Z, t = epochs[idx], labels[idx]
    
    #%%  Filtering
    if DFT:
        data_out = fft(Z)
        REAL = np.real(data_out).T
        IMAG = np.imag(data_out).T
        XFFT = np.transpose(list(itertools.chain.from_iterable(zip(IMAG, REAL))))
        bmin = f_low * dft_size_band
        bmax = f_high * dft_size_band
        # print(bmin, bmax)
        X = XFFT[:, bmin:bmax]
    else:
        nyq = 0.5 * Fs
        low = f_low / nyq
        high = f_high / nyq
        if high >= 1: high = 0.99
        b, a = butter(5, [low, high], btype='bandpass')
        # b, a = iirfilter(5, [low,high], btype='band')
        # X = lfilter(b, a, ZT) 
        X = filtfilt(b, a, ZT)
    
    #%% CSP
    Y = np.dot(Wf, X)
    features = np.log(np.mean(Y**2, axis=1))
    
    #%% LDA Classifier
    y_label = clf.predict(features.reshape(1, -1))
    y_prob = clf.predict_proba(features.reshape(1, -1))
    print(idx, t, y_label==t, y_prob)