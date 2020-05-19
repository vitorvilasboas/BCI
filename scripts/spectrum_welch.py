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
from scipy.signal import welch
import matplotlib.pyplot as plt
from scripts.bci_utils import extractEpochs, nanCleaner, Filter, labeling

# data, events, info = labeling(path='/mnt/dados/eeg_data/IV2a/', ds='IV2a', session='T', subj=1, channels=None, save=False)
data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True)
class_ids = [1,2]
smin = math.floor(0.5 * info['fs'])
smax = math.floor(2.5 * info['fs'])
buffer_len = smax - smin
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
epochs = nanCleaner(epochs)

X = [ epochs[np.where(labels == i)] for i in class_ids ]

X1 = X[0] # LH
X2 = X[1] # RH

e, c, s = X1.shape
R1 = np.zeros((e, c, s))
R2 = np.zeros((e, c, s))
D = np.eye(c,c) - np.ones((c,c))/c
for i in range(len(R1)):
    R1[i,:,:] = (X1[i,:,:].T @ D).T
    R2[i,:,:] = (X2[i,:,:].T @ D).T

canal = 17 # 13=hemisf_esquerdo(RH) ; 17=hemisf_direito(LH)

# y1 =  Y1[:,canal,:]
# y2 =  Y2[:,canal,:]

y1 =  R1[:,canal,:]
y2 =  R2[:,canal,:]

index1 = y1[np.isnan(y1)]
index2 = y2[np.isnan(y2)]

y1[np.isnan(y1)] = np.zeros(index1.shape)
y2[np.isnan(y2)] = np.zeros(index2.shape)

freq, p1 = welch(y1, fs=250, nfft=(y1.shape[-1]-1)) # nfft=499 para q=500
_, p2 = welch(y2, fs=250, nfft=(y2.shape[-1]-1)) 

p1r = np.real(p1)
p2r = np.real(p2)

m1 = np.mean(p1r,0)
m2 = np.mean(p2r,0)

# plt.semilogy(freq, np.mean(p1,0), label='LH')
# plt.semilogy(freq, np.mean(p2,0), label='RH')
plt.plot(freq, np.log10(m1), label='LH') 
plt.plot(freq, np.log10(m2), label='RH')
plt.xlim((0,40))
plt.ylim((-14, -11.5))
plt.legend()
plt.title(str(canal))