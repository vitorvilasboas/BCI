# -*- coding: utf-8 -*-
import os
import pickle
import pyOpenBCI
import collections
import numpy as np
import time
import pandas as pd
from datetime import datetime
from scipy.io import loadmat
from scipy.signal import welch
from scripts.bci_utils import extractEpochs, nanCleaner, Filter
from scipy.linalg import eigh
import math
import matplotlib.pyplot as plt
from scripts.bci_utils import labeling

data, events, info = labeling(path='/mnt/dados/eeg_data/IV2a/', ds='IV2a', session='T', subj=1, channels=None, save=False)
# data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A09T.npy', allow_pickle=True)
class_ids = [1,2]
smin = math.floor(0.5 * info['fs'])
smax = math.floor(2.5 * info['fs'])
buffer_len = smax - smin
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
epochs = nanCleaner(epochs)

X = [ epochs[np.where(labels == i)] for i in class_ids ]

Y1 = X[0]
Y2 = X[1]

R1 = np.zeros(Y1.shape)
R2 = np.zeros(Y2.shape)
D = np.eye(22,22) - np.ones((22,22))/22
for i in range(len(R1)):
    R1[i,:,:] = (Y1[i,:,:].T @ D).T
    R2[i,:,:] = (Y2[i,:,:].T @ D).T

canal = 14

# y1 =  Y1[:,canal,:]
# y2 =  Y2[:,canal,:]

y1 =  R1[:,canal,:]
y2 =  R2[:,canal,:]

index1 = y1[np.isnan(y1)]
index2 = y2[np.isnan(y2)]

y1[np.isnan(y1)] = np.zeros(index1.shape)
y2[np.isnan(y2)] = np.zeros(index2.shape)

freq, p1 = welch(y1, fs=250, nfft=499)
_, p2 = welch(y2, fs=250, nfft=499)

p1 = np.real(p1)
p2 = np.real(p2)

m1 = np.mean(p1,0)
m2 = np.mean(p2,0)


# f, pa = welch(y1, fs=250, nfft=499)
# plt.semilogy(f, np.mean(np.real(p1),0))
plt.plot(freq, np.log10(m1), label='LH') 
plt.plot(freq, np.log10(m2), label='RH')
plt.xlim((0,40))
plt.ylim((-14, -11.5))
plt.legend()
plt.title(str(canal))