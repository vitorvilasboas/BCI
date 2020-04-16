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
from sklearn.metrics import cohen_kappa_score
from bci_utils import labeling, extractEpochs
from scipy.signal import lfilter, butter, iirfilter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
           
#%% SET DATA SET INFO
''' Value Space::: 
    IV2a: subjects={1,2,...,9} 
          class_ids={1,2,3,4} 
          sessions={'T','E'} 
          channels=[:22]
    
    IV2b: subjects={1,2,...,9} 
          class_ids={1,2}     
          sessions={'01T','02T','03T','04E','05E'} 
          channels=[:3]        
    
    III3a: subjects={'K3','K6','L1'}
           class_ids={1,2,3,4}
           sessions={None} 
           channels=[:60]
           
    III4a: subjects={'aa','al','av','aw','ay'}; 
           class_ids={1,3} 
           sessions={None} 
           channels=[:118]
    
    Lee19: subjects={1,2,...,54} 
           class_ids={1,2} 
           sessions={1,2} 
           channels=[:62] 
           ch_cortex=[7,32,8,9,33,10,34,12,35,13,36,14,37,17,38,18,39,19,40,20] 
'''

dataset = 'IV2a' #{'IV2a','IV2b','III3a','III4a','Lee19'}      
path = '/mnt/dados/eeg_data/IV2a/'  ## >>> ENTER THE PATH TO THE DATASET HERE
subject = 1
channels = None
class_ids = [1, 2]

d_train, e_train, i_train = labeling(path=path, ds=dataset, session='T', subj=subject, channels=channels, save=False)
# d_train, e_train, i_train = np.load(path + 'npy/A0' + str(subject) + 'T.npy', allow_pickle=True)

if not dataset in ['III3a','III4a']: 
    d_test, e_test, i_test = labeling(path=path, ds=dataset, session='E', subj=subject, channels=channels, save=False)
    # d_test, e_test, i_test = np.load(path + 'npy/A0' + str(subject) + 'E.npy', allow_pickle=True)

#%% Segmentation
# Fs = 250 if dataset in ['IV2a', 'IV2b', 'III3a', 'Lee19'] else 100
Fs = i_train['fs']

smin, smax = math.floor(0.5 * Fs), math.floor(2.5 * Fs)
epochsT, labelsT = extractEpochs(d_train, e_train, smin, smax, class_ids)

if not dataset in ['III3a','III4a']: 
    epochsV, labelsV = extractEpochs(d_test, e_test, smin, smax, class_ids)
else: 
    epochs, labels = np.copy(epochsT), np.copy(labelsT)
    test_size = int(len(epochs) * 0.5)
    train_size = int(len(epochs) - test_size)
    train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
    epochsT, labelsT = epochs[:train_size], labels[:train_size] 
    epochsV, labelsV = epochs[train_size:], labels[train_size:]

ZT = [epochsT[np.where(labelsT==i)] for i in class_ids]
ZT = np.r_[ZT[0],ZT[1]]
tT = np.r_[class_ids[0]*np.ones(int(len(ZT)/2)), class_ids[1]*np.ones(int(len(ZT)/2))]

ZV = [epochsV[np.where(labelsV==i)] for i in class_ids]
ZV = np.r_[ZV[0],ZV[1]]
tV = np.r_[class_ids[0]*np.ones(int(len(ZV)/2)), class_ids[1]*np.ones(int(len(ZV)/2))]

#%% Filtering
f_low, f_high = 8, 30
DFT = 0 # 0=IIR, 1=DFT

if DFT:
    buffer_len = smax - smin
    dft_res_freq = Fs/buffer_len # resolução em frequência fft
    dft_size_band = round(2/dft_res_freq) # 2 representa sen e cos que foram separados do componente complexo da fft intercalados
    
    data_out = fft(ZT)
    REAL = np.transpose(np.real(data_out), (2, 0, 1))
    IMAG = np.transpose(np.imag(data_out), (2, 0, 1))
    data_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    XT_FFT = np.transpose(data_out, (1, 2, 0))
    
    data_out = fft(ZV)
    REAL = np.transpose(np.real(data_out), (2, 0, 1))
    IMAG = np.transpose(np.imag(data_out), (2, 0, 1))
    data_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    XV_FFT = np.transpose(data_out, (1, 2, 0))
    
    bmin = f_low * dft_size_band
    bmax = f_high * dft_size_band
    # print(bmin, bmax)
    XT = XT_FFT[:, :, bmin:bmax]
    XV = XV_FFT[:, :, bmin:bmax]

else: # IIR Filtering
    nyq = 0.5 * Fs
    low = f_low / nyq
    high = f_high / nyq
    if high >= 1: high = 0.99
    b, a = butter(5, [low, high], btype='bandpass')
    # b, a = iirfilter(5, [low,high], btype='band')
    # XT = lfilter(b, a, ZT) 
    # XV = lfilter(b, a, ZV)
    XT = filtfilt(b, a, ZT)
    XV = filtfilt(b, a, ZV)


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
YV = np.asarray([np.dot(Wf, ep) for ep in XV])


#%% Feature extraction
XT_CSP = np.log(np.mean(YT ** 2, axis=2))
XV_CSP = np.log(np.mean(YV ** 2, axis=2))
# XV_CSPi = np.log(np.mean(YV[0] ** 2, axis=1))


#%% LDA Classifier
clf = LDA()
clf.fit(XT_CSP, tT)
scores_labels = clf.predict(XV_CSP)
acc = np.mean(scores_labels == tV)
kappa = cohen_kappa_score(scores_labels, tV)
print('Accuracy:', round(acc,4))
print('kappa:', round(kappa,4))