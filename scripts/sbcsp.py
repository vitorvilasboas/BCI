#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 16:20:19 2020
@author: vboas
"""
import math
import itertools
import numpy as np
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.linalg import eigh
from scipy.fftpack import fft
from bci_utils import extractEpochs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import lfilter, butter, iirfilter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

class_ids = [1, 2]

data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01T.npy', allow_pickle=True)
smin, smax = math.floor(0.5 * info['fs']), math.floor(2.5 * info['fs'])
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
ZT = [epochs[np.where(labels==i)] for i in class_ids]
ZT = np.r_[ZT[0],ZT[1]]
tT = np.r_[class_ids[0]*np.ones(int(len(ZT)/2)), class_ids[1]*np.ones(int(len(ZT)/2))]

data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01E.npy', allow_pickle=True)
smin, smax = math.floor(0.5 * info['fs']), math.floor(2.5 * info['fs'])
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
ZV = [epochs[np.where(labels==i)] for i in class_ids]
ZV = np.r_[ZV[0],ZV[1]]
tV = np.r_[class_ids[0]*np.ones(int(len(ZV)/2)), class_ids[1]*np.ones(int(len(ZV)/2))]


#%% Sub-band definitions
Fs = info['fs']
f_low, f_high = 0, 40
DFT = 0
nbands = 9    

n_bins = f_high - f_low
overlap = 0.5 
step = n_bins / nbands
size = step / overlap

sub_bands = []
for i in range(nbands):
    fl_sb = round(i * step + f_low)
    fh_sb = round(i * step + size + f_low)
    if fh_sb <= f_high: sub_bands.append([fl_sb, fh_sb]) 
    # se ultrapassar o limite superior da banda total, desconsidera a última sub-banda
    # ... para casos em que a razão entre a banda total e n_bands não é exata 

# print(sub_bands)
nbands = len(sub_bands)
# print(nbands)


#%%  Filtering
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
    
    XT, XV = [], []
    for i in range(nbands):
        bmin = sub_bands[i][0] * dft_size_band
        bmax = sub_bands[i][1] * dft_size_band
        # print(bmin, bmax)
        XT.append(XT_FFT[:, :, bmin:bmax])
        XV.append(XV_FFT[:, :, bmin:bmax])
    
    
else: # IIR Filtering
    nyq = 0.5 * Fs
    XT, XV = [], []
    for i in range(nbands):
        low = sub_bands[i][0] / nyq
        high = sub_bands[i][1] / nyq
        if high >= 1: high = 0.99
        b, a = butter(5, [low, high], btype='bandpass')
        # b, a = iirfilter(5, [low,high], btype='band')
        # XT.append(lfilter(b, a, ZT)) 
        # XV.append(lfilter(b, a, ZV)) 
        XT.append(filtfilt(b, a, ZT))  
        XV.append(filtfilt(b, a, ZV)) 
    

#%% CSP
ncomp = 8
csp_filters = []
for i in range(nbands):
    e, c, s = XT[i].shape
    classes = np.unique(tT)   
    Xa = XT[i][classes[0] == tT,:,:]
    Xb = XT[i][classes[1] == tT,:,:]
    
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
    csp_filters.append(W.T[:ncomp])

XT_CSP, XV_CSP = [], []
for i in range(nbands):
    YT = np.asarray([ np.dot(csp_filters[i], ep) for ep in XT[i] ])
    YV = np.asarray([ np.dot(csp_filters[i], ep) for ep in XV[i] ])
    XT_CSP.append( np.log( np.mean( YT ** 2, axis=2 ) ) ) # Feature extraction
    XV_CSP.append( np.log( np.mean( YV ** 2, axis=2 ) ) ) # Feature extraction
    

#%% LDA
SCORE_T = np.zeros((len(ZT), nbands))
SCORE_V = np.zeros((len(ZV), nbands))
for i in range(nbands):
    lda = LDA()
    lda.fit(XT_CSP[i], tT)
    SCORE_T[:, i] = np.ravel(lda.transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
    SCORE_V[:, i] = np.ravel(lda.transform(XV_CSP[i]))
    
       
#%% Bayesian Meta-Classifier
SCORE_T0 = SCORE_T[tT == class_ids[0], :]
SCORE_T1 = SCORE_T[tT == class_ids[1], :]
p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))

    
#%% Final classification   
clf_final = SVC(kernel='linear', C=10 **(-4), probability=True)
# clf_final = LDA()
# clf_final = GaussianNB()
# clf_final = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)           
# clf_final = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=None, min_samples_split=2)
# clf_final = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, learning_rate_init=0.001, alpha=0.0001, activation='relu', hidden_layer_sizes=(100,2))

clf_final.fit(META_SCORE_T, tT)
scores_labels = clf_final.predict(META_SCORE_V)
scores_proba = clf_final.predict_proba(META_SCORE_V)

acc = np.mean(scores_labels == tV)
kappa = cohen_kappa_score(scores_labels, tV)

print(acc, kappa)