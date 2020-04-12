#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 01:09:46 2020
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

#%% Sub-band definitions
Fs = info['fs']
f_low, f_high = 0, 40
DFT = 1
nbands = 10    

n_bins = f_high - f_low
overlap = 0.5 
step = n_bins / nbands
size = step / overlap

sub_bands = []
for i in range(nbands):
    fl_sb = i * step + f_low
    fh_sb = i * step + size + f_low
    if fl_sb == 0: fl_sb = 0.001
    if fh_sb <= f_high: sub_bands.append([fl_sb, fh_sb]) 
    # se ultrapassar o limite superior da banda total, desconsidera a última sub-banda
    # ... para casos em que a razão entre a banda total e n_bands não é exata 

nbands = len(sub_bands)
print(sub_bands)

#%%  Filtering
if DFT:
    q = smax - smin
    rf = Fs/q # resolução frequência
    m = round( (f_high - f_low) / rf ) # num bins
    size_freq = round(2/rf) # num posiçoes vetor resultante FFT equivalente cada frequencia absoluta 
    # 2 = sen e cos, componente complexo FFT separados e intercalados
    
    ZTF = fft(ZT)[:,:,:m]
    REAL = np.transpose(np.real(ZTF), (2, 0, 1))
    IMAG = np.transpose(np.imag(ZTF), (2, 0, 1))
    ZTF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    ZTF = np.transpose(ZTF, (1, 2, 0))

    XT = []
    bins = []
    for i in range(nbands):
        bmin_ = sub_bands[i][0] * size_freq
        bmax_ = sub_bands[i][1] * size_freq
        bmin, bmax = int(bmin_), int(bmax_)
        bins.append([bmin, bmax])
        XT.append(ZTF[:, :, bmin:bmax])
    print(bins)

else: # IIR Filtering
    nyq = 0.5 * Fs
    XT = []
    for i in range(nbands):
        low = sub_bands[i][0] / nyq
        high = sub_bands[i][1] / nyq
        if high >= 1: high = 0.99
        b, a = butter(5, [low, high], btype='bandpass')
        # b, a = iirfilter(5, [low,high], btype='band')
        # XT.append(lfilter(b, a, ZT)) 
        XT.append(filtfilt(b, a, ZT))  
    

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
    XT_CSP.append( np.log( np.mean( YT ** 2, axis=2 ) ) ) # Feature extraction
    

#%% LDA
SCORE_T = np.zeros((len(ZT), nbands))
lda_list = []
for i in range(nbands):
    lda = LDA()
    lda.fit(XT_CSP[i], tT)
    SCORE_T[:, i] = np.ravel(lda.transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
    lda_list.append(lda)
    
#%% Bayesian Meta-Classifier
SCORE_T0 = SCORE_T[tT == class_ids[0], :]
SCORE_T1 = SCORE_T[tT == class_ids[1], :]
p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))

    
#%% Final classification
clf_final = SVC(kernel='linear', C=10 **(-4), probability=True)
# clf_final = LDA()
# clf_final = GaussianNB()
# clf_final = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)           
# clf_final = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=None, min_samples_split=2)
# clf_final = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, learning_rate_init=0.001, alpha=0.0001, activation='relu', hidden_layer_sizes=(100,2))
clf_final.fit(META_SCORE_T, tT)


############################################################
#%% ################# EVALUATE UNIQUE EPOCH
############################################################

data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01E.npy', allow_pickle=True)
smin, smax = math.floor(0.5 * info['fs']), math.floor(2.5 * info['fs'])
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)

# for ep in range(1):
idx = int(np.random.choice(epochs.shape[0], 1)) # escolhe uma época de teste
Z, t = epochs[idx], labels[idx]

#%%  Filtering
if DFT:
    ZF = fft(Z)[:,:m]
    REAL, IMAG = np.real(ZF).T, np.imag(ZF).T
    ZF = np.transpose(list(itertools.chain.from_iterable(zip(IMAG, REAL))))
    
    X = []
    for i in range(nbands):
        bmin_ = sub_bands[i][0] * size_freq # round(2/rf), rf=Fs/q 
        bmax_ = sub_bands[i][1] * size_freq
        bmin, bmax = round(bmin_), round(bmax_)
        # print(bmin_, bmin, bmax_, bmax)
        X.append(ZF[:, bmin:bmax])
    

else: # IIR Filtering
    nyq = 0.5 * Fs
    X = []
    for i in range(nbands):
        low = sub_bands[i][0] / nyq
        high = sub_bands[i][1] / nyq
        if high >= 1: high = 0.99
        b, a = butter(5, [low, high], btype='bandpass')
        # b, a = iirfilter(5, [low,high], btype='band')
        # X.append(lfilter(b, a, Z)) 
        X.append(filtfilt(b, a, Z))  
    
#%% CSP
features = []
for i in range(nbands):
    Y = np.dot(csp_filters[i], X[i])
    f = np.log(np.mean(Y**2, axis=1)) # Feature extraction
    features.append(f)
    
# csp0 = csp_filters[0] 
# X0 = X[0]
# Y0 = np.dot(csp0, X0)  
# f0_a = Y0**2
# f0_b = np.mean(f0_a, axis=1)
# f0 = np.log(f0_b)
        
#%% LDA
lda_scores = np.asarray([ np.ravel(lda_list[i].transform(features[i].reshape(1,-1))) for i in range(nbands) ]).T  
     
#%% Bayesian Meta-classifier
meta_score = np.log(p0.pdf(lda_scores) / p1.pdf(lda_scores))

#%% Final classification
y_label = clf_final.predict(meta_score.reshape(1, -1))
y_prob = clf_final.predict_proba(meta_score.reshape(1, -1))
print(idx, t, y_label==t, y_prob)