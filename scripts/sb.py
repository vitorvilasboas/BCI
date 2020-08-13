#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @author: vboas

import math
import itertools
import numpy as np
from time import time
from sklearn.svm import SVC
from scipy.stats import norm
from scipy.linalg import eigh
from scipy.fftpack import fft
from scripts.bci_utils import extractEpochs
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import lfilter, butter, iirfilter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
#%%
suj = 2
class_ids = [1, 2]
session = 'T'
data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0'+str(suj)+session+'.npy', allow_pickle=True)
Fs = info['fs']
f_low, f_high, ncsp, tmin, tmax, nbands = 4, 40, 8, 0.5, 2.5, 9
filtering = 'DFT' # IIR, DFT  

#%%
smin, smax = math.floor(tmin * Fs), math.floor(tmax * Fs)
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
epochs = [ epochs[np.where(labels==i)] for i in class_ids ]
Z = np.r_[epochs[0],epochs[1]]
t = np.r_[class_ids[0]*np.ones(int(len(Z)/2)), class_ids[1]*np.ones(int(len(Z)/2))]

#%% Sub-band definitions
n_bins = f_high - f_low
overlap = 0.5 
step = n_bins / (nbands+1)
size = step / overlap

n_samples = smax - smin
dft_resf = Fs/n_samples 
print(dft_resf)
bin_size = 2/dft_resf # 2 representa sen e cos
nyq = 0.5 * Fs

sb_ranges = []
for i in range(nbands):
    fl_sb = i * step + f_low
    fh_sb = i * step + size + f_low
    sb_ranges.append([fl_sb, fh_sb])

sb_dft = []
for i in range(nbands):
    bmin = round(sb_ranges[i][0] * bin_size)
    bmax = round(sb_ranges[i][1] * bin_size)
    sb_dft.append([bmin,bmax])
    
sb_iir = []
for i in range(nbands):
    low = sb_ranges[i][0] / nyq
    high = sb_ranges[i][1] / nyq
    if low == 0: low = 0.001
    if high >= 1: high = 0.99
    sb_iir.append([low,high])

#%%  Filtering
if filtering == 'DFT':
    ZF = fft(Z)
    REAL = np.transpose(np.real(ZF), (2, 0, 1))
    IMAG = np.transpose(np.imag(ZF), (2, 0, 1))
    ZF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    # ZF = REAL - IMAG
    ZF = np.transpose(ZF, (1, 2, 0))
    X = [ ZF[:, :, sb_dft[i][0]:sb_dft[i][1]] for i in range(nbands) ]
      
elif filtering == 'IIR':
    X = []
    for i in range(nbands):
        b, a = butter(5, [ sb_iir[i][0], sb_iir[i][1] ], btype='bandpass')
        X.append(lfilter(b, a, Z)) 
        # b, a = iirfilter(5, [ sb_iir[i][0], sb_iir[i][1] ], btype='band')
        # X.append(filtfilt(b, a, Z)) 

#%% CSP
csp_filters = []
for i in range(nbands):
    e, c, s = X[i].shape
    classes = np.unique(t)   
    Xa = X[i][classes[0] == t,:,:]
    Xb = X[i][classes[1] == t,:,:]
    
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
    csp_filters.append(W.T[:ncsp])



Y = [ np.asarray([ np.dot(csp_filters[i], ep) for ep in X[i] ]) for i in range(nbands)] # Spatial Filtering

XCSP = [ np.log(np.mean(Y[i] ** 2, axis=2)) for i in range(nbands) ] # Feature extraction
# XCSP = [ np.log(np.var(Y[i], axis=2)) for i in range(nbands) ] # Feature extraction
    
#%% LDA
lda = [ LDA() for i in range(nbands)]
for i in range(nbands): lda[i].fit(XCSP[i], t)

SCORE = np.asarray([ np.ravel(lda[i].transform(XCSP[i])) for i in range(nbands) ]).T 

# classificações de cada época nas N sub bandas - auto validação
# O LDA transforma os dados de treino e validação em cada sub-banda onde o resultado contínuo do LDA é mantido para construir um vetor de características de dimensão (n_bands)    
       
#%% Bayesian Meta-Classifier
SCORE_A = SCORE[t == class_ids[0], :]
SCORE_B = SCORE[t == class_ids[1], :] # separa os scores de cada sub-banda referentes a classe A e B
p0 = norm(np.mean(SCORE_A, axis=0), np.std(SCORE_A, axis=0)) # projeta uma distribuição normal/gaussiana a partir das médias (m0,m1) e dp (std0,std1) dos scores LDA em cada sub-banda ... 
p1 = norm(np.mean(SCORE_B, axis=0), np.std(SCORE_B, axis=0)) # entre todas as épocas de cada classe. p0 é uma espécie de filtro bayesiano favorável à classe A e p1 à classe B
META_SCORE = np.log(p0.pdf(SCORE) / p1.pdf(SCORE)) # scores aplicados nas funções de densidade de probabilidade de cada classe e calculada a razão entre elas.
# META_SCORE > 0 indica época melhor representada pela função p0
# META_SCORE < 0 a melhor representação é por p1

#%% Final classification   
clf_final = SVC(kernel='linear', C=1e-4, probability=True)
# clf_final = LDA()
# clf_final = GaussianNB()
# clf_final = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=3)           
# clf_final = DecisionTreeClassifier(criterion='entropy', random_state=0, max_depth=None, min_samples_split=2)
# clf_final = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, learning_rate_init=0.001, alpha=0.0001, activation='relu', hidden_layer_sizes=(100,2))
clf_final.fit(META_SCORE, t)



#%% EVALUATE
smin, smax = math.floor(1.5 * Fs), math.floor(3.5 * Fs)
session = 'E'
data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0'+str(suj)+session+'.npy', allow_pickle=True)
epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
epochs = [epochs[np.where(labels==i)] for i in class_ids]
Z = np.r_[epochs[0],epochs[1]]
t = np.r_[class_ids[0]*np.ones(int(len(Z)/2)), class_ids[1]*np.ones(int(len(Z)/2))]

dft_resf = Fs/n_samples 
bin_size = 2/dft_resf # 2 representa sen e cos

if filtering == 'DFT':
    ZF = fft(Z)
    REAL = np.transpose(np.real(ZF), (2, 0, 1))
    IMAG = np.transpose(np.imag(ZF), (2, 0, 1))
    ZF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    # ZF = REAL - IMAG
    ZF = np.transpose(ZF, (1, 2, 0))
    X = [ ZF[:, :, sb_dft[i][0]:sb_dft[i][1]] for i in range(nbands) ]
    
elif filtering == 'IIR':
    X = []
    for i in range(nbands):
        b, a = butter(5, [ sb_iir[i][0], sb_iir[i][1] ], btype='bandpass')
        X.append(lfilter(b, a, Z)) 
        # b, a = iirfilter(5, [ sb_iir[i][0], sb_iir[i][1] ], btype='band')
        # X.append(filtfilt(b, a, Z)) 
        
Y = [ np.asarray([ np.dot(csp_filters[i], ep) for ep in X[i] ]) for i in range(nbands) ] # Spatial Filtering

XCSP = [ np.log(np.mean(Y[i]**2, axis=2)) for i in range(nbands) ] # Feature extraction
# XCSP = [ np.log(np.var(Y[i], axis=2)) for i in range(nbands) ] # Feature extraction

score = np.dot(XCSP[0] - lda[0].xbar_, lda[0].scalings_)

SCORE = np.asarray([ np.ravel(lda[i].transform(XCSP[i])) for i in range(nbands) ]).T # Score LDA
    
META_SCORE = np.log(p0.pdf(SCORE) / p1.pdf(SCORE)) # Bayesian Meta-score

y = META_SCORE @ clf_final.coef_.T # y<0=classe A(1) ; y>0=classe B(2)
y_labels = clf_final.predict(META_SCORE) # Final classification
y_proba = clf_final.predict_proba(META_SCORE)


acc = clf_final.score(META_SCORE, t)
acc = np.mean(y_labels == t)
kappa = cohen_kappa_score(y_labels, t)
print('Accuracy:', round(acc,4))
# print('kappa:', round(kappa,4))