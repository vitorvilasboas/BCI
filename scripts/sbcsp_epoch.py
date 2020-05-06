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
from bci_utils import labeling, extractEpochs
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.signal import lfilter, butter, iirfilter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
          
#%% DATASET AND SCENARIO INFO
''' III3a: subjects={'K3','K6','L1'}; prefix=''; class_ids={1,2,3,4}; sessions={None}; channels=[:60]     
    III4a: subjects={'aa','al','av','aw','ay'}; prefix=''; class_ids={1,3}; sessions={None}; channels=[:118]
    IV2a: subjects={1,2,...,9}; prefix='A0'; class_ids={1,2,3,4} ; sessions={'T','E'} ; channels=[:22]
    IV2b: subjects={1,2,...,9}; prefix='B0'; class_ids={1,2}; sessions={'01T','02T','03T','04E','05E'}; channels=[:3]        
    Lee19: subjects={1,2,...,54}; prefix='S'; class_ids={1,2}; sessions={1,2}; channels=[:62]; suffix='sess'; ch_cortex=[7,32,8,9,33,10,34,12,35,13,36,14,37,17,38,18,39,19,40,20] 
'''

ds = 'IV2a' # 'IV2a','IV2b','III3a','III4a','Lee19'      
path = '/mnt/dados/eeg_data/' + ds + '/' # PATH TO DATASET
suj = 1
class_ids = [1, 2]
sessions = ['T', 'E']
channels = None
prefix, suffix = 'A0', ''

#%% Load data
d_train, e_train, i_train = labeling(path=path, ds=ds, session=sessions[0], subj=suj, channels=channels, save=False)
# d_train, e_train, i_train = np.load(path + 'npy/' + prefix + str(suj) + '' + sessions[0] + '.npy', allow_pickle=True)

if not ds in ['III3a','III4a']: 
    d_test, e_test, i_test = labeling(path=path, ds=ds, session=sessions[1], subj=suj, channels=channels, save=False)
    # d_test, e_test, i_test = np.load(path + 'npy/' + prefix + str(suj) + '' + sessions[1] + '.npy', allow_pickle=True)

#%% Segmentation
# Fs = 250 if dataset in ['IV2a', 'IV2b', 'III3a', 'Lee19'] else 100
Fs = i_train['fs']

smin, smax = math.floor(0.5 * Fs), math.floor(2.5 * Fs)
epochsT, labelsT = extractEpochs(d_train, e_train, smin, smax, class_ids)

if not ds in ['III3a','III4a']: 
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

#%% Sub-band definitions
f_low, f_high = 0, 40
DFT = 1 # 0=IIR, 1=DFT
nbands = 10    

n_bins = f_high - f_low
overlap = 0.5 
step = n_bins / (nbands+1)
size = step / overlap

sub_bands = []
for i in range(nbands):
    fl_sb = i * step + f_low
    fh_sb = i * step + size + f_low
    # if fh_sb <= self.f_high: sub_bands.append([fl_sb, fh_sb]) # extrapola limite superior 1: descarta última sub-banda 
    # if fh_sb > self.f_high: fh_sb = self.f_high # extrapola limite superior 2: ajusta f_high ao limite
    sub_bands.append([fl_sb, fh_sb])
   
# print(sub_bands)
nbands = len(sub_bands)

#%%  Filtering
if DFT:
    q = smax - smin
    rf = Fs/q # resolução frequência
    m = round( (f_high - f_low) / rf ) # num bins
    size_freq = 2/rf # num posiçoes vetor resultante FFT equivalente cada frequencia absoluta 
    # 2 = sen e cos, componente complexo FFT separados e intercalados
    
    ZTF = fft(ZT)[:,:,:m]
    REAL = np.transpose(np.real(ZTF), (2, 0, 1))
    IMAG = np.transpose(np.imag(ZTF), (2, 0, 1))
    ZTF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    ZTF = np.transpose(ZTF, (1, 2, 0))

    XT = []
    bins = []
    for i in range(nbands):
        bmin = round(sub_bands[i][0] * size_freq)
        bmax = round(sub_bands[i][1] * size_freq)
        bins.append([bmin, bmax])
        XT.append(ZTF[:, :, bmin:bmax])

else: # IIR Filtering
    nyq = 0.5 * Fs
    XT = []
    for i in range(nbands):
        low = sub_bands[i][0] / nyq
        high = sub_bands[i][1] / nyq
        if low == 0: low = 0.001
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
    # XT_CSP.append( np.log( np.var( YT, axis=2 ) ) )
    

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

for ep in range(1):
    idx = int(np.random.choice(epochsV.shape[0], 1)) # escolhe uma época de teste
    Z, t = epochsV[idx], labelsV[idx]
    
    #%%  Filtering
    if DFT:
        ZF = fft(Z)[:,:m]
        REAL, IMAG = np.real(ZF).T, np.imag(ZF).T
        ZF = np.transpose(list(itertools.chain.from_iterable(zip(IMAG, REAL))))
        
        X = []
        for i in range(nbands):
            bmin = round(sub_bands[i][0] * size_freq) # round(2/rf), rf=Fs/q 
            bmax = round(sub_bands[i][1] * size_freq)
            X.append(ZF[:, bmin:bmax])
        
    else: # IIR Filtering
        nyq = 0.5 * Fs
        X = []
        for i in range(nbands):
            low = sub_bands[i][0] / nyq
            high = sub_bands[i][1] / nyq
            if low == 0: low = 0.001
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
        # f = np.log(np.var(Y, axis=1))
        features.append(f)
        
    # xcsp0 = csp_filters[0] 
    # x0 = X[0]
    # xy0 = np.dot(xcsp0, x0)  
    # xf0_a = xy0**2
    # xf0_b = np.mean(xf0_a, axis=1)
    # xf0 = np.log(xf0_b)
            
    #%% LDA
    lda_scores = np.asarray([ np.ravel(lda_list[i].transform(features[i].reshape(1,-1))) for i in range(nbands) ]).T  
         
    #%% Bayesian Meta-classifier
    meta_score = np.log(p0.pdf(lda_scores) / p1.pdf(lda_scores))
    
    #%% Final classification
    y_label = clf_final.predict(meta_score.reshape(1, -1))
    y_prob = clf_final.predict_proba(meta_score.reshape(1, -1))
    print(f'Epoch idx: {idx}\nTrue target (t): {t}\nPredicted target (y): {y_label}\nLikely: {y_label==t}\nClasses Prob: {y_prob}')
    