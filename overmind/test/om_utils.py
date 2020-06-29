# -*- coding: utf-8 -*-

import os
import re
import mne
import math
import pickle
import warnings
import itertools
import collections
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from time import time, sleep
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.stats import norm, mode
from datetime import datetime
from scipy.fftpack import fft
from scipy.linalg import eigh
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from hyperopt.fmin import generate_trials_to_calculate
from hyperopt import base, fmin, tpe, rand, hp, space_eval
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter, decimate, welch
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from bci_utils import nanCleaner, corrigeNaN, extractEpochs, Filter
from functools import partial
import random
from sklearn.preprocessing import normalize

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

class CSP():
    def __init__(self, n_components):
        self.n_components = n_components
        self.filters_ = None
    def fit(self, X, y):
        e, c, s = X.shape
        classes = np.unique(y)   
        Xa = X[classes[0] == y,:,:]
        Xb = X[classes[1] == y,:,:]
        S0 = np.zeros((c, c)) 
        S1 = np.zeros((c, c))
        for epoca in range(int(e/2)):
            # S0 = np.add(S0, np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T), out=S0, casting="unsafe")
            # S1 = np.add(S1, np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T), out=S1, casting="unsafe")
            # S0 += np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T) #covA Xa[epoca]
            # S1 += np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T) #covB Xb[epoca]
            S0 += np.dot(Xa[epoca, :, :], Xa[epoca, :, :].T) / Xa[epoca].shape[-1]  # sum((Xa * Xa.T)/q)
            S1 += np.dot(Xb[epoca, :, :], Xb[epoca, :, :].T) / Xb[epoca].shape[-1]  # sum((Xb * Xb.T)/q)
        S0 /= len(Xa)
        S1 /= len(Xb)
        [D, W] = eigh(S0, (S0 + S1))# + 1e-10 * np.eye(22))
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
        ind[1::2] = np.arange(0, c // 2)
        # W += 1e-1 * np.eye(22)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self # instruction add because cross-validation pipeline
    def transform(self, X):        
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2)) # Xcsp
        # XVAR = np.log(np.var(XT, axis=2))
        return XVAR

def classic_approach(ZT, ZV, tt, tv, fl, fh, ncsp, clf, Fs='250', filt='DFT', csp_reg=None):
    if filt == 'DFT':
        ZTF = fft(ZT)
        REAL = np.transpose(np.real(ZTF), (2, 0, 1))
        IMAG = np.transpose(np.imag(ZTF), (2, 0, 1))
        ZTF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        ZTF = np.transpose(ZTF, (1, 2, 0))
        bin_size = 2 / ( Fs / ZT.shape[-1] ) # 2 representa sen e cos
        XT = ZTF[:, :, round(fl*bin_size):round(fh*bin_size)]
        
        ZVF = fft(ZV)
        REAL = np.transpose(np.real(ZVF), (2, 0, 1))
        IMAG = np.transpose(np.imag(ZVF), (2, 0, 1))
        ZVF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        ZVF = np.transpose(ZVF, (1, 2, 0))
        bin_size = 2 / ( Fs / ZV.shape[-1] ) # 2 representa sen e cos
        XV = ZVF[:, :, round(fl*bin_size):round(fh*bin_size)]
    
    elif filt == 'IIR':
        nyq = 0.5 * Fs
        low = fl / nyq
        high = fh / nyq
        if low == 0: low = 0.001
        if high >= 1: high = 0.99
        b, a = butter(5, [low,high], btype='bandpass') # iirfilter(5,[low,high],btype='band')
        XT = lfilter(b, a, ZT) # filtfilt(b, a, ZT)
        XV = lfilter(b, a, ZV) # filtfilt(b, a, ZV)
    
    # csp = CSP(n_components=int(ncsp)) # CSP 1
    csp = mne.decoding.CSP(n_components=int(ncsp), reg=csp_reg) # CSP 2
    
    csp.fit(XT, tt)
    # csp_filters = csp.filters_ # CSP 1
    csp_filters = csp.filters_[:int(ncsp)] # CSP 2
    XT_CSP = csp.transform(XT)
    XV_CSP = csp.transform(XV)
    
    if clf['model'] == 'LDA': clf_final = LDA()
    elif clf['model'] == 'LR': clf_final = LogisticRegression()
    elif clf['model'] == 'SVM': clf_final = SVC(kernel=clf['kernel']['kf'], C=10**(clf['C']), gamma='scale', degree=3, probability=True)
    elif clf['model'] == 'KNN': clf_final = KNeighborsClassifier(n_neighbors=int(clf['neig']), metric=clf['metric'], p=3)                                       
    
    clf_final.fit(XT_CSP, tt)
    y = clf_final.predict(XV_CSP)
    yp = clf_final.predict_proba(XV_CSP)

    learner = {'csp_filt':csp_filters, 'lda':None, 'p0':None, 'p1':None, 'clf_final':clf_final, 'y':y, 'yp':yp, 't':tv }   
    
    acc = np.mean(y == tv)     
    # kappa = cohen_kappa_score(y, tv)
    return acc, learner

def sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs='250', filt='DFT', csp_list=None, csp_reg=None):
    n_bins = fh - fl
    overlap = 0.5
    step = n_bins/(nbands+1)
    size = step/overlap

    sub_bands = []        
    for i in range(nbands):
        fl_sb = i * step + fl
        fh_sb = i * step + size + fl
        sub_bands.append([fl_sb, fh_sb])
    # nbands = len(sub_bands)
    
    if filt == 'DFT': 
        ZTF = fft(ZT)
        REAL = np.transpose(np.real(ZTF), (2, 0, 1))
        IMAG = np.transpose(np.imag(ZTF), (2, 0, 1))
        ZTF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        ZTF = np.transpose(ZTF, (1, 2, 0))
        bin_size = 2 / ( Fs / ZT.shape[-1] ) # 2 representa sen e cos
        XT = [ ZTF[:, :, round(sub_bands[i][0]*bin_size):round(sub_bands[i][1]*bin_size)] for i in range(nbands) ]
        
        ZVF = fft(ZV)
        REAL = np.transpose(np.real(ZVF), (2, 0, 1))
        IMAG = np.transpose(np.imag(ZVF), (2, 0, 1))
        ZVF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        ZVF = np.transpose(ZVF, (1, 2, 0))
        bin_size = 2 / ( Fs / ZV.shape[-1] ) # 2 representa sen e cos
        XV = [ ZVF[:, :, round(sub_bands[i][0]*bin_size):round(sub_bands[i][1]*bin_size)] for i in range(nbands) ]
           
    elif filt == 'IIR':
        nyq = 0.5 * Fs
        XT, XV = [], []
        for i in range(nbands):
            low = sub_bands[i][0] / nyq
            high = sub_bands[i][1] / nyq
            if low == 0: low = 0.001
            if high >= 1: high = 0.99
            b, a = butter(5, [low,high], btype='bandpass') # iirfilter(5,[low,high],btype='band')
            XT.append(lfilter(b, a, ZT)) # filtfilt(b, a, ZT)
            XV.append(lfilter(b, a, ZV)) # filtfilt(b, a, ZV)
        
    if csp_list is None:
        # chain = [ Pipeline([('CSP', CSP(n_components=int(ncsp))), ('LDA', LDA())]) for i in range(nbands) ] # CSP 1
        chain = [ Pipeline([('CSP', mne.decoding.CSP(n_components=int(ncsp), reg=csp_reg)), ('LDA', LDA())]) for i in range(nbands) ] # CSP 2
    else: 
        # chain = [ Pipeline([('CSP', CSP(n_components=csp_list[i])), ('LDA', LDA())]) for i in range(nbands) ] # uncomment to tuning ncsp
        chain = [ Pipeline([('CSP', mne.decoding.CSP(n_components=csp_list[i], reg=csp_reg)), ('LDA', LDA())]) for i in range(nbands) ] # CSP 2
    
    for i in range(nbands): chain[i]['CSP'].fit(XT[i], tt) 
    
    XT_CSP = [ chain[i]['CSP'].transform(XT[i]) for i in range(nbands) ]
    
    SCORE_T = np.zeros((len(ZT), nbands))
    for i in range(nbands): 
        chain[i]['LDA'].fit(XT_CSP[i], tt)
        SCORE_T[:, i] = np.ravel(chain[i]['LDA'].transform(XT_CSP[i])) # classificações de cada época nas N sub bandas - auto validação
    
    class_ids = np.unique(tt)
    SCORE_T0 = SCORE_T[tt == class_ids[0], :]
    SCORE_T1 = SCORE_T[tt == class_ids[1], :]
    p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
    p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
    META_SCORE_T = np.log(p0.pdf(SCORE_T) / p1.pdf(SCORE_T))
    
    if clf['model'] == 'LDA': clf_final = LDA()
    elif clf['model'] == 'LR': clf_final = LogisticRegression()
    elif clf['model'] == 'SVM': clf_final = SVC(kernel=clf['kernel']['kf'], C=10**(clf['C']), gamma='scale', degree=3, probability=True)
    elif clf['model'] == 'KNN': clf_final = KNeighborsClassifier(n_neighbors=int(clf['neig']), metric=clf['metric'], p=3)                                       
    
    clf_final.fit(META_SCORE_T, tt)
    
    if csp_list is None:
        # csp_filters = [ chain[i]['CSP'].filters_ for i in range(nbands) ] # CSP 1
        csp_filters = [ chain[i]['CSP'].filters_[:int(ncsp)] for i in range(nbands) ] # CSP 2
    else: 
        # print(csp_list)
        csp_filters = [ chain[i]['CSP'].filters_[:csp_list[i]] for i in range(nbands) ]
    
    csps = [ chain[i]['CSP'] for i in range(nbands) ]
    ldas = [ chain[i]['LDA'] for i in range(nbands) ]
       
    XV_CSP = [ chain[i]['CSP'].transform(XV[i]) for i in range(nbands) ]
    SCORE_V = np.zeros((len(ZV), nbands))
    for i in range(nbands): 
        SCORE_V[:, i] = np.ravel(chain[i]['LDA'].transform(XV_CSP[i]))
    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
    
    y = clf_final.predict(META_SCORE_V)
    yp = clf_final.predict_proba(META_SCORE_V)
    
    # learner = {'csp_filt':csp_filters,'lda':ldas, 'p0':p0, 'p1':p1, 'clf_final':clf_final, 'y':y, 'yp':yp, 't':tv}
    learner = {'csp':csps, 'csp_filt':csp_filters, 'lda':ldas, 'p0':p0, 'p1':p1, 'clf_final':clf_final, 'y':y, 'yp':yp, 't':tv}
    
    acc = np.mean(y == tv)
    # kappa = cohen_kappa_score(y, tv)
    return acc, learner

def csp_tunning(args, data, events, info, class_ids, setup):
    # csp_reg = setup['reg']
    smin = math.floor(setup['tmin'] * info['fs'])
    smax = math.floor(setup['tmax'] * info['fs'])
    epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    epochs = nanCleaner(epochs)
    
    # kf = StratifiedShuffleSplit(1, test_size=0.5, random_state=42)
    # for _train, _test in kf.split(epochs, labels):
    #     ZT, ZV, tt, tv = epochs[_train], epochs[_test], labels[_train], labels[_test]
    
    cross_acc, cross_learner = [], []
    kf = StratifiedShuffleSplit(5, test_size=0.2, random_state=42)
    for _train, _test in kf.split(epochs, labels):
        ZT, ZV, tt, tv = epochs[_train], epochs[_test], labels[_train], labels[_test]
        a, l = sbcsp_approach(ZT, ZV, tt, tv, setup['nbands'], setup['fl'], setup['fh'], setup['ncomp'], setup['clf'], 
                              Fs=info['fs'], filt='DFT', csp_list=list(map(lambda x: int(x), args)))
        cross_acc.append(a); cross_learner.append(l)
    acc, _ = np.mean(cross_acc), cross_learner[np.argmax(cross_acc)]

    return acc * (-1)

def objective(args, data, events, info, class_ids):
    # filtering = {'design':'DFT'}
    # f_low, f_high, tmin, tmax, ncomp, nbands, reg = 0, 40, 0.5, 2.5, 8, 9, 1e-1
    # clf = {'model':'SVM','kernel':{'kf':'linear'},'C':-4}     
    f_low, f_high, tmin, tmax, ncsp, nbands, clf = args # reg
    f_low, f_high = int(f_low['fl']), int(f_high['fh']) # csp_reg = 10**(int(reg)) 
    # tmin = tmin['tmin']
    

    if nbands > (f_high - f_low): nbands = (f_high - f_low)
    while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
    
    # print(f'nbands {nbands}, fl {f_low}, fh {f_high}')
    
    # if nbands > (f_high - f_low):
    #     # print(f'cont {cont}')
    #     # print(f'nbands {f_high - f_low}')
    #     nbands = (f_high - f_low)
    
    # while (tmax-tmin)<1:
    #     # print(f'cont {cont}')
    #     # print(f'tmax {tmax}')
    #     tmax+=0.5 # garante janela minima de 1seg
    
    # print(f_high)
    
    smin = math.floor(tmin * info['fs'])
    smax = math.floor(tmax * info['fs'])
    epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    epochs = nanCleaner(epochs)
    
    cross_acc, cross_learner = [], []
    kf = StratifiedShuffleSplit(5, test_size=0.2, random_state=42)
    for _train, _test in kf.split(epochs, labels):
        ZT, ZV, tt, tv = epochs[_train], epochs[_test], labels[_train], labels[_test]
        a, l = sbcsp_approach(ZT, ZV, tt, tv, int(nbands), f_low, f_high, ncsp, clf, Fs=info['fs'], filt='DFT')     
        cross_acc.append(a); cross_learner.append(l)     
    acc, learner = np.mean(cross_acc), cross_learner[np.argmax(cross_acc)]
    
    H.loc[len(H)] = [f_low, f_high, tmin, tmax, ncsp, int(nbands), clf, acc,learner]
    
    return acc * (-1)

def tester(ZV, tv, Fs, class_ids, setup):
    fl, fh, nbands = setup['fl'], setup['fh'], setup['nbands']
    learner = setup['learner']
    csp = learner['csp']
    p0, p1, filters_, lda, clf_final = learner['p0'], learner['p1'], learner['csp_filt'], learner['lda'], learner['clf_final'] 
    # if nbands > (fh-fl): nbands = (fh-fl)
    # print(len(filters_), nbands)
    # ZV, tv = extractEpochs(data, events, int(tmin*Fs), int(tmax*Fs), class_ids)

    n_bins = fh - fl
    overlap = 0.5 
    step = n_bins / (nbands+1)
    size = step / overlap
    
    sub_bands = []        
    for i in range(nbands):
        fl_sb = i * step + fl
        fh_sb = i * step + size + fl
        sub_bands.append([fl_sb, fh_sb])
    
    ZVF = fft(ZV)
    REAL = np.transpose(np.real(ZVF), (2, 0, 1))
    IMAG = np.transpose(np.imag(ZVF), (2, 0, 1))
    ZVF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    ZVF = np.transpose(ZVF, (1, 2, 0))
    bin_size = 2/(Fs/ZV.shape[-1]) # 2 representa sen e cos
    XV = [ ZVF[:, :, round(sub_bands[i][0]*bin_size):round(sub_bands[i][1]*bin_size)] for i in range(nbands) ]
        
    # YV = [ np.asarray([ np.dot(filters_[i], ep) for ep in XV[i] ]) for i in range(nbands) ] # Spatial Filtering
    # XV_CSP = [ np.log(np.mean(YV[i]**2, axis=2)) for i in range(nbands) ] # Feature extraction
    # XV_CSP = [ np.log(np.var(YV[i], axis=2)) for i in range(nbands) ] # Feature extraction
    
    XV_CSP = [ csp[i].transform(XV[i]) for i in range(nbands) ]
    
    # print(np.asarray(XV_CSP).shape, np.asarray(XV_CSP_1).shape)
    # print(np.asarray(XV_CSP)[:,0,0:2], np.asarray(XV_CSP_1)[:,0,0:2])
    
    
    SCORE_V = np.asarray([ np.ravel(lda[i].transform(XV_CSP[i])) for i in range(nbands) ]).T # Score LDA  
    
    # print(SCORE_V.shape)
    
    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V)) # Bayesian Meta-score
    y = clf_final.predict(META_SCORE_V) # Final classification
    yp = clf_final.predict_proba(META_SCORE_V)
    acc = clf_final.score(META_SCORE_V, tv)
    return acc, y, yp


def get_features(ZV, Fs, setup, learner, filt):
    fl, fh, nbands = setup['fl'], setup['fh'], setup['nbands']
    csp = learner['csp']
    p0, p1, filters_, lda, clf_final = learner['p0'], learner['p1'], learner['csp_filt'], learner['lda'], learner['clf_final'] 

    n_bins = fh - fl
    overlap = 0.5 
    step = n_bins / (nbands+1)
    size = step / overlap
    
    sub_bands = []        
    for i in range(nbands):
        fl_sb = i * step + fl
        fh_sb = i * step + size + fl
        sub_bands.append([fl_sb, fh_sb])
    
    if filt == 'DFT': 
        ZVF = fft(ZV)
        REAL = np.transpose(np.real(ZVF), (2, 0, 1))
        IMAG = np.transpose(np.imag(ZVF), (2, 0, 1))
        ZVF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        ZVF = np.transpose(ZVF, (1, 2, 0))
        bin_size = 2/(Fs/ZV.shape[-1]) # 2 representa sen e cos
        
        XV = [ ZVF[:, :, round(sub_bands[i][0]*bin_size):round(sub_bands[i][1]*bin_size)] for i in range(nbands) ]

    elif filt == 'IIR':
        nyq = 0.5 * Fs
        XV = []
        for i in range(nbands):
            low = sub_bands[i][0] / nyq
            high = sub_bands[i][1] / nyq
            if low == 0: low = 0.001
            if high >= 1: high = 0.99
            b, a = butter(5, [low,high], btype='bandpass') # iirfilter(5,[low,high],btype='band')
            XV.append(lfilter(b, a, ZV)) # filtfilt(b, a, ZV)


    XV_CSP = [ csp[i].transform(XV[i]) for i in range(nbands) ] # nbands x len(XV) x ncsp
    
    SCORE_V = np.asarray([ np.ravel(lda[i].transform(XV_CSP[i])) for i in range(nbands) ]).T # Score LDA 
    
    XV_CSP = np.transpose(XV_CSP, (1, 0, 2))

    return XV_CSP, SCORE_V, sub_bands


def half_split_data(data, events, smin, smax, class_ids):
    ep, lb = extractEpochs(data, events, smin, smax, class_ids)
    et, lt = ep[:(len(ep)//2)], lb[:(len(ep)//2)] 
    ev, lv = ep[(len(ep)//2):], lb[(len(ep)//2):]
    ZT = np.r_[ et[np.where(lt == class_ids[0])], et[np.where(lt == class_ids[1])] ]
    ZV = np.r_[ ev[np.where(lv == class_ids[0])], ev[np.where(lv == class_ids[1])] ]
    tt = np.r_[ class_ids[0]*np.ones(len(ZT)//2), class_ids[1]*np.ones(len(ZT)//2)].astype(int)
    tv = np.r_[ class_ids[0]*np.ones(len(ZV)//2), class_ids[1]*np.ones(len(ZV)//2)].astype(int)
    return ZT, ZV, tt, tv

