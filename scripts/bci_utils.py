# -*- coding: utf-8 -*-
import re
import os
import mne
import math
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from time import time, sleep
from scipy.io import loadmat
from scipy.stats import norm
from datetime import datetime
from scipy.fftpack import fft
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression as LR
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter, decimate, welch
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold 

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)


def extractEpochs(data, events, smin, smax, class_ids):
    events_list = events[:, 1] # get class labels column
    cond = False
    for i in range(len(class_ids)): cond += (events_list == class_ids[i]) #get only class_ids pos in events_list
    idx = np.where(cond)[0]
    s0 = events[idx, 0] # get initial timestamps of each class epochs
    sBegin = s0 + smin
    sEnd = s0 + smax
    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    labels = events_list[idx]
    bad_epoch_list = []
    for i in range(n_epochs):
        epoch = data[:, sBegin[i]:sEnd[i]]
        if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete
        else:
            print('Incomplete epoch detected...', n_samples, '!=', epoch.shape[1])
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels
    
    
def nanCleaner(epoch):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoch.shape[0]):
        bad_idx = np.isnan(epoch[i, :])
        epoch[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoch[i, ~bad_idx])
    return epoch
    
    
def corrigeNaN(data):
    for ch in range(data.shape[0] - 1):
        this_chan = data[ch]
        data[ch] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(data[ch])
        meanChannel = np.nanmean(data[ch])
        data[ch, mask] = meanChannel
    return data


class Filter:
    def __init__(self, fl, fh, Fs, filtering, band_type='bandpass'):
        self.ftype = filtering['design']
        if self.ftype != 'DFT':
            nyq = 0.5*Fs
            low = fl/nyq
            high = fh/nyq        
            if low == 0: low = 0.001
            if high >= 1: high = 0.99
            if self.ftype == 'IIR':
                # self.b, self.a = iirfilter(filtering['iir_order'], [low, high], btype='band')
                self.b, self.a = butter(filtering['iir_order'], [low, high], btype=band_type)
            elif self.ftype == 'FIR':
                self.b, self.a = firwin(filtering['fir_order'], [low, high], window='hamming', pass_zero=False), [1]

    def apply_filter(self, X, is_epoch=False):
        if self.ftype != 'DFT': XF = lfilter(self.b, self.a, X) # lfilter, filtfilt
        else:
            XF = fft(X)
            if is_epoch:
                real, imag = np.real(XF).T, np.imag(XF).T
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))))
            else:
                real = np.transpose(np.real(XF), (2, 0, 1))
                imag = np.transpose(np.imag(XF), (2, 0, 1))
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))), (1, 2, 0)) 
        return XF


class CSP:
    def __init__(self, n_components):
        self.n_components = n_components
        self.filters_ = None
    def fit(self, X, t):
        ch = X.shape[1]
        class_ids = np.unique(t)   
        X1 = X[class_ids[0] == t]
        X2 = X[class_ids[1] == t]
        S1, S2 = np.zeros((ch, ch)), np.zeros((ch, ch))  
        for i in range(len(X1)): S1 += np.dot(X1[i], X1[i].T) / X1[i].shape[-1] # cov X[i]
        for i in range(len(X2)): S2 += np.dot(X2[i], X2[i].T) / X2[i].shape[-1] # ...sum((X*X.T)/q)
        S1 /= len(X1); 
        S2 /= len(X2)
        [D, W] = eigh(S1, S1 + S2) # + 1e-10 * np.eye(22))
        ind = np.empty(ch, dtype=int)
        ind[0::2] = np.arange(ch-1, ch//2 - 1, -1) 
        ind[1::2] = np.arange(0, ch//2)
        # W += 1e-1 * np.eye(22)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self # used on cross-validation pipeline
    def transform(self, X):        
        Y = np.asarray([np.dot(self.filters_, ep) for ep in X])
        # FEAT = np.log(np.mean(Y**2, axis=2))
        FEAT = np.log(np.var(Y, axis=2))
        return FEAT


class BCI:    
    def __init__(self, data=None, events=None, class_ids=[1,2], fs=250, overlap=True, crossval=False, nfolds=10, test_perc=0.5, is_sbcsp=False,
                 f_low=8, f_high=30, tmin=0.5, tmax=2.5, ncsp=8, nbands=None, ap={'option':'classic'}, filtering={'design':'IIR', 'iir_order':5}, 
                 clf={'model':'LDA'}, ncsp_list=None, filt_type='IIR', filt_order=5):
        self.data = data
        self.events = events
        self.class_ids = class_ids
        self.overlap = overlap
        self.fs = fs
        self.crossval = crossval
        self.nfolds = nfolds
        self.test_perc = test_perc
        self.f_low = f_low
        self.f_high = f_high
        self.tmin = tmin
        self.tmax = tmax
        self.ncsp = ncsp
        self.nbands = nbands
        self.is_sbcsp = is_sbcsp
        self.ap = ap
        self.filtering = filtering
        self.filt_type = filt_type
        self.filt_order = filt_order
        self.clf = clf
        self.acc = None
        self.kpa = None
        self.ncsp_list = ncsp_list
        self.learner = None
        
        try:
            while (self.tmax-self.tmin)<1: self.tmax+=0.5
            smin, smax = int(self.tmin * self.fs), int(self.tmax * self.fs)
            # self.bsize = 2/(self.fs/(smax-smin)) # 2=sen/cos complexo fft
            self.epochs, self.labels = extractEpochs(self.data, self.events, smin, smax, self.class_ids)
            self.epochs = nanCleaner(self.epochs)
        except: pass
    
    def design_clf(self):
        if self.clf['model'] == 'LR': self.clf_final = LR()
        elif self.clf['model'] == 'LDA': self.clf_final = LDA()
        elif self.clf['model'] == 'Bayes': self.clf_final = GaussianNB()
        elif self.clf['model'] == 'SVM': 
            # degree = self.clf['kernel']['degree'] if self.clf['kernel']['kf'] == 'poly' else 3
            # gamma = self.clf['gamma'] if self.clf['gamma'] in ['scale', 'auto'] else 10 ** (self.clf['gamma']['gamma_float'])
            # C=10**(self.clf['C']) # para C discretizado
            self.clf_final = SVC(kernel=self.clf['kernel']['kf'], C=10**(self.clf['C']), gamma='scale', degree=3, probability=True)
        elif self.clf['model'] == 'KNN':   
            self.clf_final = KNeighborsClassifier(n_neighbors=int(self.clf['neig']), metric=self.clf['metric'], p=3) # p=self.clf['p']                                       
        elif self.clf['model'] == 'DTree':
            # if self.clf['min_split'] == 1.0: self.clf['min_split'] += 1
            # max_depth = self.clf['max_depth'] if self.clf['max_depth'] is None else int(self.clf['max_depth']['max_depth_int'])
            # min_samples_split = self.clf['min_split'] # math.ceil(self.clf['min_split']), # profundidade maxima da arvore - representa a poda;
            self.clf_final = DecisionTreeClassifier(criterion=self.clf['crit'], random_state=0, max_depth=None, min_samples_split=2)       
        elif self.clf['model'] == 'MLP':   
            self.clf_final = MLPClassifier(verbose=False, max_iter=10000, learning_rate_init=10**self.clf['eta'], activation=self.clf['activ']['af'],  
                                           # learning_rate='constant', # solver=self.clf['mlp_solver'], alpha=10**self.clf['alpha'],
                                           hidden_layer_sizes=(int(self.clf['n_neurons']), int(self.clf['n_hidden']))) 

    def evaluate(self):  
        if self.crossval:
            self.cross_scores, self.cross_kappa = [], []
            kf = StratifiedShuffleSplit(self.nfolds, test_size=self.test_perc, random_state=42)
            # kf = StratifiedKFold(self.nfolds, False)
            # self.chain = Pipeline([('CSP', CSP(...)), ('SVC', self.clf_final)])
            for train, test in kf.split(self.epochs, self.labels):
                ZT, ZV, tt, tv = self.epochs[train], self.epochs[test], self.labels[train], self.labels[test]
                acc_fold, kappa_fold, _ = self.sbcsp_chain(ZT, ZV, tt, tv) if self.is_sbcsp else self.classic_chain(ZT, ZV, tt, tv)     
                self.cross_scores.append(acc_fold) # self.cross_scores.append(self.chain.score(XV, yV))
                self.cross_kappa.append(kappa_fold)
            self.acc, self.kappa = np.mean(self.cross_scores), np.mean(self.cross_kappa)
        else:
            test_size = int(len(self.epochs) * self.test_perc)
            train_size = int(len(self.epochs) - test_size)
            train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
            ZT, tt = self.epochs[:train_size], self.labels[:train_size] 
            ZV, tv = self.epochs[train_size:], self.labels[train_size:]
            ZT = np.vstack([ZT[np.where(tt == k)] for k in self.class_ids]) 
            ZV = np.vstack([ZV[np.where(tv == k)] for k in self.class_ids])
            tt = np.hstack([ np.ones(len(ZT)//2)*k for k in self.class_ids ]).astype(int)
            tv = np.hstack([ np.ones(len(ZV)//2)*k for k in self.class_ids ]).astype(int)
            self.acc, self.kappa, self.learner = self.sbcsp_chain(ZT, ZV, tt, tv) if self.is_sbcsp else self.classic_chain(ZT, ZV, tt, tv)
    
    
    def classic_chain(self, ZT, ZV, tt, tv):
        self.filt = Filter(self.f_low, self.f_high, self.fs, self.filtering)
        XT = self.filt.apply_filter(ZT)
        XV = self.filt.apply_filter(ZV)
        if self.filtering['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
            bsize = 2/(self.fs/ZT.shape[-1])  # 2==sen/cos complexo fft intercalados / resol freq
            XT = XT[:, :, round(self.f_low * bsize):round(self.f_high * bsize)]
            bsize = 2/(self.fs/ZT.shape[-1])  # 2==sen/cos complexo fft intercalados / resol freq
            XV = XV[:, :, round(self.f_low * bsize):round(self.f_high * bsize)]
        self.csp = mne.decoding.CSP(n_components=int(self.ncsp)) # mne.decoding.CSP(...)
        self.csp.fit(XT, tt)
        csp_filters = self.csp.filters_
        FT, FV = self.csp.transform(XT), self.csp.transform(XV)
        self.design_clf()
        self.clf_final.fit(FT, tt)
        y = self.clf_final.predict(FV)
        yp = self.clf_final.predict_proba(FV)
        learner = {'csp':self.csp, 'csp_filt':csp_filters, 'lda':None, 'p0':None, 'p1':None, 'clf_final':self.clf_final, 'y':y, 'yp':yp, 't':tv}
        acc, kappa = np.mean(y == tv), cohen_kappa_score(y, tv)
        return acc, kappa, learner
        
    
    def sbcsp_chain(self, ZT, ZV, tt, tv):
        # nbands = int(self.f_high-self.f_low) if self.ap['nbands'] > (self.f_high-self.f_low) else int(self.ap['nbands'])
        nbands = int(self.f_high-self.f_low) if self.nbands > (self.f_high-self.f_low) else int(self.nbands)
        
        if self.overlap: step = (self.f_high - self.f_low) / (nbands+1); size = step/0.5 # overlap=0.5
        else: step = (self.f_high - self.f_low)/nbands; size = step
        
        sub_bands = []        
        for i in range(nbands):
            fl_sb = i * step + self.f_low
            fh_sb = i * step + size + self.f_low
            sub_bands.append([fl_sb, fh_sb])
        
        XT, XV = [], []
        if self.filtering['design'] == 'DFT':
            self.filt = Filter(self.f_low, self.f_high, self.fs, self.filtering)
            XTF = self.filt.apply_filter(ZT)
            XVF = self.filt.apply_filter(ZV)
            for i in range(nbands):
                bsize = 2/(self.fs/ZT.shape[-1])  # 2==sen/cos complexo fft intercalados / resol freq
                XT.append(XTF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])
                bsize = 2/(self.fs/ZV.shape[-1])  # 2==sen/cos complexo fft intercalados / resol freq
                XV.append(XVF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])
        elif self.filtering['design'] in ['IIR' or 'FIR']:
            for i in range(nbands):
                filt = Filter(sub_bands[i][0], sub_bands[i][1], self.fs, self.filtering)
                XT.append(filt.apply_filter(ZT))
                XV.append(filt.apply_filter(ZV))
        
        if self.ncsp_list is None: self.chain = [ Pipeline([('CSP', mne.decoding.CSP(n_components=int(self.ncsp))), ('LDA', LDA())]) for i in range(nbands) ] # mne.decoding.
        else: self.chain = [ Pipeline([('CSP', mne.decoding.CSP(n_components=self.ncsp_list[i])), ('LDA', LDA())]) for i in range(nbands) ] # to tuning ncsp
        
        for i in range(nbands): self.chain[i]['CSP'].fit(XT[i], tt)  
        FT = [ self.chain[i]['CSP'].transform(XT[i]) for i in range(nbands) ]
        FV = [ self.chain[i]['CSP'].transform(XV[i]) for i in range(nbands) ]
        ST = np.zeros((len(ZT), nbands))
        SV = np.zeros((len(ZV), nbands))
        for i in range(nbands): 
            self.chain[i]['LDA'].fit(FT[i], tt)
            ST[:, i] = np.ravel(self.chain[i]['LDA'].transform(FT[i]))  # classificações de cada época nas N sub bandas - auto validação
            SV[:, i] = np.ravel(self.chain[i]['LDA'].transform(FV[i]))
        
        if self.ncsp_list is None: csp_filters = [ self.chain[i]['CSP'].filters_[:int(self.ncsp)] for i in range(nbands) ] # chain[i]['CSP'].filters_
        else: csp_filters = [ self.chain[i]['CSP'].filters_[:self.ncsp_list[i]] for i in range(nbands) ]
                           
        csps = [ self.chain[i]['CSP'] for i in range(nbands) ]
        ldas = [ self.chain[i]['LDA'] for i in range(nbands) ] 
        
        ST0 = ST[tt == self.class_ids[0], :]
        ST1 = ST[tt == self.class_ids[1], :]
        self.p0 = norm(np.mean(ST0, axis=0), np.std(ST0, axis=0))
        self.p1 = norm(np.mean(ST1, axis=0), np.std(ST1, axis=0))
        META_ST = np.log(self.p0.pdf(ST) / self.p1.pdf(ST))
        META_SV = np.log(self.p0.pdf(SV) / self.p1.pdf(SV))
        
        self.design_clf()
        self.clf_final.fit(META_ST, tt)
        y = self.clf_final.predict(META_SV)
        yp = self.clf_final.predict_proba(META_SV)
        acc, kappa = np.mean(y == tv), cohen_kappa_score(y, tv)
        learner = {'csp':csps, 'csp_filt':csp_filters, 'lda':ldas, 'p0':self.p0, 'p1':self.p1, 'clf_final':self.clf_final, 'y':y, 'yp':yp, 't':tv}
        return acc, kappa, learner
