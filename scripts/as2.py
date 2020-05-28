# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import pickle
import numpy as np
import pandas as pd
from time import time, sleep
from hyperopt import base, fmin, tpe, rand, hp, space_eval
from hyperopt.fmin import generate_trials_to_calculate
import matplotlib.pyplot as plt
##%% #########################################################
import re
import mne
import math
import warnings
import itertools
from sklearn.svm import SVC
from scipy.io import loadmat
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter, decimate, welch
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression

from functools import partial

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

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
            print('Incomplete epoch detected...')
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels

class Filter:
    def __init__(self, fl, fh, srate, filt_info, forder=5, band_type='bandpass'):
        self.ftype = filt_info['design']
        if self.ftype != 'DFT':
            self.nyq = 0.5 * srate
            low = fl / self.nyq
            high = fh / self.nyq        
            if low == 0: low = 0.001
            if high >= 1: high = 0.99
            if self.ftype == 'IIR':
                self.forder = filt_info['iir_order']
                # self.b, self.a = iirfilter(self.forder, [low, high], btype='band')
                self.b, self.a = butter(self.forder, [low, high], btype=band_type)
            elif self.ftype == 'FIR':
                self.forder = filt_info['fir_order']
                self.b = firwin(self.forder, [low, high], window='hamming', pass_zero=False)
                self.a = [1]

    def apply_filter(self, X, is_epoch=False):
        if self.ftype != 'DFT': XF = lfilter(self.b, self.a, X) # lfilter, filtfilt
        else:
            XF = fft(X)
            if is_epoch:
                real, imag = np.real(XF).T, np.imag(XF).T
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))))
            else:
                real, imag = np.transpose(np.real(XF), (2, 0, 1)), np.transpose(np.imag(XF), (2, 0, 1))
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))), (1, 2, 0)) 
        return XF

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

class BCI():
    def __init__(self, data=None, events=None, class_ids=[1,2], fs=250, overlap=True, crossval=False, nfolds=10, test_perc=0.5, 
                 f_low=None, f_high=None, tmin=None, tmax=None, ncomp=None, nbands=None, ap=None, filt_info=None, clf=None, split='common'):
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
        self.ncomp = ncomp
        self.nbands = nbands
        self.ap = ap
        self.filt_info = filt_info
        self.clf = clf
        self.acc = None
        self.kappa = None
        self.clf_params = None
        self.csp_list = None
        self.split = split
    
    def evaluate(self): 
        if self.clf['model'] == 'LDA': self.clf_final = LDA()
            # lda_shrinkage = None
            # if not (clf_dict['lda_solver'] == 'svd'): 
            #     lda_shrinkage = self.clf['shrinkage'] if self.clf['shrinkage'] in [None,'auto'] else self.clf['shrinkage']['shrinkage_float']
            # self.clf_final = LDA(solver=self.clf['lda_solver'], shrinkage=lda_shrinkage)
        elif self.clf['model'] == 'Bayes': self.clf_final = GaussianNB()
        
        elif self.clf['model'] == 'SVM': 
            # degree = self.clf['kernel']['degree'] if self.clf['kernel']['kf'] == 'poly' else 3
            # gamma = self.clf['gamma'] if self.clf['gamma'] in ['scale', 'auto'] else 10 ** (self.clf['gamma']['gamma_float'])
            self.clf_final = SVC(kernel=self.clf['kernel']['kf'], C=10**(self.clf['C']), gamma='scale', degree=3, probability=True)
        elif self.clf['model'] == 'KNN':   
            self.clf_final = KNeighborsClassifier(n_neighbors=int(self.clf['neig']), metric=self.clf['metric'], p=3) # p=self.clf['p']                                       
        elif self.clf['model'] == 'DTree':
            # if self.clf['min_split'] == 1.0: self.clf['min_split'] += 1
            # max_depth = self.clf['max_depth'] if self.clf['max_depth'] is None else int(self.clf['max_depth']['max_depth_int'])
            # min_samples_split = self.clf['min_split'] # math.ceil(self.clf['min_split']), # profundidade maxima da arvore - representa a poda;
            self.clf_final = DecisionTreeClassifier(criterion=self.clf['crit'], random_state=0, max_depth=None, min_samples_split=2)       
        elif self.clf['model'] == 'MLP':   
            self.clf_final = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, 
                                           learning_rate_init=10**self.clf['eta'], activation=self.clf['activ']['af'],  
                                           learning_rate='constant', # solver=self.clf['mlp_solver'], alpha=10**self.clf['alpha'],
                                           hidden_layer_sizes=(int(self.clf['n_neurons']), int(self.clf['n_hidden']))) 
        elif self.clf['model'] == 'LR': self.clf_final = LogisticRegression()
        
        while (self.tmax-self.tmin)<1: self.tmax+=0.5
        smin = math.floor(self.tmin * self.fs)
        smax = math.floor(self.tmax * self.fs)
        # print(smax-smin)
        self.res_freq = self.fs/(smax-smin) # rf=Fs/Q
        self.dft_size = 2/self.res_freq # 2=sen/cos complexo fft
        self.epochs, self.labels = extractEpochs(self.data, self.events, smin, smax, self.class_ids)
        self.epochs = nanCleaner(self.epochs)
        # print(self.epochs.shape)
        # self.epochs = np.asarray([ nanCleaner(ep) for ep in self.epochs ])
        
        # if self.split == 'teste':
        #     test_size = int(len(self.epochs) * 0.5)
        #     train_size = int(len(self.epochs) - test_size)
        #     train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
        #     epochsV, labelsV = self.epochs[train_size:], self.labels[train_size:]
        #     self.epochs, self.labels = self.epochs[:train_size], self.labels[:train_size] 
            

        if self.crossval:
            self.cross_scores, self.cross_kappa, self.cross_clfs_params = [], [], []
            kf = StratifiedShuffleSplit(self.nfolds, test_size=self.test_perc, random_state=42)
            # kf = StratifiedKFold(self.nfolds, False)
            # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
            # self.cross_scores = cross_val_score(self.chain, XF, self.labels, cv=kf)
            for idx_treino, idx_teste in kf.split(self.epochs, self.labels):
                XT, XV, yT, yV = self.epochs[idx_treino], self.epochs[idx_teste], self.labels[idx_treino], self.labels[idx_teste]
                # if self.split == 'teste': XV, yV = epochsV, labelsV   
                # print(np.asarray(XT).shape, np.asarray(XV).shape)
                acc_fold, kappa_fold, params_fold = self.classic_approach(XT, XV, yT, yV) if (self.ap['option'] == 'classic') else self.sbcsp_approach(XT, XV, yT, yV)     
                self.cross_scores.append(acc_fold) # self.cross_scores.append(self.chain.score(XV, yV))
                self.cross_kappa.append(kappa_fold)
                self.cross_clfs_params.append(params_fold) # + lambda np.eye(22) 
                
            idx_max = np.argmax(self.cross_scores)
            # print(max(self.cross_scores), np.mean(self.cross_scores))
            # self.acc, self.kappa, self.clf_params = max(self.cross_scores), max(self.cross_kappa), self.cross_clfs_params[idx_max]
            self.acc, self.kappa, self.clf_params = np.mean(self.cross_scores), np.mean(self.cross_kappa), self.cross_clfs_params[idx_max]
        else:
            test_size = int(len(self.epochs) * self.test_perc)
            train_size = int(len(self.epochs) - test_size)
            train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
            epochsT, labelsT = self.epochs[:train_size], self.labels[:train_size] 
            epochsV, labelsV = self.epochs[train_size:], self.labels[train_size:]
            
            ET = [ epochsT[np.where(labelsT == i)] for i in self.class_ids ] # Extrair épocas de cada classe
            EV = [ epochsV[np.where(labelsV == i)] for i in self.class_ids ]
            XA = np.r_[ET[0], EV[0]] # class A only
            XB = np.r_[ET[1], EV[1]] # class B only
                        
            if self.split == 'common':
                XT = np.concatenate([ET[0],ET[1]]) # Train data classes A + B
                XV = np.concatenate([EV[0],EV[1]]) # Test data classes A + B 
                        
            if self.split == 'as_train':
                XT = np.r_[XA[:58], XB[:58]]
                XV = np.r_[XA[58:86], XB[58:86]]
            
            if self.split == 'as_test': 
                XT = np.r_[XA[:58], XB[:58]]
                XV = np.r_[XA[86:], XB[86:]]
                
            if self.split == 'common1':
                XT = np.concatenate([ET[0][:36],ET[1][:36]]) # Train data classes A + B
                XV = np.concatenate([EV[0],EV[1]]) # Test data classes A + B 
                  
            # print(np.asarray(XT).shape, np.asarray(XV).shape)
            yT = np.concatenate([self.class_ids[0] * np.ones(int(len(XT)/2)), self.class_ids[1] * np.ones(int(len(XT)/2))])
            yV = np.concatenate([self.class_ids[0] * np.ones(int(len(XV)/2)), self.class_ids[1] * np.ones(int(len(XV)/2))])
            self.acc, self.kappa, self.clf_params = self.classic_approach(XT, XV, yT, yV) if (self.ap['option'] == 'classic') else self.sbcsp_approach(XT, XV, yT, yV)
    
    def classic_approach(self, XT, XV, yT, yV):

        self.filt = Filter(self.f_low, self.f_high, self.fs, self.filt_info)
        XTF = self.filt.apply_filter(XT)
        XVF = self.filt.apply_filter(XV)
        if self.filt_info['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
            bmin = round(self.f_low * self.dft_size)
            bmax = round(self.f_high * self.dft_size)
            XTF = XTF[:, :, bmin:bmax]
            XVF = XVF[:, :, bmin:bmax]
        
        self.csp = CSP(n_components=int(self.ncomp))
        
        # # Option 1:
        self.csp.fit(XTF, yT)
        self.csp_filters = self.csp.filters_
        XT_CSP = self.csp.transform(XTF)
        XV_CSP = self.csp.transform(XVF) 
        self.clf_final.fit(XT_CSP, yT)
        self.scores = self.clf_final.predict(XV_CSP)
        y_proba = self.clf_final.predict_proba(XV_CSP)
        
        # # Option 2:
        # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
        # self.chain.fit(XT, yT)
        # self.csp_filters = self.chain['CSP'].filters_
        # self.scores = self.chain.predict(XV)
        
        classifier = {'csp_filt':self.csp_filters, 'lda':None, 'y':self.scores,
                      'yp':y_proba, 'p0':None, 'p1':None, 'clf_final':self.clf_final}
        
        acc = np.mean(self.scores == yV)     
        kappa = cohen_kappa_score(self.scores, yV)
        return acc, kappa, classifier
        
    def sbcsp_approach(self, XT, XV, yT, yV):
        nbands = int(self.ap['nbands'])
        if nbands > (self.f_high - self.f_low): nbands = (self.f_high - self.f_low)
        # print(nbands)
        
        n_bins = self.f_high - self.f_low
        # overlap =  if self.overlap else 1
        if self.overlap: 
            step = n_bins/(nbands+1)
            size = step/0.5 # overlap=0.5
        else:
            step = n_bins/nbands
            size = step
        
        sub_bands, bins = [], []        
        for i in range(nbands):
            fl_sb = i * step + self.f_low
            fh_sb = i * step + size + self.f_low
            # if fh_sb <= self.f_high: sub_bands.append([fl_sb, fh_sb]) # extrapola limite superior 1: descarta última sub-banda 
            # if fh_sb > self.f_high: fh_sb = self.f_high # extrapola limite superior 2: ajusta f_high ao limite
            sub_bands.append([fl_sb, fh_sb])
        # print(sub_bands)
        nbands = len(sub_bands)
        
        XTF, XVF = [], []
        if self.filt_info['design'] == 'DFT':
            self.filt = Filter(self.f_low, self.f_high, self.fs, self.filt_info)
            XT_FFT = self.filt.apply_filter(XT)
            XV_FFT = self.filt.apply_filter(XV)
            for i in range(nbands):
                bmin = round(sub_bands[i][0] * self.dft_size)
                bmax = round(sub_bands[i][1] * self.dft_size)
                XTF.append(XT_FFT[:, :, bmin:bmax])
                XVF.append(XV_FFT[:, :, bmin:bmax])
                bins.append([bmin,bmax])
            # print(bins)
        elif self.filt_info['design'] in ['IIR' or 'FIR']:
            for i in range(nbands):
                filt = Filter(sub_bands[i][0], sub_bands[i][1], self.fs, self.filt_info)
                XTF.append(filt.apply_filter(XT))
                XVF.append(filt.apply_filter(XV))
        
        # # Option 1:
        # print(self.csp_list)
        if self.csp_list is None:
            self.chain = [ Pipeline([('CSP', CSP(n_components=int(self.ncomp))), ('LDA', LDA())]) for i in range(nbands) ]
            # self.chain = [ Pipeline([('CSP', mne.decoding.CSP(n_components=int(self.ncomp), reg=None)), ('LDA', LDA())]) for i in range(nbands) ]
        else:
            self.chain = [ Pipeline([('CSP', CSP(n_components=self.csp_list[i])), ('LDA', LDA())]) for i in range(nbands) ] # uncomment to tuning ncsp
        
        for i in range(nbands): self.chain[i]['CSP'].fit(XTF[i], yT)  
        XT_CSP = [ self.chain[i]['CSP'].transform(XTF[i]) for i in range(nbands) ]
        XV_CSP = [ self.chain[i]['CSP'].transform(XVF[i]) for i in range(nbands) ]
        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        for i in range(nbands): 
            self.chain[i]['LDA'].fit(XT_CSP[i], yT)
            SCORE_T[:, i] = np.ravel(self.chain[i]['LDA'].transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(self.chain[i]['LDA'].transform(XV_CSP[i]))
        csp_filters_sblist = [ self.chain[i]['CSP'].filters_ for i in range(nbands) ]
        lda_sblist = [ self.chain[i]['LDA'] for i in range(nbands) ] 
        
        # # Option 2:
        # SCORE_T = np.zeros((len(XT), nbands))
        # SCORE_V = np.zeros((len(XV), nbands))
        # self.csp_filters_sblist = []
        # self.lda_sblist = []
        # for i in range(nbands):
        #     self.chain = Pipeline([('CSP', CSP(n_components=self.ncomp)), ('LDA', LDA()) ])
        #     self.chain['CSP'].fit(XTF, yT)
        #     XT_CSP = self.chain['CSP'].transform(XTF)
        #     XV_CSP = self.chain['CSP'].transform(XVF)
        #     self.chain['LDA'].fit(XT_CSP, yT)
        #     SCORE_T[:, i] = np.ravel(self.chain['LDA'].transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
        #     SCORE_V[:, i] = np.ravel(self.chain['LDA'].transform(XV_CSP))
        #     self.csp_filters_sblist.append(self.chain['CSP'].filters_)
        #     self.lda_sblist.append(self.chain['LDA'])
        
        SCORE_T0 = SCORE_T[yT == self.class_ids[0], :]
        SCORE_T1 = SCORE_T[yT == self.class_ids[1], :]
        self.p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
        self.p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
        META_SCORE_T = np.log(self.p0.pdf(SCORE_T) / self.p1.pdf(SCORE_T))
        META_SCORE_V = np.log(self.p0.pdf(SCORE_V) / self.p1.pdf(SCORE_V))
        self.clf_final.fit(META_SCORE_T, yT)
        self.scores = self.clf_final.predict(META_SCORE_V)
        y_proba = self.clf_final.predict_proba(META_SCORE_V)
        
        classifier = {'csp_filt':csp_filters_sblist, 'lda':lda_sblist, 'y':self.scores,
                      'yp':y_proba, 'p0':self.p0, 'p1':self.p1, 'clf_final':self.clf_final}
        
        acc = np.mean(self.scores == yV)
        kappa = cohen_kappa_score(self.scores, yV)
        return acc, kappa, classifier

##%% #############################################################################
# bci = BCI()
# bci2 = BCI()
H = pd.DataFrame(columns=['fl','fh','tmin','tmax','ncsp','nbands','clf_model','csp','lda','p0','p1','clf_final','y','yp','acc'])

# def objective_tune(args_tune):
#     # print(args_tune)
#     bci2.csp_list = list(map(lambda x: int(x), args_tune))
#     bci2.evaluate()
#     return bci2.acc * (-1)

def objective(args, subj, classes):
    bci = None
    bci = BCI()
    
    bci.data, bci.events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0'+str(suj)+'T.npy', allow_pickle=True)
    bci.class_ids = class_ids  
    bci.fs = info['fs'] 
    bci.overlap = True
    bci.crossval = False
    bci.nfolds = 5 
    bci.test_perc = 0.2 if bci.crossval else 0.5
    bci.split = 'common' # common, as_train, as_test
    bci.filt_info = {'design':'DFT'} 
    # bci.clf = {'model':'SVM','kernel':{'kf':'linear'},'C':-4} 
    # bci.ap = {'option':'sbcsp', 'nbands':9}
    # bci.f_low, bci.f_high = 4, 40
    # bci.tmin, bci.tmax = 0.5, 2.5
    # bci.ncomp = 8
    
    f_low, f_high, bci.tmin, bci.tmax, ncomp, nbands, bci.clf = args # 
    if nbands > (f_high - f_low): nbands = (f_high - f_low)
    bci.ap = {'option': 'sbcsp', 'nbands': nbands}
    bci.f_low, bci.f_high, bci.ncomp = int(f_low), int(f_high), int(ncomp)
    while (bci.tmax-bci.tmin)<1: bci.tmax+=0.5 # garante janela minima de 1seg
    
    bci.evaluate()
    
    # print(args) # subj,classes,bci.acc 
    
    H.loc[len(H)] = [bci.f_low, bci.f_high, bci.tmin, bci.tmax, bci.ncomp, nbands, bci.clf['model'], 
                     bci.clf_params['csp_filt'], bci.clf_params['lda'], bci.clf_params['p0'], bci.clf_params['p1'], 
                     bci.clf_params['clf_final'], bci.clf_params['y'],bci.clf_params['yp'], bci.acc]

    return bci.acc * (-1)

def teste(h, suj, class_ids):
    data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0'+str(suj)+'E.npy', allow_pickle=True)
    Fs = info['fs']
    f_low, f_high, ncsp, tmin, tmax, nbands = h['fl'], h['fh'], h['ncsp'], h['tmin'], h['tmax'], h['nbands']
    p0, p1 = h['p0'], h['p1']
    csp_filters = h['csp']
    lda = h['lda']
    clf_final = h['clf_final']
    clf_model = h['clf_model']
    
    # if nbands > (f_high - f_low): nbands = (f_high - f_low)
    
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
    bin_size = 2/dft_resf # 2 representa sen e cos
    nyq = 0.5 * Fs
    
    sb_ranges = []
    for i in range(nbands):
        fl_sb = i * step + f_low
        fh_sb = i * step + size + f_low
        sb_ranges.append([fl_sb, fh_sb])
        
    # nbands = len(sb_ranges)
    
    sb_dft = []
    for i in range(nbands):
        bmin = round(sb_ranges[i][0] * bin_size)
        bmax = round(sb_ranges[i][1] * bin_size)
        sb_dft.append([bmin,bmax])
        
    ZF = fft(Z)
    REAL = np.transpose(np.real(ZF), (2, 0, 1))
    IMAG = np.transpose(np.imag(ZF), (2, 0, 1))
    ZF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
    ZF = np.transpose(ZF, (1, 2, 0))
    X = [ ZF[:, :, sb_dft[i][0]:sb_dft[i][1]] for i in range(nbands) ]
    
    # print(len(csp_filters), nbands)
    
    Y = [ np.asarray([ np.dot(csp_filters[i], ep) for ep in X[i] ]) for i in range(nbands) ] # Spatial Filtering

    XCSP = [ np.log(np.mean(Y[i]**2, axis=2)) for i in range(nbands) ] # Feature extraction
    # XCSP = [ np.log(np.var(Y[i], axis=2)) for i in range(nbands) ] # Feature extraction
    
    score = np.dot(XCSP[0] - lda[0].xbar_, lda[0].scalings_)
    
    SCORE = np.asarray([ np.ravel(lda[i].transform(XCSP[i])) for i in range(nbands) ]).T # Score LDA
        
    META_SCORE = np.log(p0.pdf(SCORE) / p1.pdf(SCORE)) # Bayesian Meta-score
    
    # y = META_SCORE @ clf_final.coef_.T # y<0=classe A(1) ; y>0=classe B(2)
    y_labels = clf_final.predict(META_SCORE) # Final classification
    y_proba = clf_final.predict_proba(META_SCORE)
    
    acc = clf_final.score(META_SCORE, t)
    return acc, y_labels, y_proba, t

#%%
if __name__ == "__main__":
    n_iter = 20   
    subjects = range(1,10) 
    classes = [[1, 2]]#, [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] # 
    R = pd.DataFrame(columns=['subj','A','B','tmin','tmax','fl','fh','ncsp','nbands','clf','clf_details','as_train',
                              'as_max','as_mode','as_pmean','as_mlr','as_best','sb_dft','sb_iir','cla_dft','cla_iir'])
    for suj in subjects:
        for class_ids in classes:
            H = H.iloc[0:0] # cleaning df
            print(f'###### {suj} {class_ids} ######')        
            space = (
                hp.uniformint('fl', 0, 15),
                hp.uniformint('fh', 25, 40),
                hp.quniform('tmin', 0, 2, 0.5),
                hp.quniform('tmax', 2, 4, 0.5),
                # hp.quniform('ncomp', 2, 10, 2), 
                hp.choice('ncomp', [2,4,6,8,22]),
                hp.uniformint('nbands', 1, 25),
                hp.pchoice('clf', [
                    (0.2, {'model':'LDA'}),
                    (0.4, {'model':'SVM','C': hp.quniform('C', -6, 0, 1), # hp.loguniform('C', -6, 0), # 
                      'kernel':hp.choice('kernel',[{'kf':'linear'},{'kf':'poly'},{'kf':'sigmoid'},{'kf':'rbf'}])}),
                    (0.2, {'model':'KNN','neig':hp.uniformint('neig', 5, 20),
                      'metric':hp.choice('metric',['euclidean','manhattan','minkowski','chebyshev'])}),
                    (0.2, {'model':'LR'})
                    ])
                )
            class_ids0 = np.copy(class_ids)
            
            path = '../as_results/sbrt20/IV2a/'
            path_to_trials = path + 'A0' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
            if not os.path.isdir(path): os.makedirs(path)
            
            # acc_train = -1
            # try:
            #     trials = pickle.load(open(path_to_trials, 'rb'))
            #     acc_train = ((-1) * trials.best_trial['result']['loss'])
            # except: trials = base.Trials()
            trials = base.Trials()  
            # trials = generate_trials_to_calculate(init_vals)
            init_vals = [{'fl':4,'fh':40,'tmin':0.5,'tmax':2.5,'ncomp':8,'nbands':9,'model':'SVM','C':1e-4,'kf':'linear'}] 
            # if acc_train < 1:
            try:
                # print('N trials: ' + str(len(trials)))
                fmin_objective = partial(objective, subj=suj, classes=class_ids)
                best = fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=0, points_to_evaluate=init_vals)
                # pickle.dump(trials, open(path_to_trials, 'wb'))
            except:
                print('Exception raised')
                # pickle.dump(trials, open(path_to_trials, 'wb'))
                raise  

            ##%% ###########################################################################
            # trials = pickle.load(open(path_to_trials, 'rb'))
            acc_train = round( (-1) * trials.best_trial['result']['loss'] * 100, 2)
            best = trials.best_trial['misc']['vals']
                        
            fl = int(best['fl'][0])
            fh = int(best['fh'][0])                       
            ncsp = int(best['ncomp'][0])
            tmin = best['tmin'][0]
            tmax = best['tmax'][0]
            nbands = int(best['nbands'][0])
            
            while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
            if nbands > (fh-fl): nbands = (fh-fl)
            approach = {'option': 'sbcsp', 'nbands': nbands}
            
            if best['clf'][0] == 0: clf = {'model':'LDA'}
            elif best['clf'][0] == 1: 
                svm_kernel = 'linear' if best['kernel'][0]==0 else 'poly' if best['kernel'][0]==1 else 'sigmoid' if best['kernel'][0]==2  else 'rbf'
                clf = {'model':'SVM','kernel':{'kf':svm_kernel},'C':best['C'][0]}
            elif best['clf'][0] == 2: 
                knn_metric = 'euclidean' if best['metric'][0]==0 else 'manhattan' if best['metric'][0]==1 else 'minkowski' if best['metric'][0]==2 else 'chebyshev'
                clf = {'model':'KNN','metric':knn_metric,'neig':int(best['neig'][0]), }
            elif best['clf'][0] == 3: clf = {'model':'LR'}
            
            ##%% #########################################################################
            
            TOP = H[ H['acc'] == H['acc'].max() ].iloc[0]
            acc_max, y_, yp_, t_ = teste(TOP, suj, class_ids)
            acc_max = round(acc_max*100,2)
            
            # H = H.sort_values(by='acc', ascending=False)
            # H = H.iloc[:100]
            
            V, P = [], []
            for i in range(len(H)):
                acc_test, y_, yp_, t_ = teste(H.iloc[i], suj, class_ids)
                V.append(y_)
                P.append(yp_)
            
            ### Voting
            V = np.asarray(V).T
            ym = np.asarray([mode(V[i])[0][0] for i in range(len(V))], dtype=int) 
            acc_mode = round(np.mean(ym == t_)*100,2)
            
            ### Averaging
            PM = np.mean(np.transpose(P, (1,2,0)), axis=2)
            yp = np.asarray([ class_ids[0] if (PM[p][0]>=PM[p][1]) else class_ids[1] for p in range(len(PM))], dtype=int)
            acc_pmean = round(np.mean(yp == t_)*100,2)
            
            ### Stacking
            # # Pa = [ np.r_[np.asarray(H['prob'][i])[:72,0],np.asarray(H['prob'][i])[72:,1]] for i in range(len(H)) ]
            # # Pb = [ np.r_[np.asarray(H['prob'][i])[:72,1],np.asarray(H['prob'][i])[72:,0]] for i in range(len(H)) ]
            # Pa = np.asarray([ np.asarray(H['prob'][i])[:,0] for i in range(len(H)) ]).T
            # Pb = np.asarray([ np.asarray(H['prob'][i])[:,1] for i in range(len(H)) ]).T
            # # Pab = np.r_[Pa,Pb]
            # tp = np.ones(len(Pa))
            # Pa_te = (np.asarray(P)[:,:,0]).T # modelos x ep_treino_(prob_A) 
            # Pb_te = (np.asarray(P)[:,:,1]).T # modelos x ep_treino_(prob_B)
            # lr_model = LinearRegression()
            # lr_model.fit(Pa, tp)
            # y_pa = lr_model.predict(Pa_te)
            # lr_model.fit(Pb, tp)
            # y_pb = lr_model.predict(Pb_te)
            # print(y_pa[0],y_pb[0])
            # y_mlr = np.asarray([ class_ids[0] if (y_pa[p]>=y_pb[p]) else class_ids[1] for p in range(len(y_pa))], dtype=int)
            # acc_mlr = np.mean(y_mlr == t_)        
            YP = np.asarray([ np.asarray(H['y'][i]) for i in range(len(H)) ]).T
            tp = np.r_[class_ids[0]*np.ones(int(len(YP)/2)), class_ids[1]*np.ones(int(len(YP)/2))]
            lr_model = LinearRegression()
            lr_model.fit(YP, tp)
            y_pa = lr_model.predict(V)
            y_mlr = np.asarray([ round(a) for a in y_pa], dtype=int)
            acc_mlr = round(np.mean(y_mlr == t_)*100,2)
            
            # PH = np.transpose(np.asarray([ H['yp'][i] for i in range(len(H)) ]), (1,0,2))
            # PHA = PH[:,:,0]
            # PHB = PH[:,:,1]

            acc_best = max(acc_max, acc_mode, acc_pmean, acc_mlr)
            
            ##################
            # desvio = 4 # desvio em torno do ncsp ótimo (deve ser par)
            # min_ncsp = (ncsp - desvio) if (ncsp - desvio) > 2 else 2
            # max_ncsp = (ncsp + desvio) if (ncsp + desvio) < 22 else 22
            # space_tune = tuple([ hp.quniform('csp'+str(i), min_ncsp, max_ncsp, 2) for i in range(nbands) ])
            # bci2.data, bci2.events, bci2.class_ids, bci2.fs, bci2.overlap = data, events, class_ids, info['fs'], overlap
            # bci2.crossval, bci2.nfolds, bci2.test_perc, bci2.split = crossval, nfolds, test_perc, data_split
            # bci2.f_low, bci2.f_high, bci2.tmin, bci2.tmax, bci2.ncomp = fl, fh, tmin, tmax, ncsp
            # bci2.ap, bci2.filt_info, bci2.clf = approach, filtering, clf
            # trials_tune = base.Trials()
            # init_vals_tune = {'csp0':int(ncsp)}
            # for i in range(1, nbands-1): init_vals_tune.update({'csp'+str(i):int(ncsp)})
            # try: best_tune = fmin(objective_tune, space=space_tune, algo=tpe.suggest, max_evals=50, trials=trials_tune, points_to_evaluate=[init_vals_tune])
            # except: print('Exception raised'); raise   
            # acc_train_tune = (-1) * trials_tune.best_trial['result']['loss']
            # # print(f"AS(te_csp):{round(acc_train_tune*100,2)}")
            # csp_list = [ int(best_tune['csp'+str(i)]) for i in range(nbands) ]
            ##################
            
            icm = BCI()
            icm.data, icm.events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0'+str(suj)+'.npy', allow_pickle=True)
            icm.class_ids = class_ids
            icm.split = 'common'
            icm.fs=info['fs']; icm.overlap=True; crossval=False; nfolds=5; test_perc=0.5 
            
            icm.tmin=tmin; icm.tmax=tmax; icm.ncomp=ncsp; icm.f_low=fl; icm.f_high=fh
            icm.ap=approach; icm.filt_info={'design':'DFT'}; icm.clf=clf
            icm.evaluate()
            acc_max1 = icm.acc
            # icm.csp_list=csp_list; icm.evaluate(); acc_max1_tune = icm.acc
            
            ### Fixed SBCSP-DFT
            icm.csp_list = None
            icm.tmin=0.5; icm.tmax=2.5; icm.ncomp=8; icm.f_low=4; icm.f_high=40;  
            icm.ap={'option':'sbcsp','nbands':9}; icm.filt_info={'design':'DFT'}
            icm.clf={'model':'SVM','kernel':{'kf':'linear'},'C':-4}
            icm.evaluate(); sb_dft = round(icm.acc*100,2)
            
            ### Fixed SBCSP-IIR
            icm.filt_info={'design':'IIR','iir_order':5}
            icm.evaluate(); sb_iir = round(icm.acc*100,2)
            
            ### Fixed CSP-LDA-DFT
            icm.f_low=4; icm.f_high=40;
            icm.ap={'option':'classic'}; icm.filt_info={'design':'DFT'}; icm.clf={'model':'LDA'}
            icm.evaluate(); cla_dft = round(icm.acc*100,2)
            
            ### Fixed CSP-LDA-IIR
            icm.filt_info={'design':'IIR','iir_order':5}
            icm.evaluate(); cla_iir = round(icm.acc*100,2)
            
            # print(f"Best Setup:{fl}-{fh}Hz; {tmin}-{tmax}s; Ns={nbands}; R={ncsp}; CLF={clf}") # {csp_list}  
            # print(f"Train:{acc_train} | Max:{acc_max} | Moda:{acc_mode} | Media:{acc_pmean} | Regre:{acc_mlr}")
            # print(f"ASBEST:{acc_best} | SBDFT:{sb_dft}' | SBIIR:{sb_iir}' | CLDFT:{cla_dft}' | CLIIR:{cla_iir}\n")

            R.loc[len(R)] = [suj, class_ids[0], class_ids[1], tmin, tmax, fl, fh, ncsp, nbands, clf['model'], clf, acc_train, 
                             acc_max, acc_mode, acc_pmean, acc_mlr, acc_best, sb_dft, sb_iir, cla_dft, cla_iir]
    
    print(R.iloc[:,11:].mean())



    #%% PLOT GRAFIC #####################################################################
    acc_as = R['as_best']*100
    ref = ['cla_iir','sb_dft']
    plt.rcParams.update({'font.size':12})
    plt.figure(3, facecolor='mintcream')
    plt.subplots(figsize=(10, 12), facecolor='mintcream')
    for i in range(2):
        acc_ref = R[ref[i]]*100
        plt.subplot(2, 1, i+1)
        plt.scatter(np.asarray(acc_ref).reshape(-1,1), np.asarray(acc_as).reshape(-1,1), facecolors = 'c', marker = 'o', s=50, alpha=.9, edgecolors='firebrick', zorder=3)
        plt.scatter(round(acc_ref.mean(),2), round(acc_as.mean(),2), facecolors = 'dodgerblue', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label=r'Acurácia Média', zorder=5)
        plt.plot(np.linspace(40, 110, 1000), np.linspace(40, 110, 1000), color='dimgray', linewidth=1, linestyle='--', zorder=0) #linha pontilhada diagonal - limiar 
        plt.ylim((48, 102))
        plt.xlim((48, 102))
        plt.xticks(np.arange(50, 102, 5))
        plt.yticks(np.arange(50, 102, 5)) 
        plt.plot(np.ones(1000)*round(acc_ref.mean(),2), np.linspace(40, round(acc_as.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) # linha pontilhada verical - acc média auto setup
        plt.plot(np.linspace(40, round(acc_ref.mean(),2), 1000), np.ones(1000)*round(acc_as.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) # linha pontilhada horizontal - acc média ref
        plt.xlabel('Acurácia ' + ('CSP-LDA' if i==0 else 'SBCSP' ) + ' (configuração única) (%)', fontsize=12)
        plt.ylabel('Acurácia Auto Setup (%)', fontsize=12)
        plt.legend(loc='lower right', fontsize=12)
    # plt.savefig('/home/vboas/Desktop/scatter_y_'+datetime.now().strftime('%d-%m-%Y_%Hh%Mm')+'.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        
    ##%% SAVE RESULTS PICKLE FILE ########################################################
    # pd.to_pickle(R, '/home/vboas/Desktop/RESULTS_'+datetime.now().strftime('%d-%m-%Y_%Hh%Mm')+'.pkl')           
    
    # R = pd.read_pickle("/home/vboas/Desktop/RESULTS_01.pkl")