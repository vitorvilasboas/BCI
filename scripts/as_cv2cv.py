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
from bci_utils import nanCleaner, corrigeNaN, extractEpochs, Filter
from functools import partial

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

H = pd.DataFrame(columns=['fl','fh','tmin','tmax','ncsp','nbands','csp_reg','clf_model','csp','lda','p0','p1','clf_final','t','y','yp','acc'])

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
                 f_low=None, f_high=None, tmin=None, tmax=None, ncomp=None, nbands=None, ap=None, filt_info=None, clf=None, 
                 csp_reg=None, split='common', csp_list=None):
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
        self.csp_list = csp_list
        self.split = split
        self.csp_reg = csp_reg
        
        self.idx_search = None
        self.idx_testout = None
         
    def evaluate(self): 
        if self.clf['model'] == 'LDA': self.clf_final = LDA()
        elif self.clf['model'] == 'SVM': 
            self.clf_final = SVC(kernel=self.clf['kernel']['kf'], C=10**(self.clf['C']), gamma='scale', degree=3, probability=True)
        elif self.clf['model'] == 'KNN':   
            self.clf_final = KNeighborsClassifier(n_neighbors=int(self.clf['neig']), metric=self.clf['metric'], p=3) # p=self.clf['p']                                       
        elif self.clf['model'] == 'LR': self.clf_final = LogisticRegression()

        smin = math.floor(self.tmin * self.fs)
        smax = math.floor(self.tmax * self.fs)
        self.res_freq = self.fs/(smax-smin) # rf=Fs/Q
        self.dft_size = 2/self.res_freq # 2=sen/cos complexo fft
        
        # extração de épocas conforme classes e buffer
        self.epochs, self.labels = extractEpochs(self.data, self.events, smin, smax, self.class_ids)
        self.epochs = nanCleaner(self.epochs)
        
        # epocas e rótulos agrupadas por classes (primeira metade A, segunda metade B)
        self.epochs = [ self.epochs[np.where(self.labels == i)] for i in self.class_ids ]
        XA = self.epochs[0] # class A only
        XB = self.epochs[1] # class B only
        self.epochs = np.r_[XA,XB] 
        self.labels = np.r_[self.class_ids[0]*np.ones(len(XA)), self.class_ids[1]*np.ones(len(XB))].astype(int)
        
        # print([np.asarray(self.epochs).shape, np.asarray(self.labels).shape])
        # print(self.labels)

        # backup da estrutura com todas as epocas/rotulos agrupadas (primeira metade A, segunda metade B)
        self.epochs_bkp = self.epochs.copy() 
        self.labels_bkp = self.labels.copy() 
        
        # Somente se uma lista de indices foi passada no argumento (se None mantém todas as épocas do conjunto carregado)
        if not self.idx_search is None:
            # seleciona apenas as épocas/rótulos de interesse para o processo de auto-setup (treino e teste interno)
            self.epochs, self.labels = self.epochs[idx_search], self.labels[idx_search]
            
            # reagrupa as épocas/rótulos por classe (primeira metade A, segunda metade B)
            self.epochs = [ self.epochs[np.where(self.labels == i)] for i in self.class_ids ]
            XA = self.epochs[0] # class A only
            XB = self.epochs[1]
            self.epochs = np.r_[XA,XB] 
            self.labels = np.r_[self.class_ids[0]*np.ones(len(XA)), self.class_ids[1]*np.ones(len(XB))].astype(int)

        # print([np.asarray(self.epochs).shape, np.asarray(self.labels).shape])
        # print(self.labels)
        # self.epochs = nanCleaner(self.epochs)

        if self.crossval:
            self.cross_scores, self.cross_kappa, self.cross_clfs_params = [], [], []
            kf = StratifiedShuffleSplit(self.nfolds, test_size=self.test_perc, random_state=42)
            # kf = StratifiedKFold(self.nfolds, False)
            # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
            # self.cross_scores = cross_val_score(self.chain, XF, self.labels, cv=kf)
            for idx_treino, idx_valida in kf.split(self.epochs, self.labels):
                XT, XV, yT, yV = self.epochs[idx_treino], self.epochs[idx_valida], self.labels[idx_treino], self.labels[idx_valida]
                # print([np.asarray(XT).shape, np.asarray(XV).shape, yT.shape, yV.shape])
                # print([np.unique(yT, return_counts=True), np.unique(yV, return_counts=True)])
                acc_fold, kappa_fold, params_fold = self.classic_approach(XT, XV, yT, yV) if (self.ap['option'] == 'classic') else self.sbcsp_approach(XT, XV, yT, yV)     
                
                self.cross_scores.append(acc_fold) # self.cross_scores.append(self.chain.score(XV, yV))
                self.cross_kappa.append(kappa_fold)
                self.cross_clfs_params.append(params_fold) 
                
                
            idx_max = np.argmax(self.cross_scores)
            # print(max(self.cross_scores), np.mean(self.cross_scores))
            # self.acc, self.kappa, self.clf_params = max(self.cross_scores), max(self.cross_kappa), self.cross_clfs_params[idx_max]
            self.acc, self.kappa, self.clf_params = np.mean(self.cross_scores), np.mean(self.cross_kappa), self.cross_clfs_params[idx_max]
        else:
            test_size_A = int(len(XA) * self.test_perc)
            train_size_A = int(len(XA) - test_size_A)
            train_size_A = train_size_A if (train_size_A % 2 == 0) else train_size_A - 1 # garantir balanço entre as classes (amostragem estratificada)
            
            test_size_B = int(len(XB) * self.test_perc)
            train_size_B = int(len(XB) - test_size_B)
            train_size_B = train_size_B if (train_size_B % 2 == 0) else train_size_B - 1 # garantir balanço entre as classes (amostragem estratificada)
            
            # No idx_search and No idx_testout
            XT = np.r_[XA[:train_size_A], XB[:train_size_B]] # Train data classes A + B
            XV = np.r_[XA[train_size_A:], XB[train_size_B:]] # Test data classes A + B 
            
            if not self.idx_testout is None:
                self.epochs_bkp, self.labels_bkp = self.epochs_bkp[self.idx_testout], self.labels_bkp[self.idx_testout]
                self.epochs_bkp = [ self.epochs_bkp[np.where(self.labels_bkp == i)] for i in self.class_ids ]
                XV = np.r_[self.epochs_bkp[0],self.epochs_bkp[1]] # conjunto de test externo
                XT = np.r_[XA,XB] if self.split == 'full_train' else XT # XT é todo o conjunto de search ou é o mesmo sub-conjunto de search
                
            yT = np.r_[self.class_ids[0] * np.ones(int(len(XT)/2)), self.class_ids[1] * np.ones(int(len(XT)/2))].astype(int)
            yV = np.r_[self.class_ids[0] * np.ones(int(len(XV)/2)), self.class_ids[1] * np.ones(int(len(XV)/2))].astype(int)
            
            self.acc, self.kappa, self.clf_params = self.classic_approach(XT, XV, yT, yV) if (self.ap['option'] == 'classic') else self.sbcsp_approach(XT, XV, yT, yV)
            
        # print([np.asarray(XT).shape, yT.shape, np.asarray(XV).shape, yV.shape], '\n')
        
    
    def classic_approach(self, XT, XV, yT, yV):
        self.filt = Filter(self.f_low, self.f_high, self.fs, self.filt_info)
        XTF = self.filt.apply_filter(XT)
        XVF = self.filt.apply_filter(XV)
        if self.filt_info['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
            bmin = round(self.f_low * self.dft_size)
            bmax = round(self.f_high * self.dft_size)
            XTF = XTF[:, :, bmin:bmax]
            XVF = XVF[:, :, bmin:bmax]
        
        # self.csp = CSP(n_components=int(self.ncomp)) # CSP 1
        self.csp = mne.decoding.CSP(n_components=int(self.ncomp), reg=self.csp_reg) # CSP 2
        
        self.csp.fit(XTF, yT)
        # self.csp_filters = self.csp.filters_ # CSP 1
        self.csp_filters = self.csp.filters_[:int(self.ncomp)] # CSP 2
        XT_CSP = self.csp.transform(XTF)
        XV_CSP = self.csp.transform(XVF) 
        self.clf_final.fit(XT_CSP, yT)
        self.scores = self.clf_final.predict(XV_CSP)
        y_proba = self.clf_final.predict_proba(XV_CSP)

        classifier = {'csp_filt':self.csp_filters, 'lda':None, 'y':self.scores,
                      'yp':y_proba, 't':yV, 'p0':None, 'p1':None, 'clf_final':self.clf_final}
        
        acc = np.mean(self.scores == yV)     
        kappa = cohen_kappa_score(self.scores, yV)
        return acc, kappa, classifier
        
    
    def sbcsp_approach(self, XT, XV, yT, yV):
        nbands = int(self.ap['nbands'])
        n_bins = self.f_high - self.f_low
        if self.overlap: 
            step = n_bins/(nbands+1)
            size = step/0.5
        else:
            step = n_bins/nbands
            size = step
        sub_bands = []        
        for i in range(nbands):
            fl_sb = i * step + self.f_low
            fh_sb = i * step + size + self.f_low
            sub_bands.append([fl_sb, fh_sb])
        # nbands = len(sub_bands)
        
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
        elif self.filt_info['design'] in ['IIR' or 'FIR']:
            for i in range(nbands):
                filt = Filter(sub_bands[i][0], sub_bands[i][1], self.fs, self.filt_info)
                XTF.append(filt.apply_filter(XT))
                XVF.append(filt.apply_filter(XV))
        
        if self.csp_list is None:
            # self.chain = [ Pipeline([('CSP', CSP(n_components=int(self.ncomp))), ('LDA', LDA())]) for i in range(nbands) ] # CSP 1
            self.chain = [ Pipeline([('CSP', mne.decoding.CSP(n_components=int(self.ncomp), reg=self.csp_reg)), ('LDA', LDA())]) for i in range(nbands) ] # CSP 2
        else: self.chain = [ Pipeline([('CSP', CSP(n_components=self.csp_list[i])), ('LDA', LDA())]) for i in range(nbands) ] # uncomment to tuning ncsp
        
        for i in range(nbands): self.chain[i]['CSP'].fit(XTF[i], yT)  
        XT_CSP = [ self.chain[i]['CSP'].transform(XTF[i]) for i in range(nbands) ]
        XV_CSP = [ self.chain[i]['CSP'].transform(XVF[i]) for i in range(nbands) ]
        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        for i in range(nbands): 
            self.chain[i]['LDA'].fit(XT_CSP[i], yT)
            SCORE_T[:, i] = np.ravel(self.chain[i]['LDA'].transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(self.chain[i]['LDA'].transform(XV_CSP[i]))
        # csp_filters_sblist = [ self.chain[i]['CSP'].filters_ for i in range(nbands) ] # CSP 1
        # csp_filters_sblist = [ self.chain[i]['CSP'].filters_[:int(self.ncomp)] for i in range(nbands) ] # CSP 2
        if self.csp_list is None: csp_filters_sblist = [ self.chain[i]['CSP'].filters_[:int(self.ncomp)] for i in range(nbands) ] # CSP 2
        else:csp_filters_sblist = [ self.chain[i]['CSP'].filters_[:self.csp_list[i]] for i in range(nbands) ]
        lda_sblist = [ self.chain[i]['LDA'] for i in range(nbands) ] 

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
                      'yp':y_proba, 't':yV, 'p0':self.p0, 'p1':self.p1, 'clf_final':self.clf_final}
        
        acc = np.mean(self.scores == yV)
        kappa = cohen_kappa_score(self.scores, yV)
        return acc, kappa, classifier



def csp_tunning(args, subj, classes, idx_search, setup):
    path = '/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + '.npy'
    bci = BCI()
    bci.data, bci.events, info = np.load(path, allow_pickle=True)  
    bci.class_ids, bci.fs = class_ids, info['fs'] 
    bci.overlap = True
    bci.crossval = False 
    bci.nfolds = 5; 
    bci.test_perc = 0.2 if bci.crossval else 0.5
    bci.split = 'common' # common, as_train, as_test
    bci.filt_info = {'design':'DFT'}
    bci.idx_search = idx_search
    bci.f_low = setup['fl'] 
    bci.f_high = setup['fh']
    bci.tmin = setup['tmin']
    bci.tmax = setup['tmax']
    bci.ncomp = setup['ncomp']
    # bci.csp_reg = setup['reg']
    bci.ap = {'option': 'sbcsp', 'nbands': setup['nbands']}
    bci.clf = setup['clf']
    bci.csp_list = list(map(lambda x: int(x), args))
    bci.evaluate()
    return bci.acc * (-1)


def objective(args, subj, classes, idx_search):
    path = '/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + '.npy'
    bci = BCI()
    bci.data, bci.events, info = np.load(path, allow_pickle=True)
    bci.class_ids = class_ids  
    bci.fs = info['fs'] 
    bci.overlap = True
    bci.crossval = True                    ### False to MLR
    bci.nfolds = 5 
    bci.test_perc = 0.205 if bci.crossval else 0.5
    bci.split = 'common' # common, as_train, as_test
    bci.filt_info = {'design':'DFT'}
    bci.idx_search = idx_search
        
    f_low, f_high, tmin, tmax, ncomp, nbands, clf = args # reg
    f_low, f_high = f_low['fl'],f_high['fh']
    # f_low, f_high, tmin, tmax, ncomp, nbands, reg = 0, 40, 0.5, 2.5, 8, 9, 1e-1
    # clf = {'model':'SVM','kernel':{'kf':'linear'},'C':-4}
        
    if nbands > (f_high - f_low): nbands = (f_high - f_low)
    while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
    bci.tmin, bci.tmax = tmin, tmax
    bci.f_low, bci.f_high, bci.ncomp = int(f_low), int(f_high), int(ncomp)
    # bci.csp_reg = 10**(int(reg))
    bci.ap = {'option': 'sbcsp', 'nbands': nbands}
    bci.clf = clf
    bci.evaluate()
    # print((f_low, f_high, tmin, tmax, ncomp, nbands, clf))
    
    H.loc[len(H)] = [bci.f_low, bci.f_high, bci.tmin, bci.tmax, bci.ncomp, nbands, bci.csp_reg, bci.clf['model'], 
                     bci.clf_params['csp_filt'], bci.clf_params['lda'], bci.clf_params['p0'], bci.clf_params['p1'], 
                     bci.clf_params['clf_final'], bci.clf_params['t'], bci.clf_params['y'],bci.clf_params['yp'], bci.acc]
    
    return bci.acc * (-1)


def teste(h, suj, class_ids, idx_test):
    data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + '.npy', allow_pickle=True) 
    f_low, f_high, tmin, tmax, nbands = h['fl'], h['fh'], h['tmin'], h['tmax'], h['nbands']
    p0 = h['p0']; p1 = h['p1']; csp_filters = h['csp']; lda = h['lda']; clf_final = h['clf_final']; 
    if nbands > (f_high - f_low): nbands = (f_high - f_low)
    smin, smax = math.floor(tmin * info['fs']), math.floor(tmax * info['fs'])
    
    epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    epochs = [ epochs[np.where(labels==i)] for i in class_ids ]
    ZA = epochs[0] # class A only
    ZB = epochs[1] # class B only
    Z = np.r_[ZA,ZB]
    t = np.r_[class_ids[0]*np.ones(len(ZA)), class_ids[1]*np.ones(len(ZB))].astype(int)
    
    if not idx_test is None:
        Z, t = Z[idx_test], t[idx_test]
        Z = [ Z[np.where(t == i)] for i in class_ids ]
        ZA = Z[0]
        ZB = Z[1]
        Z = np.r_[ZA,ZB] 
        t = np.r_[class_ids[0]*np.ones(len(ZA)), class_ids[1]*np.ones(len(ZB))].astype(int)
    
    n_bins = f_high - f_low
    overlap = 0.5 
    step = n_bins / (nbands+1)
    size = step / overlap
    n_samples = smax - smin
    dft_resf = info['fs']/n_samples 
    bin_size = 2/dft_resf # 2 representa sen e cos
    sb_dft = []
    for i in range(nbands):
        bmin = round((i * step + f_low) * bin_size)
        bmax = round((i * step + size + f_low) * bin_size)
        sb_dft.append([bmin,bmax]) 
    # nbands = len(sb_dft)
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
    SCORE = np.asarray([ np.ravel(lda[i].transform(XCSP[i])) for i in range(nbands) ]).T # Score LDA  
    META_SCORE = np.log(p0.pdf(SCORE) / p1.pdf(SCORE)) # Bayesian Meta-score
    y_labels = clf_final.predict(META_SCORE) # Final classification
    y_proba = clf_final.predict_proba(META_SCORE)
    acc = clf_final.score(META_SCORE, t)
    return acc, y_labels, y_proba, t


if __name__ == "__main__":
    n_iter = 100
    n_iter_tune = 20
    subjects = range(1,10) 
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] # 
    R = pd.DataFrame(columns=['subj','A','B','tmin','tmax','fl','fh','ncsp','nbands','clf','clf_details','csp_list','as_train','as_train_tune',
                              'as_max','as_tune','as_mode','as_pmean','acc_mlr','as_best','sb_dft','sb_iir','cla_dft','cla_iir'])
       
    kf = StratifiedShuffleSplit(10, test_size=0.5, random_state=42)
    ep_fake = np.empty((288,22,500))
    lb_fake = np.r_[1*np.ones(int(len(ep_fake)/2)), 2*np.ones(int(len(ep_fake)/2))]
    for idx_search, idx_test in kf.split(ep_fake, lb_fake):
        
        # print('Search:', idx_search, '\n\n')
        # print('Test:', idx_test, '\n\n')
        
        ACC = []
        for suj in subjects:
            for class_ids in classes:    
                H = H.iloc[0:0] # cleaning df
                print(f'###### {suj} {class_ids} ######')  
                fl_ = hp.uniformint("fl", 0, 15)
                fh_ = hp.uniformint("fh", 10, 25)
                space = (
                    {"fl": fl_},
                    {"fh": (fh_ + fl_)}, # fl_ + 20, # hp.uniform("fh", (10 + fl_), 40),
                    hp.quniform('tmin', 0, 2, 0.5),
                    hp.quniform('tmax', 2, 4, 0.5),
                    # hp.quniform('ncomp', 2, 10, 2), 
                    hp.choice('ncomp', [2,4,6,8,22]),
                    hp.uniformint('nbands', 1, 25),
                    # hp.quniform('reg', -5, 0, 1),
                    hp.pchoice('clf', [
                        (0.2, {'model':'LDA'}),
                        (0.4, {'model':'SVM','C': hp.quniform('C', -6, 0, 1), # hp.loguniform('C', -6, 0), # 
                          'kernel':hp.choice('kernel',[{'kf':'linear'},{'kf':'poly'},{'kf':'sigmoid'},{'kf':'rbf'}])}),
                        (0.2, {'model':'KNN','neig':hp.uniformint('neig', 5, 20),
                          'metric':hp.choice('metric',['euclidean','manhattan','minkowski','chebyshev'])}),
                        (0.2, {'model':'LR'})
                        ])
                    )
    
                path = '../as_results/sbrt20/IV2a/'
                path_to_trials = path + 'A0' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
                if not os.path.isdir(path): os.makedirs(path)
                
                trials = base.Trials() 
                # try: trials = pickle.load(open(path_to_trials, 'rb'))
                # except: trials = base.Trials()
                # # trials = generate_trials_to_calculate(init_vals)
                init_vals = [{'fl':4,'fh':40,'tmin':0.5,'tmax':2.5,'ncomp':8,'nbands':9,'model':'SVM','C':1e-4,'kf':'linear'}] 
                try:
                    # print('N trials: ' + str(len(trials)))
                    fmin_objective = partial(objective, subj=suj, classes=class_ids, idx_search=idx_search)
                    best = fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=0, points_to_evaluate=init_vals)
                    # pickle.dump(trials, open(path_to_trials, 'wb'))
                except:
                    print('Exception raised')
                    # pickle.dump(trials, open(path_to_trials, 'wb'))
                    raise  
    
                ##%% ###########################################################################
                acc_train = round( (-1) * trials.best_trial['result']['loss'] * 100, 2)
                best = trials.best_trial['misc']['vals']
                
                fl = int(best['fl'][0])
                fh = int(best['fl'][0]) + int(best['fh'][0])                     
                ncsp = [2,4,6,8,22][best['ncomp'][0]] # int(best['ncomp'][0])
                tmin = best['tmin'][0]
                tmax = best['tmax'][0]
                nbands = int(best['nbands'][0])
                reg = None # 10**int(best['reg'][0])
                
                while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
                if nbands > (fh-fl): nbands = (fh-fl)
                
                if best['clf'][0] == 0: clf = {'model':'LDA'}
                elif best['clf'][0] == 1: 
                    svm_kernel = 'linear' if best['kernel'][0]==0 else 'poly' if best['kernel'][0]==1 else 'sigmoid' if best['kernel'][0]==2  else 'rbf'
                    clf = {'model':'SVM','kernel':{'kf':svm_kernel},'C':best['C'][0]}
                elif best['clf'][0] == 2: 
                    knn_metric = 'euclidean' if best['metric'][0]==0 else 'manhattan' if best['metric'][0]==1 else 'minkowski' if best['metric'][0]==2 else 'chebyshev'
                    clf = {'model':'KNN','metric':knn_metric,'neig':int(best['neig'][0]), }
                elif best['clf'][0] == 3: clf = {'model':'LR'}
                # print((fl, fh, tmin, tmax, ncsp, nbands, reg, clf))
                
                ##%% #########################################################################
                desvio = 4 # desvio em torno do ncsp ótimo (deve ser par)
                min_ncsp = (ncsp - desvio) if (ncsp - desvio) > 2 else 2
                max_ncsp = (ncsp + desvio) if (ncsp + desvio) < 22 else 22
                space_tune = tuple([ hp.quniform('csp'+str(i), min_ncsp, max_ncsp, 2) for i in range(nbands) ])
                fmin_tunning = partial(csp_tunning, subj=suj, classes=class_ids, idx_search=idx_search, setup={'fl':fl, 'fh':fh, 'tmin':tmin, 'tmax':tmax, 'ncomp':ncsp, 'nbands':nbands, 'reg':reg, 'clf':clf})
                trials_tune = base.Trials()
                init_vals_tune = {'csp0':int(ncsp)}
                for i in range(1, nbands-1): init_vals_tune.update({'csp'+str(i):int(ncsp)})
                try: best_tune = fmin(fmin_tunning, space=space_tune, algo=tpe.suggest, max_evals=n_iter_tune, trials=trials_tune, points_to_evaluate=[init_vals_tune])
                except: print('Exception raised'); raise   
                acc_train_tune = round((-1) * trials_tune.best_trial['result']['loss'] * 100, 2)
                csp_list = None
                csp_list = [ int(best_tune['csp'+str(i)]) for i in range(nbands) ]
                ##%% #########################################################################
                
                TOP = H[ H['acc'] == H['acc'].max() ].iloc[0]
                acc_max, y_, yp_, t_ = teste(TOP, suj, class_ids, idx_test)
                acc_max = round(acc_max*100,2)
                
                # H = H.sort_values(by='acc', ascending=False)
                # H = H.iloc[0:50]
                # H = H.reset_index(drop=True)
                
                V, P = [], []
                for i in range(len(H)):
                    acc_test, y_, yp_, t_ = teste(H.iloc[i], suj, class_ids, idx_test)
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
                
                acc_mlr = 0
                ## Stacking      
                # PH = np.asarray([ np.asarray(H['yp'][i]) for i in range(len(H)) ]).T
                # PHA = PH[0]
                # PHB = PH[1]
                # th = [ np.asarray(H['t'][i]) for i in range(len(H)) ][0]
                
                # # Sem cross-validation:
                # # problema ao usar CV.. o th deve ser o mesmo em todas as iterações, logo os folds devem ser iguais
                # # nas iterações com indices de epocas e labels permanecendo fixos
    
                # LR_A = LinearRegression()
                # LR_A.fit(PHA, th)
                
                # LR_B = LinearRegression()
                # LR_B.fit(PHB, th)
                
                # PA = np.transpose(P, (2,1,0))[0]
                # PB = np.transpose(P, (2,1,0))[1]
                
                # r2_a = LR_A.score(PA, t_)*(-1)
                # r2_b = LR_B.score(PB, t_)*(-1)
                # # print(r2_a, r2_b)
                
                # scores_a = LR_A.predict(PA)
                # scores_b = LR_B.predict(PB)
                # ya = np.asarray([ round(scores_a[i]) for i in range(len(scores_a)) ])
                # yb = np.asarray([ round(scores_b[i]) for i in range(len(scores_b)) ])
                # ya = np.where(ya > class_ids[0], class_ids[1], class_ids[0])
                # yb = np.where(yb > class_ids[0], class_ids[1], class_ids[0])
                # # erro_a = np.square(t_ - scores_a)
                # # erro_b = np.square(t_ - scores_b)
                
                # acc_mlr = round(max( np.mean(ya == t_), np.mean(yb == t_) ) *100, 2)
                # print(acc_mlr)
                
                dt, ev, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + '.npy', allow_pickle=True)
                
                ### Max h :: Re-train (more epochs) and test
                icm = BCI(data=dt, events=ev, class_ids=class_ids, fs=info['fs'], overlap=True, split='full_train', csp_list=None,
                          csp_reg=None, crossval=False, nfolds=5, test_perc=0.5, tmin=tmin, tmax=tmax, ncomp=ncsp, 
                          f_low=fl, f_high=fh, ap={'option':'sbcsp', 'nbands':nbands}, filt_info={'design':'DFT'}, clf=clf)
                icm.idx_search = idx_search; icm.idx_testout = idx_test
                icm.evaluate(); acc_max_retrain = round(icm.acc*100,2)
                # print(f'Confirm: {acc_max_retrain}')
                
                ### Tunning CSP
                icm = BCI(data=dt, events=ev, class_ids=class_ids, fs=info['fs'], overlap=True, csp_list=csp_list,
                          csp_reg=None, crossval=False, nfolds=5, test_perc=0.5, tmin=tmin, tmax=tmax, ncomp=ncsp, 
                          f_low=fl, f_high=fh, ap={'option':'sbcsp', 'nbands':nbands}, filt_info={'design':'DFT'}, clf=clf)
                icm.idx_search = idx_search; icm.idx_testout = idx_test
                icm.evaluate(); acc_tune = round(icm.acc*100,2)
                
                ### Fixed SBCSP-DFT
                icm = BCI(data=dt, events=ev, class_ids=class_ids, fs=info['fs'], overlap=True, split='full_train', csp_list=None,
                          csp_reg=None, crossval=False, nfolds=5, test_perc=0.5, tmin=0.5, tmax=2.5, ncomp=8, 
                          f_low=4, f_high=40, ap={'option':'sbcsp','nbands':9}, filt_info={'design':'DFT'}, clf={'model':'SVM','kernel':{'kf':'linear'},'C':-4})
                icm.idx_search = idx_search; icm.idx_testout = idx_test
                icm.evaluate(); sb_dft = round(icm.acc*100,2)
                
                ### Fixed SBCSP-IIR
                icm = BCI(data=dt, events=ev, class_ids=class_ids, fs=info['fs'], overlap=True, split='full_train', csp_list=None,
                          csp_reg=None, crossval=False, nfolds=5, test_perc=0.5, tmin=0.5, tmax=2.5, ncomp=8, 
                          f_low=4, f_high=40, ap={'option':'sbcsp','nbands':9}, filt_info={'design':'IIR','iir_order':5}, clf={'model':'SVM','kernel':{'kf':'linear'},'C':-4})
                icm.idx_search = idx_search; icm.idx_testout = idx_test
                icm.evaluate(); sb_iir = round(icm.acc*100,2)
                
                ### Fixed CSP-LDA-DFT
                icm = BCI(data=dt, events=ev, class_ids=class_ids, fs=info['fs'], overlap=True, split='full_train', csp_list=None,
                          csp_reg=None, crossval=False, nfolds=5, test_perc=0.5, tmin=0.5, tmax=2.5, ncomp=8, 
                          f_low=8, f_high=30, ap={'option':'classic'}, filt_info={'design':'DFT'}, clf={'model':'LDA'})
                icm.idx_search = idx_search; icm.idx_testout = idx_test
                icm.evaluate(); cla_dft = round(icm.acc*100,2)
                
                ### Fixed CSP-LDA-IIR
                icm = BCI(data=dt, events=ev, class_ids=class_ids, fs=info['fs'], overlap=True, split='full_train', csp_list=None,
                          csp_reg=None, crossval=False, nfolds=5, test_perc=0.5, tmin=0.5, tmax=2.5, ncomp=8, 
                          f_low=8, f_high=30, ap={'option':'classic'}, filt_info={'design':'IIR','iir_order':5}, clf={'model':'LDA'})
                icm.idx_search = idx_search; icm.idx_testout = idx_test
                icm.evaluate(); cla_iir = round(icm.acc*100,2)
                
                acc_best = max(acc_max, acc_mode, acc_pmean, acc_tune, acc_mlr)
                # print(f"Best Setup: {fl}-{fh}Hz; {tmin}-{tmax}s; Ns={nbands}; R={ncsp} {csp_list}; Reg={reg}; CLF={clf}")
                # print(f"Train :{acc_train} | Train Tune:{acc_train_tune} | Max:{acc_max} | Tune:{acc_tune} | Moda:{acc_mode} | Media:{acc_pmean} | MLR:{acc_mlr}")
                # print(f"ASBEST:{acc_best} | SBDFT:{sb_dft} | SBIIR:{sb_iir} | CLDFT:{cla_dft} | CLIIR:{cla_iir}\n")
    
                R.loc[len(R)] = [suj, class_ids[0], class_ids[1], tmin, tmax, fl, fh, ncsp, nbands, clf['model'], clf, csp_list, acc_train, acc_train_tune, 
                                  acc_max, acc_tune, acc_mode, acc_pmean, acc_mlr, acc_best, sb_dft, sb_iir, cla_dft, cla_iir]
        
        print(R.iloc[:,12:].mean())
               
        ##%% PLOT GRAFIC #####################################################################
        acc_as = R['as_best']
        ref = ['cla_iir','sb_dft']
        plt.rcParams.update({'font.size':12})
        plt.figure(3, facecolor='mintcream')
        plt.subplots(figsize=(10, 12), facecolor='mintcream')
        for i in range(2):
            acc_ref = R[ref[i]]
            plt.subplot(2, 1, i+1)
            plt.scatter(np.asarray(acc_ref).reshape(-1,1), np.asarray(acc_as).reshape(-1,1), facecolors = 'c', marker = 'o', s=50, alpha=.9, edgecolors='firebrick', zorder=3)
            plt.scatter(acc_ref.mean(), acc_as.mean(), facecolors = 'dodgerblue', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label=r'Acurácia Média', zorder=5)
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
        
        ##%% SAVE RESULTS
        plt.savefig('/home/vboas/cloud/devto/BCI/as_results/sbrt20/scatter_y_'+datetime.now().strftime('%d-%m-%Y_%Hh%Mm')+'.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        pd.to_pickle(R, '/home/vboas/cloud/devto/BCI/as_results/sbrt20/RESULTS_'+datetime.now().strftime('%d-%m-%Y_%Hh%Mm')+'.pkl')           
        # R = pd.read_pickle("/home/vboas/Desktop/as/RESULTS_01.pkl")