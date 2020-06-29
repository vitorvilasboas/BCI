# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import re
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
    
    csp = CSP(n_components=int(ncsp)) # CSP 1
    # csp = mne.decoding.CSP(n_components=int(ncsp), reg=csp_reg) # CSP 2
    
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
        chain = [ Pipeline([('CSP', CSP(n_components=int(ncsp))), ('LDA', LDA())]) for i in range(nbands) ] # CSP 1
        # chain = [ Pipeline([('CSP', mne.decoding.CSP(n_components=int(ncsp), reg=csp_reg)), ('LDA', LDA())]) for i in range(nbands) ] # CSP 2
    else: 
        chain = [ Pipeline([('CSP', CSP(n_components=csp_list[i])), ('LDA', LDA())]) for i in range(nbands) ] # uncomment to tuning ncsp
        # chain = [ Pipeline([('CSP', mne.decoding.CSP(n_components=csp_list[i], reg=csp_reg)), ('LDA', LDA())]) for i in range(nbands) ] # CSP 2
    
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
    ldas = [ chain[i]['LDA'] for i in range(nbands) ]
       
    XV_CSP = [ chain[i]['CSP'].transform(XV[i]) for i in range(nbands) ]
    SCORE_V = np.zeros((len(ZV), nbands))
    for i in range(nbands): 
        SCORE_V[:, i] = np.ravel(chain[i]['LDA'].transform(XV_CSP[i]))
    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))
    
    y = clf_final.predict(META_SCORE_V)
    yp = clf_final.predict_proba(META_SCORE_V)
    
    learner = {'csp_filt':csp_filters,'lda':ldas, 'p0':p0, 'p1':p1, 'clf_final':clf_final, 'y':y, 'yp':yp, 't':tv}
    
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
    if nbands > (f_high - f_low): nbands = (f_high - f_low)
    while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
    
    smin = math.floor(tmin * info['fs'])
    smax = math.floor(tmax * info['fs'])
    epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    epochs = nanCleaner(epochs)
    
    cross_acc, cross_learner = [], []
    kf = StratifiedShuffleSplit(5, test_size=0.2, random_state=42)
    for _train, _test in kf.split(epochs, labels):
        ZT, ZV, tt, tv = epochs[_train], epochs[_test], labels[_train], labels[_test]
        a, l = sbcsp_approach(ZT, ZV, tt, tv, nbands, f_low, f_high, ncsp, clf, Fs=info['fs'], filt='DFT')     
        cross_acc.append(a); cross_learner.append(l)     
    acc, _ = np.mean(cross_acc), cross_learner[np.argmax(cross_acc)]
    
    H.loc[len(H)] = [f_low, f_high, tmin, tmax, ncsp, nbands, clf, acc]
    
    return acc * (-1)

def tester(ZV, tv, Fs, class_ids, setup):
    fl, fh, nbands = setup['fl'], setup['fh'], setup['nbands']
    p0, p1, filters_, lda, clf_final = setup['p0'], setup['p1'], setup['csp'], setup['lda'], setup['clf_final'] 
    if nbands > (fh-fl): nbands = (fh-fl)
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
        
    YV = [ np.asarray([ np.dot(filters_[i], ep) for ep in XV[i] ]) for i in range(nbands) ] # Spatial Filtering
    XV_CSP = [ np.log(np.mean(YV[i]**2, axis=2)) for i in range(nbands) ] # Feature extraction
    # XV_CSP = [ np.log(np.var(YV[i], axis=2)) for i in range(nbands) ] # Feature extraction
    SCORE_V = np.asarray([ np.ravel(lda[i].transform(XV_CSP[i])) for i in range(nbands) ]).T # Score LDA  
    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V)) # Bayesian Meta-score
    y = clf_final.predict(META_SCORE_V) # Final classification
    yp = clf_final.predict_proba(META_SCORE_V)
    acc = clf_final.score(META_SCORE_V, tv)
    return acc, y, yp

def half_split_data(data, events, smin, smax, class_ids):
    ep, lb = extractEpochs(data, events, smin, smax, class_ids)
    et, lt = ep[:(len(ep)//2)], lb[:(len(ep)//2)] 
    ev, lv = ep[(len(ep)//2):], lb[(len(ep)//2):]
    ZT = np.r_[ et[np.where(lt == class_ids[0])], et[np.where(lt == class_ids[1])] ]
    ZV = np.r_[ ev[np.where(lv == class_ids[0])], ev[np.where(lv == class_ids[1])] ]
    tt = np.r_[ class_ids[0]*np.ones(len(ZT)//2), class_ids[1]*np.ones(len(ZT)//2)].astype(int)
    tv = np.r_[ class_ids[0]*np.ones(len(ZV)//2), class_ids[1]*np.ones(len(ZV)//2)].astype(int)
    return ZT, ZV, tt, tv

H = pd.DataFrame(columns=['fl','fh','tmin','tmax','ncsp','nbands','clf','acc'])

if __name__ == "__main__":
    n_iter = 100
    n_iter_tune = n_iter//10
    subjects = range(1,55) 
    classes = [[1, 2]]

    # R = pd.DataFrame(columns=['subj','A','B','as_best','sb_iir','cla_iir'])
    
    R = pd.DataFrame(columns=['subj', 'A', 'B', 'fl', 'fh', 'tmin', 'tmax', 'nbands', 'ncsp', 'csp_list', 'clf', 'clf_details',
                              'as_max', 'as_tune', 'as_mode', 'as_pmean', 'as_best', 'sb_iir', 'cla_iir', 
                              'as_on', 'sb_on', 'cla_on', 'as_command', 'sb_command', 'cla_command',
                              'as_comlist', 'sb_comlist', 'cla_comlist'])
    
    ds = 'Lee19'
    for suj in subjects:
        for class_ids in classes:  
            H = H.iloc[0:0] # cleaning df
            
            data, events, info = np.load('/mnt/dados/eeg_data/Lee19/npy1/S' + str(suj) + '.npy', allow_pickle=True)
            data_cal, events_cal, _ = np.load('/mnt/dados/eeg_data/Lee19/npy1/S' + str(suj) + 'T.npy', allow_pickle=True) # sess1
            data_val, events_val, _ = np.load('/mnt/dados/eeg_data/Lee19/npy1/S' + str(suj) + 'E.npy', allow_pickle=True) # sess2
            Fs = info['fs']
            filtering = 'DFT'
            
            # cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
            # data, data_cal, data_val = data[cortex], data_cal[cortex], data_val[cortex] 
            
            print(f'###### {suj} {class_ids} ######')
            
            ##%% ###########################################################################
            fl_ = hp.uniformint("fl", 0, 15)
            fh_ = hp.uniformint("fh", 10, 25)
            space = (
                {"fl": fl_},
                {"fh": (fh_ + fl_)}, # fl_ + 20, # hp.uniform("fh", (10 + fl_), 40),
                hp.quniform('tmin', 0, 2, 0.5),
                hp.quniform('tmax', 2, 4, 0.5),
                # hp.quniform('ncomp', 2, 10, 2), 
                hp.choice('ncomp', [2,4,6,8,20]),
                hp.uniformint('nbands', 1, 25),
                # hp.quniform('reg', -5, 0, 1),
                hp.pchoice('clf', [
                    (0.25, {'model':'LDA'}),
                    (0.25, {'model':'SVM','C': hp.quniform('C', -6, 0, 1), # hp.loguniform('C', -6, 0), # 
                      'kernel':hp.choice('kernel',[{'kf':'linear'},{'kf':'poly'},{'kf':'sigmoid'},{'kf':'rbf'}])}),
                    (0.25, {'model':'KNN','neig':hp.uniformint('neig', 5, 20),
                      'metric':hp.choice('metric',['euclidean','manhattan','minkowski','chebyshev'])}),
                    (0.25, {'model':'LR'})
                    ])
                )
            path = '/home/vboas/cloud/devto/overmind/tests/'
            path_to_trials = path + 'S' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
            if not os.path.isdir(path): os.makedirs(path)
            trials = base.Trials() 
            # try: trials = pickle.load(open(path_to_trials, 'rb'))
            # except: trials = base.Trials()
            # trials = generate_trials_to_calculate(init_vals)
            init_vals = [{'fl':4,'fh':40,'tmin':0.5,'tmax':2.5,'ncomp':8,'nbands':9,'model':'SVM','C':1e-4,'kf':'linear'}] 
            try:
                # print('N trials: ' + str(len(trials)))
                fmin_objective = partial(objective, data=data_cal, events=events_cal, info=info, class_ids=class_ids)
                best = fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=0, points_to_evaluate=init_vals)
                # pickle.dump(trials, open(path_to_trials, 'wb'))
            except:
                print('Exception raised')
                # pickle.dump(trials, open(path_to_trials, 'wb'))
                raise  
        
            ##%% ###########################################################################
            as_max_cal = round( (-1) * trials.best_trial['result']['loss'] * 100, 2)
            best = trials.best_trial['misc']['vals']
            
            fl = int(best['fl'][0])
            fh = int(best['fl'][0]) + int(best['fh'][0])                     
            ncsp = [2,4,6,8,20][best['ncomp'][0]] # int(best['ncomp'][0])
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
            
            ##%% #########################################################################
            desvio = 4 # desvio em torno do ncsp ótimo (deve ser par)
            min_ncsp = (ncsp - desvio) if (ncsp - desvio) > 2 else 2
            max_ncsp = (ncsp + desvio) if (ncsp + desvio) < 22 else 22
            space = tuple([ hp.quniform('csp'+str(i), min_ncsp, max_ncsp, 2) for i in range(nbands) ])
            fmin_tunning = partial(csp_tunning, data=data_cal, events=events_cal, info=info, class_ids=class_ids, 
                                    setup={'fl':fl, 'fh':fh, 'tmin':tmin, 'tmax':tmax, 'ncomp':ncsp, 'nbands':nbands, 'reg':reg, 'clf':clf})
            trials = base.Trials()
            init_vals_tune = {'csp0':int(ncsp)}
            for i in range(1, nbands-1): init_vals_tune.update({'csp'+str(i):int(ncsp)})
            try: best = fmin(fmin_tunning, space=space, algo=tpe.suggest, max_evals=n_iter_tune, trials=trials, points_to_evaluate=[init_vals_tune])
            except: print('Exception raised'); raise   
            as_tune_cal = round((-1) * trials.best_trial['result']['loss'] * 100, 2)
            csp_list = [ int(best['csp'+str(i)]) for i in range(nbands) ]
            # print(f'csp_list: {csp_list}')
            
            ##%% #########################################################################
            hmax = H[ H['acc'] == H['acc'].max()].iloc[0]
            ZT, ZV, tt, tv = half_split_data(data, events, int(tmin*Fs), int(tmax*Fs), class_ids)
            as_max, _ = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=Fs, filt=filtering) 
            as_max = round(as_max*100,2)
            
            ##%% #########################################################################
            ZT, ZV, tt, tv = half_split_data(data, events, int(tmin*Fs), int(tmax*Fs), class_ids)
            as_tune, _ = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=Fs, filt=filtering, csp_list=csp_list)     
            as_tune = round(as_tune*100,2)
            
            ##%% #########################################################################
            V, P = [], []
            for i in range(len(H)):
                setup = H.iloc[i]
                ZT, ZV, tt, tv = half_split_data(data, events, int(setup['tmin']*Fs), int(setup['tmax']*Fs), class_ids)
                acc, learner = sbcsp_approach(ZT, ZV, tt, tv, setup['nbands'], setup['fl'], setup['fh'], setup['ncsp'], setup['clf'], Fs=Fs, filt=filtering)
                tv = learner['t']
                V.append(learner['y'])
                P.append(learner['yp'])
            
            ### Voting
            V = np.asarray(V).T
            yv = np.asarray([mode(V[i])[0][0] for i in range(len(V))], dtype=int)
            as_mode = round(np.mean(yv == tv)*100,2)
            
            ### Averaging
            PM = np.mean(np.transpose(P, (1,2,0)), axis=2)
            pv = np.asarray([ class_ids[0] if (PM[p][0]>=PM[p][1]) else class_ids[1] for p in range(len(PM))], dtype=int)
            as_pmean = round(np.mean(pv == tv)*100,2)
            
            as_best = max(as_tune, as_max, as_mode, as_pmean) 
            
            ##%% ###########################################################################
            ZT, ZV, tt, tv = half_split_data(data, events, int(0.5*Fs), int(2.5*Fs), class_ids)
            sb_iir, _ = sbcsp_approach(ZT, ZV, tt, tv, 9, 4, 40, 8, {'model':'SVM','kernel':{'kf':'linear'},'C':-4}, Fs=Fs, filt='IIR')
            cla_iir, _ = classic_approach(ZT, ZV, tt, tv, 8, 30, 8, {'model':'LDA'}, Fs=Fs, filt='IIR')
            sb_iir, cla_iir = round(sb_iir*100,2), round(cla_iir*100,2)
            
            ##%% ###########################################################################
            # print(f"Max: {as_max} | {as_max_cal} >> {fl}-{fh}Hz; {tmin}-{tmax}s; Ns={nbands}; R={ncsp}; {clf}")
            # print(f"Tune: {as_tune} | {as_tune_cal} >> {csp_list}")
            # print(f"Moda: {as_mode} | Media: {as_pmean}")
            # print(f"AS: {as_best} | SB: {sb_iir} | CLA: {cla_iir}\n")
            
            # R.loc[len(R)] = [suj, class_ids[0], class_ids[1], as_best, sb_iir, cla_iir]

            data_cal, events_cal = None, None
            ##%% ############################################################################
            # ============================================================================= #   
            # ON-LINE SIMULATION - BUFFER                                                   #
            # ============================================================================= #
            ##%% ############################################################################                                                         
            
            # raw = mne.io.read_raw_gdf('/mnt/dados/eeg_data/IV2a/A0' + str(suj) + 'E.gdf').load_data()
            # data_val = raw.get_data()[:22] # [channels x samples]
            # events_val = mne.events_from_annotations(raw)  # raw.find_edf_events()
            
            cond = False
            for cl in class_ids: cond += (events_val[:,1]==cl) 
            idx = np.ravel(np.where(cond)) # idx mantém os indices das amostras que marcam a apresentação das dicas são apresentadas nas trials ao longo da sessão
            labels = events_val[idx,1]
        
            #### EASY
            smin, smax = int(tmin*Fs), int(tmax*Fs)
            s0 = events_val[idx,0] + smin # cue moment
            sn = s0 + smax #int(4*Fs)    # start break moment 
        
            #### MODERATE
            # s0 = events_val[idx,0] - int(3*Fs) # start trial moment
            # sn = s0 + int(9*Fs)      # min end break moment # int(min(np.diff(s0))) # menor tempo entre os inicios de dois trials = 7.58s ou 1895 amostras
        
            #### HARD
            # s0 = events_val[idx,0] - (3*Fs) # start trial moment
            # sn = [ s0[i] for i in range(1, len(s0), 1) ] # end break moment 
            # sn.append(data_val.shape[-1] - (Fs*5)) # add last end break moment (5 seg antes do sinal acabar)
            # sn = np.asarray(sn)
            
            trials = [ data_val[:, s0[i]:sn[i]] for i in range(len(labels)) ]
            trials_a = np.asarray(trials)[np.where(labels == class_ids[0])]
            trials_b = np.asarray(trials)[np.where(labels == class_ids[1])]
            
            SA, SB = trials_a[0], trials_b[0]         
            for i in range(1, len(trials_a)): SA = np.c_[ SA, trials_a[i] ]
            for i in range(1, len(trials_b)): SB = np.c_[ SB, trials_b[i] ]
            
            delta_t = 0.2 # deslocamento em segundos
            delta_s = int(delta_t * Fs) # deslocamento em amostras
            delay = int(5/delta_t)  # 2/delta_t == 2 seg == x épocas, representa épocas "perdidas" entre partidas
            
            n_rounds = 6        # 6, número de partidas jogadas
            round_time = 40     # 40, simula tempo (segundos) que o asteroid leva para percorrer a tela
            
            # no_asteroidx = [ np.random.choice(class_ids) for i in range(n_rounds) ] # representa os comandos desejados a cada partida (fuga do asteróidd, lado oposto da tela de onde ele surge)
            no_asteroidx = [1,1,1,2,2,2]; random.shuffle(no_asteroidx)
            # print(no_asteroidx, '\n')
            
            limiar_gatilho = 0.9
            
            # data_val, events_val = None, None
            
            ##%% ###########################################################################
            #### CONSTRUIDO BUFFER DESLIZANTE - LARGURA OTIMIZADA                       ####
            ##%% ###########################################################################
            
            q = int(tmax*Fs) - int(tmin*Fs) # 500 tamanho do buffer
            
            # constrói épocas do sinal continuo usando as dimensões do buffer (q) e o deslocamento (simula o deslizar do buffer)
            buffers_a = np.asarray([ SA[:, i-q:i] for i in range(q, SA.shape[-1], delta_s) ]) # n_
            buffers_b = np.asarray([ SB[:, i-q:i] for i in range(q, SB.shape[-1], delta_s) ])
        
            ZV, tv = [], [] # acumulador de buffers/épocas e respectivos rótulos para validação
            cont_a, cont_b = delay, delay # contadores de buffers/épocas A, B adicionadas ao acumulador para validação
            for i in range(n_rounds): 
                samples = int(round_time/delta_t)   # 40/0.2 = 200, número de janelas avaliadas na partida 
                if no_asteroidx[i] == class_ids[0]: 
                    ZV.append(buffers_a[cont_a : cont_a + samples]) # add buffers/épocas da classe A no acumulador
                    cont_a += (samples + delay) # incrementa o contador de buffers/épocas A usadas
                else:
                    ZV.append(buffers_b[cont_b : cont_b + samples]) # add buffers/épocas da classe A no acumulador
                    cont_b += (samples + delay) # incrementa o contador de buffers/épocas A usadas
                tv.append(no_asteroidx[i] * np.ones(samples))
            
            if n_rounds > 1: ZV, tv = np.vstack(ZV), np.ravel(tv).astype(int) # formatando para validação dos modelos para todas as partidas de uma única vez
            else: ZV, tv = ZV[0], tv[0].astype(int) 
            
            ################################ AS ON-LINE ##################################
            
            ZT, _, tt, _ = half_split_data(data, events, int(tmin*Fs), int(tmax*Fs), class_ids)
            as_on, LA = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=info['fs'], filt='DFT')
            # as_online, y, p = tester(ZV, tv, Fs, class_ids, hmax) # (TREINO SOMENTE E)
            as_on = round(as_on*100,2); # print(f"ASon : {as_on}") 
            
            #### GERAÇÃO DE COMANDO (AS)
            
            y, p = LA['y'], LA['yp']
            tta = tmax-tmin # tempo de ação em segundos ### tempo de ação menor == mais possibilidades de comandos durante a partida, e vice-versa
            nta = int(tta/delta_t) # tempo de ação em número de épocas no buffer circular (amostras passadas)
            cont_com_a, cont_com_b, cont_no_com = 0, 0, 0
            as_comlist = []
            
            i = nta
            while (i < len(y)): # for i in range(nta, len(y), 1):
                A = y[i-nta:i] #buffer circular externo
                U1 = list(A).count(class_ids[0]) # conta as classificações A no buffer
                U2 = list(A).count(class_ids[1]) # conta as classificaçòes B no buffer
                U1_prop = U1 / nta # proporção de classificações A no buffer
                U2_prop = U2 / nta # proporção de classificações B no buffer
                if (U1_prop >= limiar_gatilho): #  and (tv[i]==class_ids[0])
                    as_comlist.append(np.array([class_ids[0], i])) # [ comando, momento do envio (indice época)]
                    # cont_com_a += 1; rota += '< '; # print([i-nta,i])
                    i += nta # como o comando foi enviado o buffer SALTA, iniciando a partir da época do comando na próxima iteração
                if (U2_prop >= limiar_gatilho): #  and (tv[i]==class_ids[1])
                    as_comlist.append(np.array([class_ids[1], i]))
                    # cont_com_b += 1; rota += '> '; # print([i-nta,i])
                    i += nta # se um comando foi enviado, o próximo A se inicia no instante que o comando foi enviado (sem sobreposição) 
                if (U1_prop < limiar_gatilho) and (U2_prop < 0.9): 
                    # cont_no_com += 1; 
                    i += 1 # se nenhum comando foi enviado, o próximo A se inicia deslocado de delta_t em relação ao inicio do A anterior (sobrepopsição de nta-1)
            as_comlist = np.asarray(as_comlist)
            as_comlist = np.c_[as_comlist, np.zeros(len(as_comlist))].astype(int)
            
            for i in range(len(as_comlist)): 
                if as_comlist[i,0] == tv[as_comlist[i,1]]: as_comlist[i, 2] = 1 
            
            as_command = round(np.mean(as_comlist[:, 0] == tv[as_comlist[:,1]])*100,2)
            # taxa de acerto de comando == comando enviado / comando esperado : corretos_cont/len(command_list) ou np.mean(command_list[:, 0] == tv[command_list[:,1]])
            # print(no_asteroidx, cont_com_a, cont_com_b, cont_no_com)
            # print(rota); print(corretos_list)
            # print(f'Comando via Acum. : {as_command} | n_comandos: {cont_com_a + cont_com_b} | n_corretos: {corretos_cont}')

            # t_ = np.asarray([ tv[i] for i in range(nta, len(y), nta) ]) # tv[0::nta].astype(int)
            # m_ = np.asarray([ mode(y[i-nta:i])[0][0] for i in range(nta, len(y), nta) ])                 
            # p_ = np.asarray([ np.argmax(np.mean((p[i-nta:i]), axis=0)) + 1 for i in range(nta, len(p), nta) ])
            # print(f'Comando via Moda  : {round(np.mean(m_ == t_)*100,2)} | n_comandos: {len(m_)}')
            # print(f'Comando via PMedia: {round(np.mean(p_ == t_)*100,2)} | n_comandos: {len(p_)}\n')
            
            
            ##%% ###########################################################################
            #### CONSTRUIDO BUFFER DESLIZANTE - LARGURA FIXA 2s                         ####
            ##%% ###########################################################################
            
            smin, smax = int(0.5*Fs), int(2.5*Fs)
            q = smax - smin # tamanho do buffer
            
            # cond = False
            # for cl in class_ids: cond += (events_val[:,1]==cl) 
            # idx = np.ravel(np.where(cond)) # idx mantém os indices das amostras que marcam a apresentação das dicas são apresentadas nas trials ao longo da sessão
            # labels = events_val[idx,1]
            s0 = events_val[idx,0] + smin # cue moment
            sn = s0 + smax #int(4*Fs)    # start break moment 
            trials = [ data_val[:, s0[i]:sn[i]] for i in range(len(labels)) ]
            trials_a = np.asarray(trials)[np.where(labels == class_ids[0])]
            trials_b = np.asarray(trials)[np.where(labels == class_ids[1])]
            SA, SB = trials_a[0], trials_b[0]         
            for i in range(1, len(trials_a)): SA = np.c_[ SA, trials_a[i] ]
            for i in range(1, len(trials_b)): SB = np.c_[ SB, trials_b[i] ]
            
            # constrói épocas do sinal continuo usando as dimensões do buffer (q) e o deslocamento (simula o deslizar do buffer)
            buffers_a = np.asarray([ SA[:, i-q:i] for i in range(q, SA.shape[-1], delta_s) ]) # n_
            buffers_b = np.asarray([ SB[:, i-q:i] for i in range(q, SB.shape[-1], delta_s) ])
        
            ZV, tv = [], [] # acumulador de buffers/épocas e respectivos rótulos para validação
            cont_a, cont_b = delay, delay # contadores de buffers/épocas A, B adicionadas ao acumulador para validação
            for i in range(n_rounds): 
                samples = int(round_time/delta_t)   # 200, número de janelas avaliadas na partida 
                if no_asteroidx[i] == class_ids[0]: 
                    ZV.append(buffers_a[cont_a : cont_a + samples]) # add buffers/épocas da classe A no acumulador
                    cont_a += (samples + delay) # incrementa o contador de buffers/épocas A usadas
                else:
                    ZV.append(buffers_b[cont_b : cont_b + samples]) # add buffers/épocas da classe A no acumulador
                    cont_b += (samples + delay) # incrementa o contador de buffers/épocas A usadas
                tv.append(no_asteroidx[i] * np.ones(samples))
        
            if n_rounds > 1: ZV, tv = np.vstack(ZV), np.ravel(tv).astype(int) # formatando para validação dos modelos para todas as partidas de uma única vez
            else: ZV, tv = ZV[0], tv[0].astype(int) 
        
            ################################ SBCSP ON-LINE ################################
            
            ZT, _, tt, _ = half_split_data(data, events, smin, smax, class_ids)
            sb_on, LS = sbcsp_approach(ZT, ZV, tt, tv, 9, 4, 40, 8, {'model':'SVM','kernel':{'kf':'linear'},'C':-4}, Fs=info['fs'], filt='IIR')
            sb_on = round(sb_on*100,2); # print(f"SBon : {sb_on}");
            
            #### GERAÇÃO DE COMANDO (SBCSP)
            
            ys, ps = LS['y'], LS['yp']
            tta = tmax-tmin # tempo de ação em segundos ### tempo de ação menor == mais possibilidades de comandos durante a partida, e vice-versa
            nta = int(tta/delta_t) # tempo de ação em número de épocas no buffer circular (amostras passadas)
            cont_com_a, cont_com_b, cont_no_com = 0, 0, 0
            sb_comlist, rota = [], ''
            
            i = nta
            while (i < len(ys)): # for i in range(nta, len(ys), 1):
                A = ys[i-nta:i] #buffer circular externo
                U1 = list(A).count(class_ids[0]) # conta as classificações A no buffer
                U2 = list(A).count(class_ids[1]) # conta as classificaçòes B no buffer
                U1_prop = U1 / nta # proporção de classificações A no buffer
                U2_prop = U2 / nta # proporção de classificações B no buffer
                if (U1_prop >= limiar_gatilho): #  and (tv[i]==class_ids[0])
                    sb_comlist.append(np.array([class_ids[0], i])) # [ comando, momento do envio (indice época)]
                    cont_com_a += 1; rota += '< '; # print([i-nta,i])
                    i += nta # como o comando foi enviado o buffer SALTA, iniciando a partir da época do comando na próxima iteração
                if (U2_prop >= limiar_gatilho): #  and (tv[i]==class_ids[1])
                    sb_comlist.append(np.array([class_ids[1], i]))
                    cont_com_b += 1; rota += '> '; # print([i-nta,i])
                    i += nta # se um comando foi enviado, o próximo A se inicia no instante que o comando foi enviado (sem sobreposição) 
                if (U1_prop < limiar_gatilho) and (U2_prop < 0.9): 
                    cont_no_com += 1; i += 1 # se nenhum comando foi enviado, o próximo A se inicia deslocado de delta_t em relação ao inicio do A anterior (sobrepopsição de nta-1)
            sb_comlist = np.asarray(sb_comlist)
            sb_comlist = np.c_[sb_comlist, np.zeros(len(sb_comlist))].astype(int)
            
            corretos_cont = 0; corretos_list = ''
            for i in range(len(sb_comlist)): 
                if sb_comlist[i,0] == tv[sb_comlist[i,1]]: sb_comlist[i,2] = 1 # corretos_cont += 1; corretos_list += '1 '
                # else: corretos_list += '0 '
            
            sb_command = round(np.mean(sb_comlist[:, 0] == tv[sb_comlist[:,1]])*100,2)
            # taxa de acerto de comando == comando enviado / comando esperado : corretos_cont/len(command_list) ou np.mean(command_list[:, 0] == tv[command_list[:,1]])
            # print(no_asteroidx, cont_com_a, cont_com_b, cont_no_com)
            # print(rota); print(corretos_list)
            # print(f'Comando via Acum. : {round(np.mean(command_list[:, 0] == tv[command_list[:,1]])*100,2)} | n_comandos: {cont_com_a + cont_com_b} | n_corretos: {corretos_cont}')

            # t_ = np.asarray([ tv[i] for i in range(nta, len(ys), nta) ]) # tv[0::nta].astype(int)
            # m_ = np.asarray([ mode(ys[i-nta:i])[0][0] for i in range(nta, len(ys), nta) ])                 
            # p_ = np.asarray([ np.argmax(np.mean((ps[i-nta:i]), axis=0)) + 1 for i in range(nta, len(ps), nta) ])
            # print(f'Comando via Moda  : {round(np.mean(m_ == t_)*100,2)} | n_comandos: {len(m_)}')
            # print(f'Comando via PMedia: {round(np.mean(p_ == t_)*100,2)} | n_comandos: {len(p_)}\n')
            

            # ########################### CLASSIC ON-LINE ################################
            
            ZT, _, tt, _ = half_split_data(data, events, smin, smax, class_ids)
            cla_on, LC = classic_approach(ZT, ZV, tt, tv, 8, 30, 8, {'model':'LDA'}, Fs=info['fs'], filt='IIR')
            cla_on = round(cla_on*100,2) # print(f"CLAon: {cla_on}")
            
            #### GERAÇÃO DE COMANDO (CLASSIC)
            
            yc, pc = LC['y'], LC['yp']
            tta = 2 # tempo de ação em segundos ### tempo de ação menor == mais possibilidades de comandos durante a partida, e vice-versa
            nta = int(tta/delta_t) # tempo de ação em número de épocas no buffer circular (amostras passadas)
            cont_com_a, cont_com_b, cont_no_com = 0, 0, 0
            cla_comlist, rota_cla_iir = [], ''
            
            i = nta
            while (i < len(yc)): # for i in range(nta, len(yc), 1):
                A = yc[i-nta:i] #buffer circular externo
                U1 = list(A).count(class_ids[0]) # conta as classificações A no buffer
                U2 = list(A).count(class_ids[1]) # conta as classificaçòes B no buffer
                U1_prop = U1 / nta # proporção de classificações A no buffer
                U2_prop = U2 / nta # proporção de classificações B no buffer
                if (U1_prop >= limiar_gatilho): #  and (tv[i]==class_ids[0])
                    cla_comlist.append(np.array([class_ids[0], i])) # [ comando, momento do envio (indice época)]
                    cont_com_a += 1; rota_cla_iir += '< '; # print([i-nta,i])
                    i += nta # como o comando foi enviado o buffer SALTA, iniciando a partir da época do comando na próxima iteração
                if (U2_prop >= limiar_gatilho): #  and (tv[i]==class_ids[1])
                    cla_comlist.append(np.array([class_ids[1], i]))
                    cont_com_b += 1; rota_cla_iir += '> '; # print([i-nta,i])
                    i += nta # se um comando foi enviado, o próximo A se inicia no instante que o comando foi enviado (sem sobreposição) 
                if (U1_prop < limiar_gatilho) and (U2_prop < 0.9): 
                    cont_no_com += 1; i += 1 # se nenhum comando foi enviado, o próximo A se inicia deslocado de delta_t em relação ao inicio do A anterior (sobrepopsição de nta-1)
            cla_comlist = np.asarray(cla_comlist)
            cla_comlist = np.c_[cla_comlist, np.zeros(len(cla_comlist))].astype(int)
            
            corretos_cont_cla_iir = 0; corretos_list_cla_iir = ''
            for i in range(len(cla_comlist)): 
                if cla_comlist[i,0] == tv[cla_comlist[i,1]]: cla_comlist[i,2] = 1 # corretos_cont_cla_iir += 1; corretos_list_cla_iir += '1 '
                # else: corretos_list_cla_iir += '0 '
            
            # taxa de acerto de comando == comando enviado / comando esperado : corretos_cont/len(command_list) ou np.mean(command_list[:, 0] == tv[command_list[:,1]])
            # print(no_asteroidx, cont_com_a, cont_com_b, cont_no_com)
            cla_command = round(np.mean(cla_comlist[:, 0] == tv[cla_comlist[:,1]])*100,2)
            # print(rota_cla_iir); print(corretos_list_cla_iir)
            # print(f'Comando via Acum. : {acc_cla_iir_acum} | n_comandos: {cont_com_a + cont_com_b} | n_corretos: {corretos_cont}')

            # t_ = np.asarray([ tv[i] for i in range(nta, len(yc), nta) ]) # tv[0::nta].astype(int)
            # m_ = np.asarray([ mode(yc[i-nta:i])[0][0] for i in range(nta, len(yc), nta) ])                 
            # p_ = np.asarray([ np.argmax(np.mean((pc[i-nta:i]), axis=0)) + 1 for i in range(nta, len(pc), nta) ])
            # print(f'Comando via Moda  : {round(np.mean(m_ == t_)*100,2)} | n_comandos: {len(m_)}')
            # print(f'Comando via PMedia: {round(np.mean(p_ == t_)*100,2)} | n_comandos: {len(p_)}\n')
            

            # # z = np.asarray([tv, y, ys, yc]).T
            
            R.loc[0] = [suj, class_ids[0], class_ids[1], fl, fh, tmin, tmax, nbands, ncsp, csp_list, clf['model'], clf,
                                      as_max, as_tune, as_mode, as_pmean, as_best, sb_iir, cla_iir, as_on, sb_on, cla_on, 
                                      as_command, sb_command, cla_command, as_comlist, sb_comlist, cla_comlist]
            
            
            
            pd.to_pickle(R, '/home/vboas/cloud/results/' + ds + '_cortex/R_' + ds + '_' + str(suj) + '.pkl')
            
            data, data_cal, data_val = None, None, None
            events, events_cal, events_val = None, None, None
            
            # del globals()['events_cal']; del globals()['data_cal'] del globals()['events'] del globals()['trials'] del globals()['space']
            
    

    
        
    