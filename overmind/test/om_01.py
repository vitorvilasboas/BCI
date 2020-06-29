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
import random

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

H = pd.DataFrame(columns=['fl','fh','tmin','tmax','ncsp','nbands','clf_model','csp','lda','p0','p1','clf_final','t','y','yp','acc','clf'])

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
        bin_size = 2/(Fs/ZT.shape[-1]) # 2 representa sen e cos
        XT = ZTF[:, :, round(fl*bin_size):round(fh*bin_size)]
        
        ZVF = fft(ZV)
        REAL = np.transpose(np.real(ZVF), (2, 0, 1))
        IMAG = np.transpose(np.imag(ZVF), (2, 0, 1))
        ZVF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        ZVF = np.transpose(ZVF, (1, 2, 0))
        bin_size = 2/(Fs/ZV.shape[-1]) # 2 representa sen e cos
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
    
    # csp = CSP(n_components=int(self.ncsp)) # CSP 1
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
        bin_size = (2/(Fs/ZT.shape[-1])) # 2 representa sen e cos
        XT = [ ZTF[:, :, round(sub_bands[i][0]*bin_size):round(sub_bands[i][1]*bin_size)] for i in range(nbands) ]
        
        ZVF = fft(ZV)
        REAL = np.transpose(np.real(ZVF), (2, 0, 1))
        IMAG = np.transpose(np.imag(ZVF), (2, 0, 1))
        ZVF = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
        ZVF = np.transpose(ZVF, (1, 2, 0))
        bin_size = (2/(Fs/ZV.shape[-1])) # 2 representa sen e cos
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
        # csp_filters_sblist = [ chain[i]['CSP'].filters_ for i in range(nbands) ] # CSP 1
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
    acc, learner = np.mean(cross_acc), cross_learner[np.argmax(cross_acc)]

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
    acc, learner = np.mean(cross_acc), cross_learner[np.argmax(cross_acc)]
    
    H.loc[len(H)] = [f_low, f_high, tmin, tmax, ncsp, nbands, clf['model'], learner['csp_filt'], learner['lda'], learner['p0'], 
                     learner['p1'], learner['clf_final'], learner['t'], learner['y'], learner['yp'], acc, clf]
    
    return acc * (-1)


def tester(ZV, tv, Fs, class_ids, setup):
    fl, fh, nbands = setup['fl'], setup['fh'], setup['nbands']
    p0, p1, filters_, lda, clf_final = setup['p0'], setup['p1'], setup['csp'], setup['lda'], setup['clf_final'] 
    if nbands > (fh-fl): nbands = (fh-fl)
    # print(len(filters_), nbands)
    # smin, smax = math.floor(tmin * info['fs']), math.floor(tmax * info['fs'])
    # ZV, tv = extractEpochs(data, events, smin, smax, class_ids)

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


if __name__ == "__main__":
    n_iter = 100
    n_iter_tune = 10
    subjects = [5] # range(1,10) 
    classes = [[1, 2]]#, [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] # 
    R = pd.DataFrame(columns=['subj','A','B','as_best','sb_iir','cla_iir'])
    for suj in subjects:
        for class_ids in classes:
            H = H.iloc[0:0] # cleaning df
    
            data_cal, events_cal, info_cal = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'T.npy', allow_pickle=True)
            data_val, events_val, info_val = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'E.npy', allow_pickle=True)
            
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
        
            path = '/home/vboas/cloud/devto/overmind/tests/'
            path_to_trials = path + 'A0' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
            if not os.path.isdir(path): os.makedirs(path)
            
            trials = base.Trials() 
            # try: trials = pickle.load(open(path_to_trials, 'rb'))
            # except: trials = base.Trials()
            # trials = generate_trials_to_calculate(init_vals)
            init_vals = [{'fl':4,'fh':40,'tmin':0.5,'tmax':2.5,'ncomp':8,'nbands':9,'model':'SVM','C':1e-4,'kf':'linear'}] 
            try:
                # print('N trials: ' + str(len(trials)))
                fmin_objective = partial(objective, data=data_cal, events=events_cal, info=info_cal, class_ids=class_ids)
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
                
            # ##%% #########################################################################
            desvio = 4 # desvio em torno do ncsp ótimo (deve ser par)
            min_ncsp = (ncsp - desvio) if (ncsp - desvio) > 2 else 2
            max_ncsp = (ncsp + desvio) if (ncsp + desvio) < 22 else 22
            space_tune = tuple([ hp.quniform('csp'+str(i), min_ncsp, max_ncsp, 2) for i in range(nbands) ])
            fmin_tunning = partial(csp_tunning, data=data_cal, events=events_cal, info=info_cal, class_ids=class_ids, 
                                    setup={'fl':fl, 'fh':fh, 'tmin':tmin, 'tmax':tmax, 'ncomp':ncsp, 'nbands':nbands, 'reg':reg, 'clf':clf})
            trials_tune = base.Trials()
            init_vals_tune = {'csp0':int(ncsp)}
            for i in range(1, nbands-1): init_vals_tune.update({'csp'+str(i):int(ncsp)})
            try: best_tune = fmin(fmin_tunning, space=space_tune, algo=tpe.suggest, max_evals=n_iter_tune, trials=trials_tune, points_to_evaluate=[init_vals_tune])
            except: print('Exception raised'); raise   
            acc_train_tune = round((-1) * trials_tune.best_trial['result']['loss'] * 100, 2)
            csp_list = [ int(best_tune['csp'+str(i)]) for i in range(nbands) ]
            # ##%% #########################################################################
            
            hmax = H[ H['acc'] == H['acc'].max()].iloc[0]
            ZV, tv = extractEpochs(data_val, events_val, math.floor(hmax['tmin']*info_val['fs']), math.floor(hmax['tmax']*info_val['fs']), class_ids)
            acc_max_E, yv, ypv = tester(ZV, tv, info_val['fs'], class_ids, hmax)
            acc_max_E = round(acc_max_E*100,2)
            
            # H = H.sort_values(by='acc', ascending=False)
            # H = H.iloc[0:50]
            # H = H.reset_index(drop=True)
            
            V, P = [], []
            for i in range(len(H)):
                ZV, tv = extractEpochs(data_val, events_val, math.floor(H.iloc[i]['tmin']*info_val['fs']), math.floor(H.iloc[i]['tmax']*info_val['fs']), class_ids)
                acc_test, yv, ypv = tester(ZV, tv, info_val['fs'], class_ids, H.iloc[i])
                V.append(yv)
                P.append(ypv)
            
            ### Voting
            V = np.asarray(V).T
            yvm = np.asarray([mode(V[i])[0][0] for i in range(len(V))], dtype=int) 
            acc_mode_E = round(np.mean(yvm == tv)*100,2)
            
            ### Averaging
            PM = np.mean(np.transpose(P, (1,2,0)), axis=2)
            ypv = np.asarray([ class_ids[0] if (PM[p][0]>=PM[p][1]) else class_ids[1] for p in range(len(PM))], dtype=int)
            acc_pmean_E = round(np.mean(ypv == tv)*100,2)
            
            
            #######################################################
            data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + '.npy', allow_pickle=True)
            #######################################################
            
            V, P = [], []
            for i in range(len(H)):
                setup = H.iloc[i]
                smin, smax = math.floor(setup['tmin'] * info['fs']), math.floor(setup['tmax'] * info['fs'])
            
                epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
                epochs = nanCleaner(epochs)
                epochsT, labelsT = epochs[:(len(epochs)//2)], labels[:(len(epochs)//2)] 
                epochsV, labelsV = epochs[(len(epochs)//2):], labels[(len(epochs)//2):]
                epochsT = [ epochsT[np.where(labelsT == i)] for i in class_ids ] 
                epochsV = [ epochsV[np.where(labelsV == i)] for i in class_ids ]
                ZT = np.concatenate([epochsT[0],epochsT[1]])
                ZV = np.concatenate([epochsV[0],epochsV[1]])
                tt = np.concatenate([class_ids[0]*np.ones(len(ZT)//2), class_ids[1]*np.ones(len(ZT)//2)])
                tv = np.concatenate([class_ids[0]*np.ones(len(ZV)//2), class_ids[1]*np.ones(len(ZV)//2)])
                
                acc, learner = sbcsp_approach(ZT, ZV, tt, tv, setup['nbands'], setup['fl'], setup['fh'], setup['ncsp'], setup['clf'], Fs=info['fs'], filt='DFT')
                tv = learner['t']
                V.append(learner['y'])
                P.append(learner['yp'])
            
            ### Voting
            V = np.asarray(V).T
            yvm = np.asarray([mode(V[i])[0][0] for i in range(len(V))], dtype=int)
            acc_mode_ET = round(np.mean(yvm == tv)*100,2)
            
            ### Averaging
            PM = np.mean(np.transpose(P, (1,2,0)), axis=2)
            ypv = np.asarray([ class_ids[0] if (PM[p][0]>=PM[p][1]) else class_ids[1] for p in range(len(PM))], dtype=int)
            acc_pmean_ET = round(np.mean(ypv == tv)*100,2)
            
            #############################################################################
            
            smin, smax = math.floor(tmin * info['fs']), math.floor(tmax * info['fs'])
            
            epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
            epochs = nanCleaner(epochs)
            epochsT, labelsT = epochs[:(len(epochs)//2)], labels[:(len(epochs)//2)] 
            epochsV, labelsV = epochs[(len(epochs)//2):], labels[(len(epochs)//2):]
            epochsT = [ epochsT[np.where(labelsT == i)] for i in class_ids ] 
            epochsV = [ epochsV[np.where(labelsV == i)] for i in class_ids ]
            ZT = np.concatenate([epochsT[0],epochsT[1]])
            ZV = np.concatenate([epochsV[0],epochsV[1]])
            tt = np.concatenate([class_ids[0]*np.ones(len(ZT)//2), class_ids[1]*np.ones(len(ZT)//2)])
            tv = np.concatenate([class_ids[0]*np.ones(len(ZV)//2), class_ids[1]*np.ones(len(ZV)//2)])
            
            acc_max_ET, _ = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=info['fs'], filt='DFT')     
            acc_max_ET = round(acc_max_ET*100,2)
            
            acc_tune_ET, _ = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=info['fs'], filt='DFT', csp_list=csp_list)     
            acc_tune_ET = round(acc_tune_ET*100,2)
            
            #######################################################
            
            smin, smax = math.floor(0.5 * info['fs']), math.floor(2.5 * info['fs'])
            
            epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
            epochs = nanCleaner(epochs)
            epochsT, labelsT = epochs[:(len(epochs)//2)], labels[:(len(epochs)//2)] 
            epochsV, labelsV = epochs[(len(epochs)//2):], labels[(len(epochs)//2):]
            epochsT = [ epochsT[np.where(labelsT == i)] for i in class_ids ] 
            epochsV = [ epochsV[np.where(labelsV == i)] for i in class_ids ]
            ZT = np.concatenate([epochsT[0],epochsT[1]])
            ZV = np.concatenate([epochsV[0],epochsV[1]])
            tt = np.concatenate([class_ids[0]*np.ones(len(ZT)//2), class_ids[1]*np.ones(len(ZT)//2)])
            tv = np.concatenate([class_ids[0]*np.ones(len(ZV)//2), class_ids[1]*np.ones(len(ZV)//2)])
            
            sb_iir, _ = sbcsp_approach(ZT, ZV, tt, tv, 9, 4, 40, 8, {'model':'SVM','kernel':{'kf':'linear'},'C':-4}, Fs=info['fs'], filt='IIR')
            cla_iir, _ = classic_approach(ZT, ZV, tt, tv, 8, 30, 8, {'model':'LDA'}, Fs=info['fs'], filt='IIR')
            
            acc_best = max(acc_max_E, acc_mode_E, acc_pmean_E, acc_tune_ET, acc_max_ET, acc_mode_ET, acc_pmean_ET)
            
            print(f"{fl}-{fh}Hz; {tmin}-{tmax}s; Ns={nbands}; R={ncsp} {csp_list}; {clf}")
            print(f"Max: {acc_max_E}  {acc_max_ET}")
            print(f"Moda: {acc_mode_E}  {acc_mode_ET}")
            print(f"Media: {acc_pmean_E}  {acc_pmean_ET}")
            print(f"Tune: {acc_tune_ET}")
            print(f"BEST: {acc_best}")
            print(f"SB: {round(sb_iir*100,2)}")
            print(f"CLA: {round(cla_iir*100,2)}")
            
            R.loc[len(R)] = [suj, class_ids[0], class_ids[1], acc_best, sb_iir, cla_iir]
    
            ###############################################################################
    
            # print(hmax['fl'], hmax['fh'], hmax['tmin'], hmax['tmax'], hmax['nbands'], hmax['ncsp'], hmax['clf'], hmax['acc'])                                                             
            # print(fl, fh, tmin, tmax, nbands, ncsp, clf)
    
    
    
    
    
    
    
    #         data_val, events_val, info_val = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'E.npy', allow_pickle=True)
               
    
    #         #######  ON-LINE SIMULATION - BUFFER ###################### 
            
    #         cond = False
    #         for cl in class_ids: cond += (events_val[:,1]==cl) 
    #         idx = np.ravel(np.where(cond))
    #         labels = events_val[idx,1]
            

    #         ##### EASY
    #         # s0 = events_val[idx,0]
    #         # sn = s0 + math.floor(4 * info_val['fs']) # min(np.diff(s0))/Fs ## menor tempo entre os inicios de dois trials = 7.58s ou 1895 amostras
            
    #         #### MODERATE
    #         s0 = events_val[idx-1,0] # indice amostra que inicia cada trial
    #         sn = s0 + math.floor(7 * info_val['fs']) # math.floor(min(np.diff(s0)))
            
    #         #### HARD
    #         # sn = [ s0[i] for i in range(1, len(s0), 1) ]
    #         # sn.append(data_val.shape[-1] - (info_val['fs']*5))
    #         # sn = np.asarray(sn)
            
            
    #         trials = [data_val[:, s0[i]:sn[i]] for i in range(len(labels))]
            
    #         sa = np.asarray([ s0[np.where(labels == class_ids[0])], sn[np.where(labels == class_ids[0])] ]).T
    #         sb = np.asarray([ s0[np.where(labels == class_ids[1])], sn[np.where(labels == class_ids[1])] ]).T
            
    #         SA = data_val[:, sa[0,0]:sa[0,1]]
    #         for i in range(1, len(sa)): SA = np.c_[ SA, data_val[:, sa[i,0]:sa[i,1]] ]
            
    #         SB = data_val[:, sb[0,0]:sb[0,1]]
    #         for i in range(1, len(sb)): SB = np.c_[ SB, data_val[:, sb[i,0]:sb[i,1]] ]
            
            
    #         asteroidx = [ np.random.choice(class_ids) for i in range(5) ]
            
            
            
            
            
            
            
    #         #### CONSTRUIDO BUFFER COM EPOCAS DE LARGURA OTIMIZADA ############### 
            
    #         q = math.floor(tmax * info['fs']) - math.floor(tmin * info['fs']) # 500 ## # tamanho do buffer
    #         delta_t = 0.2 # deslocamento do buffer em segundos
            
    #         delta_s = int(delta_t * info_val['fs']) # deslocamento do buffer em amostras
            
    #         buffer_a = np.asarray([ SA[:, i-q:i] for i in range(q, SA.shape[-1], delta_s) ])
    #         buffer_b = np.asarray([ SB[:, i-q:i] for i in range(q, SB.shape[-1], delta_s) ])
            
    #         delay = int(2/delta_t)  # 2 segundos
    #         cont_a, cont_b = delay, delay
    #         ZV, tv = [], []
    #         for i in range(5): # número de partidas
    #             tempo = 40 # em segundos (simula tempo que o asteroid percorre a tela)
    #             samples = int(tempo/delta_t)
    #             t = asteroidx[i]
    #             # print(t)
                
    #             if t == class_ids[0]: 
    #                 cont = cont_a
    #                 buff = np.copy(buffer_a) 
    #                 cont_a += (samples + delay)
    #             else:
    #                 cont = cont_b
    #                 buff = np.copy(buffer_b)
    #                 cont_b += (samples + delay)
                    
    #             buff = buff[cont : cont + samples]
                
    #             ZV.append(buff)
    #             tv.append(t * np.ones(len(buff)))
            
    #         ZV, tv = np.vstack(ZV), np.ravel(tv)
            
    #         ct = tv[0::10].astype(int)
            
            
            
            
            
            
    #         ################## AS ON-LINE TREINADO SOMENTE E ######################
            
    #         acc_test, yv, ypv = tester(ZV, tv, info_val['fs'], class_ids, hmax)
            
    #         cy = np.asarray([ mode(yv[i:i+10])[0][0] for i in range(0, len(yv), 10) ])
            
    #         # O yv[i:i+10] PODE SER USADO PARA OBTER O COMANDO E PLOTAR O SINAL DE COMANDO ACUMULADO
            
    #         cyp = np.asarray([ np.argmax(np.mean((ypv[i:i+10]), axis=0)) for i in range(0, len(ypv), 10) ])
    #         cyp = (cyp + np.ones(len(cyp))).astype(int)

    #         print('AS E:', (np.mean(cy == ct), np.mean(cyp == ct)))
            
    #         ################# AS ON-LINE TREINADO E e T ######################
            
    #         data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + '.npy', allow_pickle=True)
            
    #         smin, smax = math.floor(tmin * info['fs']), math.floor(tmax * info['fs'])
            
    #         epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    #         epochs = nanCleaner(epochs)
    #         epochsT, labelsT = epochs[:(len(epochs)//2)], labels[:(len(epochs)//2)] 
    #         epochsV, labelsV = epochs[(len(epochs)//2):], labels[(len(epochs)//2):]
    #         epochsT = [ epochsT[np.where(labelsT == i)] for i in class_ids ] 
    #         epochsV = [ epochsV[np.where(labelsV == i)] for i in class_ids ]
    #         ZT = np.concatenate([epochsT[0],epochsT[1]])
    #         tt = np.concatenate([class_ids[0]*np.ones(len(ZT)//2), class_ids[1]*np.ones(len(ZT)//2)])
            
    #         acc_max_test, learner = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=info['fs'], filt='DFT')     
            
    #         yv0 = learner['y']
    #         cy0 = np.asarray([ mode(yv0[i:i+10])[0][0] for i in range(0, len(yv0), 10) ])
            
    #         ypv0 = learner['yp']
    #         cyp0 = np.asarray([ np.argmax(np.mean((ypv0[i:i+10]), axis=0)) for i in range(0, len(ypv0), 10) ])
    #         cyp0 = (cyp0 + np.ones(len(cyp0))).astype(int)
            
    #         print('AS ET:', (np.mean(cy0 == ct), np.mean(cyp0 == ct)))
            
            
    #         ################
            
    #         #### CONSTRUIDO BUFFER COM EPOCAS DE LARGURA FIXA 2s ############### 
            
            
    #         q = 500 ## # tamanho do buffer
    #         delta_t = 0.2 # deslocamento do buffer em segundos
            
    #         delta_s = int(delta_t * info_val['fs']) # deslocamento do buffer em amostras
            
    #         buffer_a = np.asarray([ SA[:, i-q:i] for i in range(q, SA.shape[-1], delta_s) ])
    #         buffer_b = np.asarray([ SB[:, i-q:i] for i in range(q, SB.shape[-1], delta_s) ])
            
    #         delay = int(2/delta_t)  # 2 segundos
    #         cont_a, cont_b = delay, delay
    #         ZV, tv = [], []
    #         for i in range(5): # número de partidas
    #             tempo = 40 # em segundos (simula tempo que o asteroid percorre a tela)
    #             samples = int(tempo/delta_t)
    #             t = asteroidx[i]
    #             # print(t)
                
    #             if t == class_ids[0]: 
    #                 cont = cont_a
    #                 buff = np.copy(buffer_a) 
    #                 cont_a += (samples + delay)
    #             else:
    #                 cont = cont_b
    #                 buff = np.copy(buffer_b)
    #                 cont_b += (samples + delay)
                    
    #             buff = buff[cont : cont + samples]
                
    #             ZV.append(buff)
    #             tv.append(t * np.ones(len(buff)))
            
    #         ZV, tv = np.vstack(ZV), np.ravel(tv)
            
    #         ct = tv[0::10].astype(int)
            
    #         ################## CLASSIC E SBCSP ON-LINE TREINADOS E e T ######################
            
    #         smin, smax = math.floor(0.5 * info['fs']), math.floor(2.5 * info['fs'])
    #         epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    #         epochs = nanCleaner(epochs)
    #         epochsT, labelsT = epochs[:(len(epochs)//2)], labels[:(len(epochs)//2)] 
    #         epochsV, labelsV = epochs[(len(epochs)//2):], labels[(len(epochs)//2):]
    #         epochsT = [ epochsT[np.where(labelsT == i)] for i in class_ids ] 
    #         epochsV = [ epochsV[np.where(labelsV == i)] for i in class_ids ]
    #         ZT = np.concatenate([epochsT[0],epochsT[1]])
    #         tt = np.concatenate([class_ids[0]*np.ones(len(ZT)//2), class_ids[1]*np.ones(len(ZT)//2)])
            
    #         sb_iir_test, _ = sbcsp_approach(ZT, ZV, tt, tv, 9, 4, 40, 8, {'model':'SVM','kernel':{'kf':'linear'},'C':-4}, Fs=info['fs'], filt='IIR')
    #         cla_iir_test, learner = classic_approach(ZT, ZV, tt, tv, 8, 30, 8, {'model':'LDA'}, Fs=info['fs'], filt='IIR')
            
    #         yv1 = learner['y']
    #         cy1 = np.asarray([ mode(yv1[i:i+10])[0][0] for i in range(0, len(yv1), 10) ])
            
    #         ypv1 = learner['yp']
    #         cyp1 = np.asarray([ np.argmax(np.mean((ypv1[i:i+10]), axis=0)) for i in range(0, len(ypv1), 10) ])
    #         cyp1 = (cyp1 + np.ones(len(cyp1))).astype(int)
    #         print('CLA:', (np.mean(cy1 == ct), np.mean(cyp1 == ct)))
            
            
    #         # print(round(acc_test*100,2), round(sb_iir_test*100,2), round(cla_iir_test*100,2))
            
            
    
    # print(R.iloc[:,3:].mean())