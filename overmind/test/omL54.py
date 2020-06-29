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

# np.seterr(divide='ignore', invalid='ignore')
# warnings.filterwarnings("ignore", category=DeprecationWarning)
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

def csp_tunning(args, data, labels, Fs, class_ids, setup):
    # csp_reg = setup['reg']
    smin = math.floor(setup['tmin'] * Fs)
    smax = math.floor(setup['tmax'] * Fs)
    
    epochs = data[:,:,smin:smax]
    
    # epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    # epochs = nanCleaner(epochs)
    
    # kf = StratifiedShuffleSplit(1, test_size=0.5, random_state=42)
    # for _train, _test in kf.split(epochs, labels):
    #     ZT, ZV, tt, tv = epochs[_train], epochs[_test], labels[_train], labels[_test]
    
    cross_acc, cross_learner = [], []
    kf = StratifiedShuffleSplit(5, test_size=0.2, random_state=42)
    for _train, _test in kf.split(epochs, labels):
        ZT, ZV, tt, tv = epochs[_train], epochs[_test], labels[_train], labels[_test]
        a, l = sbcsp_approach(ZT, ZV, tt, tv, setup['nbands'], setup['fl'], setup['fh'], setup['ncomp'], setup['clf'], 
                              Fs=Fs, filt='DFT', csp_list=list(map(lambda x: int(x), args)))
        cross_acc.append(a); cross_learner.append(l)
    acc, _ = np.mean(cross_acc), cross_learner[np.argmax(cross_acc)]

    return acc * (-1)

def objective(args, data, labels, Fs, class_ids):
    # filtering = {'design':'DFT'}
    # f_low, f_high, tmin, tmax, ncomp, nbands, reg = 0, 40, 0.5, 2.5, 8, 9, 1e-1
    # clf = {'model':'SVM','kernel':{'kf':'linear'},'C':-4}     
    f_low, f_high, tmin, tmax, ncsp, nbands, clf = args # reg
    f_low, f_high = int(f_low['fl']), int(f_high['fh']) # csp_reg = 10**(int(reg)) 
    if nbands > (f_high - f_low): nbands = (f_high - f_low)
    while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
    
    smin = math.floor(tmin * Fs)
    smax = math.floor(tmax * Fs)
    
    epochs = data[:,:,smin:smax]
    
    # epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
    # epochs = nanCleaner(epochs)
    
    cross_acc, cross_learner = [], []
    kf = StratifiedShuffleSplit(5, test_size=0.2, random_state=42)
    for _train, _test in kf.split(epochs, labels):
        ZT, ZV, tt, tv = epochs[_train], epochs[_test], labels[_train], labels[_test]
        a, l = sbcsp_approach(ZT, ZV, tt, tv, nbands, f_low, f_high, ncsp, clf, Fs=Fs, filt='DFT')     
        cross_acc.append(a); cross_learner.append(l)     
    acc, _ = np.mean(cross_acc), cross_learner[np.argmax(cross_acc)]
    
    H.loc[len(H)] = [f_low, f_high, tmin, tmax, ncsp, nbands, clf, acc]
    
    return acc * (-1)


H = pd.DataFrame(columns=['fl','fh','tmin','tmax','ncsp','nbands','clf','acc'])

if __name__ == "__main__":
    n_iter = 50
    n_iter_tune = n_iter//10
    subjects = range(1,2) 
    classes = [[1, 2]]

    R = pd.DataFrame(columns=['subj', 'A', 'B', 'fl', 'fh', 'tmin', 'tmax', 'nbands', 'ncsp', 'csp_list', 'clf', 'clf_details',
                              'as_max', 'as_tune', 'as_mode', 'as_pmean', 'as_best', 'sb_iir', 'cla_iir', 
                              'as_on', 'sb_on', 'cla_on', 'as_command', 'sb_command', 'cla_command',
                              'as_comlist', 'sb_comlist', 'cla_comlist'])
    
    ds = 'Lee19'
    for suj in subjects:
        for class_ids in classes:  
            H = H.iloc[0:0] # cleaning df
            
            ZT1, ZV1, ZT2, ZV2, tt1, tv1, tt2, tv2, info = np.load('/mnt/dados/eeg_data/Lee19/npy_epochs/S' + str(suj) + '.npy', allow_pickle=True)
            
            data_cal, labels_cal = np.r_[ZT1, ZT2], np.r_[tt1, tt2]
            data_val, labels_val = np.r_[ZV1, ZV2], np.r_[tv1, tv2]
            
            del (globals()['ZT1'], globals()['ZV1'], globals()['ZT2'], globals()['ZV2'], globals()['tt1'], globals()['tv1'], globals()['tt2'], globals()['tv2'])
        
            Fs = info['fs']
            filtering = 'DFT'

            print(f'###### {suj} {class_ids} ######')
            
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
                fmin_objective = partial(objective, data=data_cal, labels=labels_cal, Fs=Fs, class_ids=class_ids)
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
            fmin_tunning = partial(csp_tunning, data=data_cal, labels=labels_cal, Fs=Fs, class_ids=class_ids, 
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
            ZT, ZV = data_cal[:,:,int(tmin*Fs):int(tmax*Fs)], data_val[:,:,int(tmin*Fs):int(tmax*Fs)]
            tt, tv = labels_cal, labels_val
            as_max, _ = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=Fs, filt=filtering) 
            as_max = round(as_max*100,2)
            
            ##%% #########################################################################
            ZT, ZV = data_cal[:,:,int(tmin*Fs):int(tmax*Fs)], data_val[:,:,int(tmin*Fs):int(tmax*Fs)]
            tt, tv = labels_cal, labels_val
            as_tune, _ = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=Fs, filt=filtering, csp_list=csp_list)     
            as_tune = round(as_tune*100,2)
            
            ##%% #########################################################################
            V, P = [], []
            for i in range(len(H)):
                setup = H.iloc[i]
                ZT, ZV = data_cal[:,:,int(setup['tmin']*Fs):int(setup['tmax']*Fs)], data_val[:,:,int(setup['tmin']*Fs):int(setup['tmax']*Fs)]
                tt, tv = labels_cal, labels_val
                acc, learner = sbcsp_approach(ZT, ZV, tt, tv, setup['nbands'], setup['fl'], setup['fh'], setup['ncsp'], setup['clf'], Fs=Fs, filt=filtering)
                tv = learner['t']
                V.append(learner['y'])
                P.append(learner['yp'])
            
            ### Voting
            V = np.asarray(V).T
            y = np.asarray([mode(V[i])[0][0] for i in range(len(V))], dtype=int)
            as_mode = round(np.mean(y == tv)*100,2)
            
            ### Averaging
            PM = np.mean(np.transpose(P, (1,2,0)), axis=2)
            p = np.asarray([ class_ids[0] if (PM[p][0]>=PM[p][1]) else class_ids[1] for p in range(len(PM))], dtype=int)
            as_pmean = round(np.mean(p == tv)*100,2)
            
            as_best = max(as_tune, as_max, as_mode, as_pmean) 
            
            ##%% ###########################################################################
            ZT, ZV = data_cal[:,:,int(0.5*Fs):int(2.5*Fs)], data_val[:,:,int(0.5*Fs):int(2.5*Fs)]
            tt, tv = labels_cal, labels_val
            sb_iir, _ = sbcsp_approach(ZT, ZV, tt, tv, 9, 4, 40, 8, {'model':'SVM','kernel':{'kf':'linear'},'C':-4}, Fs=Fs, filt='IIR')
            cla_iir, _ = classic_approach(ZT, ZV, tt, tv, 8, 30, 8, {'model':'LDA'}, Fs=Fs, filt='IIR')
            sb_iir, cla_iir = round(sb_iir*100,2), round(cla_iir*100,2)
            
            ##%% ############################################################################
            # ============================================================================= #   
            # ON-LINE SIMULATION - BUFFER                                                   #
            # ============================================================================= #
            ##%% ############################################################################                                                         
            
            #%%
            delta_t = 0.05               # deslocamento em segundos
            delta_s = int(delta_t * Fs) # deslocamento em amostras
            
            L = tmax - tmin     # largura da janela do sujeito
            delay_t = L
                   
            fps = 60            # px/s velocidade máxima do asteroid 
            
            v_relativa = fps / L # velocidade do asteroid em relação à largura da janela do sujeito
            
            screen_size_x = 800 # largura da tela do jogo (resolução)
            screen_size_y = 600 # altura da tela do jogo
            round_time = screen_size_y / v_relativa  # simula tempo (segundos) que o asteroid leva para percorrer a tela
            
            # n_epochs = int((round_time - delay_t)/delta_t)   # número de janelas avaliadas na partida (exemplos/samples para classificação por partida)
            n_epochs = int((round_time)/delta_t) # ponta do asteroide a vista, asteroide parado por L ()
            
            n_rounds = 10        # número de partidas jogadas
            safe_targets = [ np.random.choice(class_ids) for i in range(n_rounds) ] # representa os comandos desejados a cada partida (fuga do asteróidd, lado oposto da tela de onde ele surge)
            # safe_target = [1,1,1,1,1,2,2,2,2,2]; random.shuffle(safe_target)
            # print(safe_target, '\n')
            
            limiar_gatilho = 0.8

            ##%% ###########################################################################
            #### CONSTRUIDO BUFFER DESLIZANTE - LARGURA OTIMIZADA                       ####
            ##%% ###########################################################################
            smin, smax = int(tmin*Fs), int(tmax*Fs)
            trials_a = np.asarray(data_val)[np.where(labels_val == class_ids[0])][:,:,smin:smax]
            trials_b = np.asarray(data_val)[np.where(labels_val == class_ids[1])][:,:,smin:smax]
            q = smax - smin # tamanho do buffer
            limiar_gatilho = as_best/100
            n_wins = 0
            n_commands = 0
            as_command, as_on = [], []
            
            for k in range(n_rounds):
                
                if safe_targets[k] == class_ids[0]:
                    ### Opção 1: Seleciona épocas por indice de das trials
                    # ZV = trials_a[ np.random.randint(0, len(trials_a)-1, n_epochs) ]) # escolha epocas aleatorias da classe A 
                    
                    ### Opção 2: Seleciona épocas ao simular janela deslizante
                    sig = trials_b[np.random.randint(0, len(trials_a)-1)] # trials_a[0]
                    for i in range(1, len(trials_b)): sig = np.c_[ sig, trials_a[np.random.randint(0, len(trials_a)-1)] ] # trials_a[i]
                    slider = np.asarray([ sig[:, i-q:i] for i in range(q, sig.shape[-1], delta_s) ]) # extrai do sinal continuo, épocas com dimensão (q) a cada deslocamento (delta_s)
                    ZV = slider[:n_epochs] # ZV(acumulador) = buffers/épocas da classe A
                    
                    tv = class_ids[0] * np.ones(len(ZV))
                else:
                    ### Opção 1: Seleciona épocas por indice de das trials
                    # ZV = trials_b[ np.random.randint(0, len(trials_b)-1, n_epochs) ]) # escolha epocas aleatorias da classe B 
                    
                    ### Opção 2: Seleciona épocas ao simular janela deslizante
                    sig = trials_b[np.random.randint(0, len(trials_b)-1)] # trials_b[0]
                    for i in range(1, len(trials_b)): sig = np.c_[ sig, trials_b[np.random.randint(0, len(trials_b)-1)] ] # trials_b[i]
                    slider = np.asarray([ sig[:, i-q:i] for i in range(q, sig.shape[-1], delta_s) ]) # extrai do sinal continuo, épocas com dimensão (q) a cada deslocamento (delta_s)
                    ZV = slider[:n_epochs] # ZV(acumulador) = buffers/épocas da classe B
                    
                    tv = class_ids[1] * np.ones(len(ZV))
                    
                ### CALCULATE AS ON-LINE ACCURACY 
                ZT, tt = data_cal[:,:,smin:smax], labels_cal
                acc_on, learner = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=info['fs'], filt='DFT')
                as_on.append(acc_on)
            
                ### GERAÇÃO DE COMANDO (AS)
                y, p = learner['y'], learner['yp']
                tta = tmax-tmin # tempo de ação em segundos ### tempo de ação menor == mais possibilidades de comandos durante a partida, e vice-versa
                nta = int(tta/delta_t) # tempo de ação em número de épocas no buffer circular (amostras passadas)
                as_comlist = []
                i = nta
                z = []
            
                coord_ast = screen_size_x*0.25 if (safe_targets[k] == class_ids[1]) else screen_size_x*0.75 
                coord_nave = np.copy(coord_ast) 
                
                while (i < len(y)): 
                    A = y[i-nta:i] # buffer circular externo
                    U1 = list(A).count(class_ids[0]) # conta as classificações A no buffer
                    U2 = list(A).count(class_ids[1]) # conta as classificaçòes B no buffer
                    U1_prop = U1 / nta # proporção de classificações A no buffer
                    U2_prop = U2 / nta # proporção de classificações B no buffer
                    # z.append([U1_prop, U2_prop,i-nta,i])
                    if (U1_prop >= limiar_gatilho): 
                        if coord_nave > 0: coord_nave -= 50; # print(coord_nave)
                        as_comlist.append(np.array([class_ids[0], i, coord_nave])) # [ comando, momento do envio (indice época)]
                        i += int(1/delta_t) # nta  # como o comando foi enviado o buffer SALTA, iniciando a partir da época do comando na próxima iteração
                    if (U2_prop >= limiar_gatilho):
                        if coord_nave < screen_size_x: coord_nave += 50; # print(coord_nave)    
                        as_comlist.append(np.array([class_ids[1], i, coord_nave]))
                        i += int(1/delta_t) # nta  # se um comando foi enviado, o próximo A se inicia no instante que o comando foi enviado (sem sobreposição) 
                    if (U1_prop < limiar_gatilho) and (U2_prop < 0.9): 
                        i += 1 # se nenhum comando foi enviado, o próximo A se inicia deslocado de delta_t em relação ao inicio do A anterior (sobrepopsição de nta-1)
                
                distancia_final = abs(coord_ast - coord_nave) # distancia da nave até o centro do asteróide...
                
                if distancia_final > 200: n_wins += 1; print(f'Congratulations, you WIN!! Wins={n_wins}!'); # ... deve ser maior que 200 para considerar-se seguro.
                else: print(f'Ops, you LOSE. Try again! Wins={n_wins}!') 
                
                if not as_comlist == []:
                    as_comlist = np.c_[np.asarray(as_comlist), np.zeros(len(as_comlist))].astype(int)
                    print(as_comlist[:,1])
                    # for i in range(len(as_comlist)): 
                    #     if as_comlist[i,0] == tv[as_comlist[i,1]]: as_comlist[i,2] = 1
                    as_comlist[:,3] = np.where(as_comlist[:,0] == tv[as_comlist[:,1]], 1, as_comlist[:,3])
                    acc_command = np.mean(as_comlist[:,0] == tv[as_comlist[:,1]]) # list(as_comlist[:, 2]).count(1)/len(as_comlist), tx acerto comando == com enviado/com esperado, corretos/len(as_comlist)
                    n_commands += len(as_comlist)
                else: acc_command = 0.0; as_comlist = None; n_commands = 0
                as_command.append(acc_command)
                
                max_comands_possiveis = 0
                for i in range(nta, len(y), int(1/delta_t)): max_comands_possiveis += 1
                # print(max_comands_possiveis)                

                # t_ = np.asarray([ tv[i] for i in range(nta, len(y), nta) ]) # tv[0::nta].astype(int)
                # m_ = np.asarray([ mode(y[i-nta:i])[0][0] for i in range(nta, len(y), nta) ])                 
                # p_ = np.asarray([ np.argmax(np.mean((p[i-nta:i]), axis=0)) + 1 for i in range(nta, len(p), nta) ])
                # print(f'Comando via Moda  : {round(np.mean(m_ == t_)*100,2)} | n_comandos: {len(m_)}')
                # print(f'Comando via PMedia: {round(np.mean(p_ == t_)*100,2)} | n_comandos: {len(p_)}\n')
            
                ## ITR uma partida 
                n_acertos = list(as_comlist[:,3]).count(1) #list(as_comlist[:,0] == tv[as_comlist[:,1]]).count(True)
                n = 2
                br = (acc_command * np.log2(acc_command) + (1-acc_command) * np.log2((1-acc_command)/(n-1)) + np.log2(n))
                d = len(as_comlist)/(round_time/60) # len(as_comlist)/(round_time/60)
                itr = d * br # bits/min
                print(itr, acc_command, d)
                
            as_on = round(np.mean(as_on)*100,2)
            as_command = round(np.mean(as_command)*100,2)
            
            ## ITR todas as partidas
            N = 2 # LR/RH
            P = n_wins/n_rounds # as_command # prob de acertar a decisão (vitoria ou comando individual)
            
            # taxa de informação útil (comando)
            bit_rate = P * np.log2(P) + (1-P) * np.log2((1-P)/(N-1)) + np.log2(N) # taxa por partida
            
            # incluir a velocidade da decisão na medida de desempenho
            n_decisoes = n_rounds # n_commands
            V = n_decisoes/(n_rounds * round_time/60) # decisoes por minuto
 
            ITR = V * bit_rate # bits por partida
            print(bit_rate, ITR)
        
            
            #%%
            ##%% ###########################################################################
            #### CONSTRUINDO BUFFER DESLIZANTE - LARGURA FIXA 2s                        ####
            ##%% ###########################################################################
            
            smin, smax = int(0.5*Fs), int(2.5*Fs)
            trials_a = np.asarray(data_val)[np.where(labels_val == class_ids[0])][:,:,smin:smax]
            trials_b = np.asarray(data_val)[np.where(labels_val == class_ids[1])][:,:,smin:smax]
            q = smax - smin # tamanho do buffer
            limiar_gatilho = as_best/100
            n_wins = 0
            n_commands = 0
            as_command, as_on = [], []
            
            for k in range(n_rounds):
                
                if safe_targets[k] == class_ids[0]:
                    
                    ### Opção 1: Seleciona épocas por indice de das trials
                    # ZV = trials_a[ np.random.randint(0, len(trials_a)-1, n_epochs) ]) # escolha epocas aleatorias da classe A 
                    
                    ### Opção 2: Seleciona épocas ao simular janela deslizante
                    sig = trials_b[np.random.randint(0, len(trials_a)-1)] # trials_a[0]
                    for i in range(1, len(trials_b)): sig = np.c_[ sig, trials_a[np.random.randint(0, len(trials_a)-1)] ] # trials_a[i]
                    slider = np.asarray([ sig[:, i-q:i] for i in range(q, sig.shape[-1], delta_s) ]) # extrai do sinal continuo, épocas com dimensão (q) a cada deslocamento (delta_s)
                    ZV = slider[:n_epochs] # ZV(acumulador) = buffers/épocas da classe A
                    
                    tv = class_ids[0] * np.ones(len(ZV))
                    

            #### Option 1: Sempre o segmento ideal do sujeito
            # ZV, tv = [], [] # acumulador de buffers/épocas e respectivos rótulos para validação
            # for i in range(n_rounds):
            #     if no_asteroidx[i] == class_ids[0]:ZV.append(trials_a[ np.random.randint(0, len(trials_a)-1, n_epochs) ]) # escolhe samples aleatorias da classe A
            #     else: ZV.append(trials_b[ np.random.randint(0, len(trials_b)-1, n_epochs) ]) # escolhe samples aleatorias da classe B
            #     tv.append(no_asteroidx[i] * np.ones(n_epochs))
            # if n_rounds > 1: ZV, tv = np.vstack(ZV), np.ravel(tv).astype(int) # formatando para validação dos modelos para todas as partidas de uma única vez
            # else: ZV, tv = ZV[0], tv[0].astype(int)
            
            
            
            # trials_a = np.asarray(data_val)[np.where(labels_val == class_ids[0])][:,:,smin:smax]
            # trials_b = np.asarray(data_val)[np.where(labels_val == class_ids[1])][:,:,smin:smax]
            
            # q = smax - smin
            
            # SA, SB = trials_a[0], trials_b[0]         
            # for i in range(1, len(trials_a)): SA = np.c_[ SA, trials_a[i] ]
            # for i in range(1, len(trials_b)): SB = np.c_[ SB, trials_b[i] ]
            
            # buffers_a = np.asarray([ SA[:, i-q:i] for i in range(q, SA.shape[-1], delta_s) ])
            # buffers_b = np.asarray([ SB[:, i-q:i] for i in range(q, SB.shape[-1], delta_s) ])

            # ZV, tv = [], [] 
            # cont_a, cont_b = delay, delay 
            # for i in range(n_rounds): 
            #     samples = int(round_time/delta_t)
                
            #     # if no_asteroidx[i] == class_ids[0]: 
            #     #     ZV.append(buffers_a[cont_a : cont_a + samples]) 
            #     #     cont_a += (samples + delay)
            #     # else:
            #     #     ZV.append(buffers_b[cont_b : cont_b + samples]) 
            #     #     cont_b += (samples + delay)
                
            #     if no_asteroidx[i] == class_ids[0]:ZV.append(trials_a[ np.random.randint(0, len(trials_a)-1, samples) ]) # escolhe samples aleatorias da classe A
            #     else: ZV.append(trials_b[ np.random.randint(0, len(trials_b)-1, samples) ]) # escolhe samples aleatorias da classe B
                
            #     tv.append(no_asteroidx[i] * np.ones(samples))
        
            # if n_rounds > 1: ZV, tv = np.vstack(ZV), np.ravel(tv).astype(int)
            # else: ZV, tv = ZV[0], tv[0].astype(int) 
        
            # ################################ SBCSP ON-LINE ################################
            # ZT, tt = data_cal[:,:,smin:smax], labels_cal
            # sb_on, learner = sbcsp_approach(ZT, ZV, tt, tv, 9, 4, 40, 8, {'model':'SVM','kernel':{'kf':'linear'},'C':-4}, Fs=info['fs'], filt='IIR')
            # sb_on = round(sb_on*100,2); # print(f"SBon : {sb_on}");
            
            # #### GERAÇÃO DE COMANDO (SBCSP)
            # ys, ps = learner['y'], learner['yp']
            # tta = 2 # 2.5 - 0.5
            # nta = int(tta/delta_t)
            # sb_comlist = []
            # i = nta
            # while (i < len(ys)):
            #     A = ys[i-nta:i] 
            #     U1 = list(A).count(class_ids[0]) 
            #     U2 = list(A).count(class_ids[1]) 
            #     U1_prop = U1 / nta
            #     U2_prop = U2 / nta 
            #     if (U1_prop >= limiar_gatilho): 
            #         sb_comlist.append(np.array([class_ids[0], i])) 
            #         i += nta 
            #     if (U2_prop >= limiar_gatilho): 
            #         sb_comlist.append(np.array([class_ids[1], i]))
            #         i += nta 
            #     if (U1_prop < limiar_gatilho) and (U2_prop < 0.9): 
            #         i += 1 
            # sb_comlist = np.asarray(sb_comlist)
            # sb_comlist = np.c_[sb_comlist, np.zeros(len(sb_comlist))].astype(int)
            # # for i in range(len(sb_comlist)): 
            # #     if sb_comlist[i,0] == tv[sb_comlist[i,1]]: sb_comlist[i,2] = 1 
            # sb_comlist[:,2] = np.where(sb_comlist[:,0] == tv[sb_comlist[:,1]], 1, sb_comlist[:,2])
            # sb_command = round(np.mean(sb_comlist[:,0] == tv[sb_comlist[:,1]])*100,2)

            # ########################### CLASSIC ON-LINE ################################
            # cla_on, learner = classic_approach(ZT, ZV, tt, tv, 8, 30, 8, {'model':'LDA'}, Fs=info['fs'], filt='IIR')
            # cla_on = round(cla_on*100,2) 
            
            # #### GERAÇÃO DE COMANDO (CLASSIC)
            # yc, pc = learner['y'], learner['yp'] 
            # cla_comlist = []
            # i = nta
            # while (i < len(yc)):
            #     A = yc[i-nta:i]
            #     U1 = list(A).count(class_ids[0])
            #     U2 = list(A).count(class_ids[1])
            #     U1_prop = U1 / nta
            #     U2_prop = U2 / nta
            #     if (U1_prop >= limiar_gatilho):
            #         cla_comlist.append(np.array([class_ids[0], i])); i += nta 
            #     if (U2_prop >= limiar_gatilho):
            #         cla_comlist.append(np.array([class_ids[1], i])); i += nta 
            #     if (U1_prop < limiar_gatilho) and (U2_prop < 0.9): i += 1 
            # cla_comlist = np.asarray(cla_comlist)
            # cla_comlist = np.c_[cla_comlist, np.zeros(len(cla_comlist))].astype(int)
            # # for i in range(len(cla_comlist)): 
            # #     if cla_comlist[i,0] == tv[cla_comlist[i,1]]: cla_comlist[i,2] = 1 
            # cla_comlist[:,2] = np.where(cla_comlist[:,0] == tv[cla_comlist[:,1]], 1, cla_comlist[:,2])
            # cla_command = round(np.mean(cla_comlist[:,0] == tv[cla_comlist[:,1]])*100,2)
            
            # # z = np.asarray([tv, y, ys, yc]).T
            
            # R.loc[0] = [suj, class_ids[0], class_ids[1], fl, fh, tmin, tmax, nbands, ncsp, csp_list, clf['model'], clf,
            #                           as_max, as_tune, as_mode, as_pmean, as_best, sb_iir, cla_iir, as_on, sb_on, cla_on, 
            #                           as_command, sb_command, cla_command, as_comlist, sb_comlist, cla_comlist]
            # # print(R.iloc[:,12:25].mean())
            # pd.to_pickle(R, '/home/vboas/cloud/results/' + ds + '/R_' + str(suj) + '.pkl')
            
            # del (globals()['data_cal'], globals()['data_val'], globals()['labels_cal'], globals()['labels_val'])
