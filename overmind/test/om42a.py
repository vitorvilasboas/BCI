# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
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
from bci_utils import nanCleaner, corrigeNaN, extractEpochs, Filter, CSP
from functools import partial
import random
from sklearn.preprocessing import normalize


np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

def classic_approach(ZT, ZV, tt, tv, fl, fh, ncsp, clf, Fs='250', filtering={'design':'DFT'}, csp_reg=None):
    filt = Filter(fl, fh, Fs, filtering)
    XT = filt.apply_filter(ZT)
    XV = filt.apply_filter(ZV)
    if filtering['design'] == 'DFT':
        bsize = 2 / ( Fs / ZT.shape[-1] ) # 2 representa sen e cos
        XT = XT[:, :, round(fl*bsize):round(fh*bsize)]
        bsize = 2 / ( Fs / ZV.shape[-1] )
        XV = XV[:, :, round(fl*bsize):round(fh*bsize)]
 
    csp = mne.decoding.CSP(n_components=int(ncsp), reg=csp_reg) # CSP MNE
    # csp = CSP(n_components=int(ncsp)) # My CSP
    csp.fit(XT, tt)
    csp_filters = csp.filters_[:int(ncsp)] # csp_filters = csp.filters_
    XT_CSP = csp.transform(XT)
    XV_CSP = csp.transform(XV)

    if clf['model'] == 'LDA': clf_final = LDA()
    elif clf['model'] == 'LR': clf_final = LogisticRegression()
    elif clf['model'] == 'SVM': clf_final = SVC(kernel=clf['kernel']['kf'], C=10**(clf['C']), gamma='scale', degree=3, probability=True)
    elif clf['model'] == 'KNN': clf_final = KNeighborsClassifier(n_neighbors=int(clf['neig']), metric=clf['metric'], p=3)

    clf_final.fit(XT_CSP, tt)
    y = clf_final.predict(XV_CSP)
    yp = clf_final.predict_proba(XV_CSP)
    acc, kappa = np.mean(y == tv), cohen_kappa_score(y, tv)
    learner = {'csp_filt':csp_filters, 'lda':None, 'p0':None, 'p1':None, 'clf_final':clf_final, 'y':y, 'yp':yp, 't':tv }
    return acc, learner


def sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs='250', filtering={'design':'DFT'}, csp_list=None, csp_reg=None):
    step = (fh-fl) / (nbands+1) # n_bins/nbands+1
    size = step / 0.5 # step/overlap
    sub_bands = []
    for i in range(nbands):
        fl_sb = i * step + fl
        fh_sb = i * step + size + fl
        sub_bands.append([fl_sb, fh_sb])
    # nbands = len(sub_bands)
    
    XT, XV = [], []
    if filtering['design'] == 'DFT':
        filt = Filter(fl, fh, Fs, filtering)
        XTF = filt.apply_filter(ZT)
        XVF = filt.apply_filter(ZV)
        for i in range(nbands):
            bsize = 2 / ( Fs / ZT.shape[-1] ) # 2 representa sen e cos
            XT.append(XTF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])
            bsize = 2 / ( Fs / ZV.shape[-1] )
            XV.append(XVF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])         
    elif filtering['design'] == 'IIR':
        for i in range(nbands):
            filt = Filter(sub_bands[i][0], sub_bands[i][1], Fs, filtering)
            XT.append(filt.apply_filter(ZT))
            XV.append(filt.apply_filter(ZV))

    if csp_list is None: chain = [ Pipeline([('CSP', mne.decoding.CSP(n_components=int(ncsp), reg=csp_reg)), ('LDA', LDA())]) for i in range(nbands) ] # CSP MNE
    else: chain = [ Pipeline([('CSP', mne.decoding.CSP(n_components=csp_list[i], reg=csp_reg)), ('LDA', LDA())]) for i in range(nbands) ] # CSP MNE - to tuning ncsp
    # if csp_list is None: chain = [ Pipeline([('CSP', CSP(n_components=int(ncsp))), ('LDA', LDA())]) for i in range(nbands) ] # My CSP
    # else: chain = [ Pipeline([('CSP', CSP(n_components=csp_list[i])), ('LDA', LDA())]) for i in range(nbands) ] # My CSP - to tuning ncsp
        
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

    if csp_list is None: csp_filters = [ chain[i]['CSP'].filters_[:int(ncsp)] for i in range(nbands) ] # chain[i]['CSP'].filters_
    else: csp_filters = [ chain[i]['CSP'].filters_[:csp_list[i]] for i in range(nbands) ]

    csps = [ chain[i]['CSP'] for i in range(nbands) ]
    ldas = [ chain[i]['LDA'] for i in range(nbands) ]

    XV_CSP = [ chain[i]['CSP'].transform(XV[i]) for i in range(nbands) ]
    SCORE_V = np.zeros((len(ZV), nbands))
    for i in range(nbands):
        SCORE_V[:, i] = np.ravel(chain[i]['LDA'].transform(XV_CSP[i]))
    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V))

    y = clf_final.predict(META_SCORE_V)
    yp = clf_final.predict_proba(META_SCORE_V)
    acc, kappa = np.mean(y == tv), cohen_kappa_score(y, tv)
    # learner = {'csp_filt':csp_filters,'lda':ldas, 'p0':p0, 'p1':p1, 'clf_final':clf_final, 'y':y, 'yp':yp, 't':tv}
    learner = {'csp':csps, 'csp_filt':csp_filters, 'lda':ldas, 'p0':p0, 'p1':p1, 'clf_final':clf_final, 'y':y, 'yp':yp, 't':tv}
    return acc, learner


def auto_csp_tunning(args, data, events, Fs, class_ids, setup, filtering={'design':'DFT'}):
    # csp_reg = setup['reg']
    epochs, labels = extractEpochs(data, events, int(setup['tmin']*Fs), int(setup['tmax']*Fs), class_ids)
    epochs = nanCleaner(epochs)

    # kf = StratifiedShuffleSplit(1, test_size=0.5, random_state=42)
    # for _train, _test in kf.split(epochs, labels):
    #     ZT, ZV, tt, tv = epochs[_train], epochs[_test], labels[_train], labels[_test]

    cross_acc, cross_learner = [], []
    kf = StratifiedShuffleSplit(5, test_size=0.2, random_state=42)
    for _train, _test in kf.split(epochs, labels):
        ZT, ZV, tt, tv = epochs[_train], epochs[_test], labels[_train], labels[_test]
        a, l = sbcsp_approach(ZT, ZV, tt, tv, setup['nbands'], setup['fl'], setup['fh'], setup['ncsp'], setup['clf'],
                              Fs=Fs, filtering=filtering, csp_list=list(map(lambda x: int(x), args)))
        cross_acc.append(a); cross_learner.append(l)
    acc, _ = np.mean(cross_acc), cross_learner[np.argmax(cross_acc)]
    return acc * (-1)


def auto_setup(args, data, events, Fs, class_ids, filtering={'design':'DFT'}):
    # f_low, f_high, tmin, tmax, ncomp, nbands, reg = 0, 40, 0.5, 2.5, 8, 9, 1e-1
    # clf = {'model':'SVM','kernel':{'kf':'linear'},'C':-4}
    f_low, f_high, tmin, tmax, ncsp, nbands, clf = args # reg, tmin['tmin']
    f_low, f_high = int(f_low['fl']), int(f_high['fh']) # csp_reg = 10**(int(reg))
    
    if nbands > (f_high - f_low): nbands = (f_high - f_low)
    while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg

    # epochs, labels = data[:, :, int(tmin*Fs):int(tmax*Fs)], events
    epochs, labels = extractEpochs(data, events, int(tmin*Fs), int(tmax*Fs), class_ids)
    epochs = nanCleaner(epochs)

    cross_acc, cross_learner = [], []
    kf = StratifiedShuffleSplit(5, test_size=0.2, random_state=42)
    for _train, _test in kf.split(epochs, labels):
        ZT, ZV, tt, tv = epochs[_train], epochs[_test], labels[_train], labels[_test]
        a, l = sbcsp_approach(ZT, ZV, tt, tv, int(nbands), f_low, f_high, ncsp, clf, Fs=Fs, filtering=filtering)
        cross_acc.append(a); cross_learner.append(l)
    acc, learner = np.mean(cross_acc), cross_learner[np.argmax(cross_acc)]

    H.loc[len(H)] = [f_low, f_high, tmin, tmax, ncsp, int(nbands), clf, acc, learner]
    return acc * (-1)


def test_sbcsp(ZV, tv, Fs, setup, learner, filtering={'design':'DFT'}):
    fl, fh, nbands = setup['fl'], setup['fh'], setup['nbands']
    csp = learner['csp']
    p0, p1, filters_, lda, clf_final = learner['p0'], learner['p1'], learner['csp_filt'], learner['lda'], learner['clf_final']   
    
    step = (fh-fl) / (nbands+1) # n_bins/nbands+1
    size = step / 0.5 # step/overlap
    sub_bands = []
    for i in range(nbands):
        fl_sb = i * step + fl
        fh_sb = i * step + size + fl
        sub_bands.append([fl_sb, fh_sb])
        
    if filtering['design'] == 'DFT':
        filt = Filter(fl, fh, Fs, filtering)
        XVF = filt.apply_filter(ZV)
        bsize = 2 / ( Fs / ZV.shape[-1] )
        XV = [ XVF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)] for i in range(nbands) ]       
    elif filtering['design'] == 'IIR':
        XV = []
        for i in range(nbands):
            filt = Filter(sub_bands[i][0], sub_bands[i][1], Fs, filtering)
            XV.append(filt.apply_filter(ZV))
    
    # YV = [ np.asarray([ np.dot(filters_[i], ep) for ep in XV[i] ]) for i in range(nbands) ] # Spatial Filtering
    # XV_CSP = [ np.log(np.mean(YV[i]**2, axis=2)) for i in range(nbands) ] # Feature extraction
    # XV_CSP = [ np.log(np.var(YV[i], axis=2)) for i in range(nbands) ] # Feature extraction

    XV_CSP = [ csp[i].transform(XV[i]) for i in range(nbands) ]
    SCORE_V = np.asarray([ np.ravel(lda[i].transform(XV_CSP[i])) for i in range(nbands) ]).T # Score LDA
    META_SCORE_V = np.log(p0.pdf(SCORE_V) / p1.pdf(SCORE_V)) # Bayesian Meta-score
    y = clf_final.predict(META_SCORE_V) # Final classification
    yp = clf_final.predict_proba(META_SCORE_V)
    acc = clf_final.score(META_SCORE_V, tv)
    return acc, y, yp


def get_features(Z, Fs, setup, learner, filtering={'design':'DFT'}, is_epoch=False):
    step = (setup['fh']-setup['fl']) / (setup['nbands']+1) # n_bins/setup['nbands']+1
    size = step/0.5 # step/overlap
    sub_bands = []
    for i in range(setup['nbands']):
        fl_sb = i * step + setup['fl']
        fh_sb = i * step + size + setup['fl']
        sub_bands.append([fl_sb, fh_sb])

    if filtering['design'] == 'DFT':
        filt = Filter(setup['fl'], setup['fh'], Fs, filtering)
        bsize = 2 / ( Fs / Z.shape[-1] )
        if is_epoch: 
            XF = filt.apply_filter(Z, is_epoch=True)
            X = [ XF[:, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)] for i in range(setup['nbands']) ]
        else: 
            XF = filt.apply_filter(Z)
            X = [ XF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)] for i in range(setup['nbands']) ]             
    elif filtering['design'] == 'IIR':
        X = []
        for i in range(setup['nbands']):
            filt = Filter(sub_bands[i][0], sub_bands[i][1], Fs, filtering)
            X.append(filt.apply_filter(Z))
    
    if is_epoch:
        # F = np.asarray([ np.log(np.mean(np.dot(learner['csp_filt'][i], X[i])**2, axis=1)) for i in range(setup['nbands']) ]) # Spatial Filtering, Feature extraction and Scoring LDA
        F = np.asarray([ np.log(np.var(np.dot(learner['csp_filt'][i], X[i]), axis=1)) for i in range(setup['nbands']) ]) 
        SCORE = np.ravel([ learner['lda'][i].transform(F[i].reshape(1,-1)) for i in range(setup['nbands']) ])
    else:
        F = [ learner['csp'][i].transform(X[i]) for i in range(setup['nbands']) ] # nbands x len(X) setup['nbands'] ncsp
        SCORE = np.asarray([ np.ravel(learner['lda'][i].transform(F[i])) for i in range(setup['nbands']) ]).T # Score LDA
        F = np.transpose(F, (1, 0, 2))   
    return F, SCORE


H = pd.DataFrame(columns=['fl','fh','tmin','tmax','ncsp','nbands','clf','acc','learner'])


if __name__ == "__main__":
    n_iter = 100
    class_ids = [1,2]
    # R = pd.DataFrame(columns=['subj', 'A', 'B', 'fl', 'fh', 'tmin', 'tmax', 'nbands', 'ncsp', 'csp_list', 'clf', 'clf_details',
    #                           'as_max', 'as_tune', 'as_mode', 'as_pmean', 'as_best', 'sb_iir', 'cla_iir',
    #                           'as_on', 'sb_on', 'cla_on', 'as_command', 'sb_command', 'cla_command',
    #                           'as_comlist', 'sb_comlist', 'cla_comlist'])
    for suj in [1]:
        print(f'###### {suj} {class_ids} ######')
        # data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + '.npy', allow_pickle=True)  # sessões 1 e 2
        data1, events1, info  = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'T.npy', allow_pickle=True) # sessão 1
        data2, events2, _  = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'E.npy', allow_pickle=True) # sessão 2
        Fs = info['fs']
        
        ##%% ##########################################################################
        fl_ = hp.uniformint("fl", 0, 10) # 0, 15
        fh_ = hp.uniformint("fh", 10, 25)
        # fh_ = hp.uniformint("fh", fl_ + 8, 40)
        # tmin_ = hp.quniform('tmin', 0, 2, 0.5)
        space = (
            {"fl": fl_},
            {"fh": (fh_ + fl_)}, # fl_ + 20, # hp.uniform("fh", (10 + fl_), 40),
            # {"fh": fh_},
            hp.quniform('tmin', 0, 2, 0.5),
            hp.quniform('tmax', 2, 4, 0.5),
            # {"tmin": tmin_},
            # hp.quniform('tmax', tmin_+1, 4, 0.5),
            # hp.quniform('ncsp', 2, 10, 2),
            hp.choice('ncsp', [2,4,6,8,22]),
            hp.uniformint('nbands', 1, 25),
            # hp.quniform('nbands', 1, ((fh_+fl_)-fl_)//2, 1),
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
        trials = base.Trials()
        init_vals = [{'fl':4,'fh':40,'tmin':0.5,'tmax':2.5,'ncomp':8,'nbands':9,'model':'SVM','C':1e-4,'kf':'linear'}]
        try:
            fmin_objective = partial(auto_setup, data=data1, events=events1, Fs=Fs, class_ids=class_ids, filtering={'design':'DFT'})
            best = fmin(fmin_objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=0, points_to_evaluate=init_vals)
        except: print('Exception raised'); raise # tpe.suggest
        # acc_as_cal = trials.best_trial['result']['loss']*(-1)
        best = trials.argmin # trials.best_trial['misc']['vals']
        del (globals()['init_vals'], globals()['space'], globals()['trials'], globals()['info'])
        
        hmax = H[ H['acc'] == H['acc'].max()].iloc[0]
        # learner_as_cal = hmax['learner']
        
        ##%% #########################################################################
        # fl = int(best['fl'])
        # fh = int(best['fl']) + int(best['fh'])
        # tmin = best['tmin']
        # tmax = best['tmax']
        # ncsp = [2,4,6,8,22][best['ncsp']] # best['ncsp'][0]
        # nbands = int(best['nbands'])
        # reg = None # 10**int(best['reg'])
        
        # while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
        # if nbands > (fh-fl): nbands = (fh-fl)
    
        # if best['clf'] == 0: clf = {'model':'LDA'}
        # elif best['clf'] == 1:
        #     svm_kernel = 'linear' if best['kernel']==0 else 'poly' if best['kernel']==1 else 'sigmoid' if best['kernel']==2  else 'rbf'
        #     clf = {'model':'SVM','kernel':{'kf':svm_kernel},'C':best['C']}
        # elif best['clf'] == 2:
        #     knn_metric = 'euclidean' if best['metric']==0 else 'manhattan' if best['metric']==1 else 'minkowski' if best['metric']==2 else 'chebyshev'
        #     clf = {'model':'KNN','metric':knn_metric,'neig':int(best['neig']), }
        # elif best['clf'] == 3: clf = {'model':'LR'}
        
        # setup = {'fl':fl, 'fh':fh, 'tmin':tmin, 'tmax':tmax, 'ncsp':ncsp, 'nbands':nbands, 'reg':reg, 'clf':clf}
        
        desvio = 4 # desvio em torno do ncsp ótimo (deve ser par)
        min_ncsp = (hmax['ncsp'] - desvio) if (hmax['ncsp'] - desvio) > 2 else 2
        max_ncsp = (hmax['ncsp'] + desvio) if (hmax['ncsp'] + desvio) < 22 else 22
        space = tuple([ hp.quniform('csp'+str(i), min_ncsp, max_ncsp, 2) for i in range(hmax['nbands']) ])
        fmin_tunning = partial(auto_csp_tunning, data=data1, events=events1, Fs=Fs, class_ids=class_ids, setup=hmax, filtering={'design':'DFT'})
        trials = base.Trials()
        init_vals = {'csp0':hmax['ncsp']}
        for i in range(1, hmax['nbands']-1): init_vals.update({'csp'+str(i):int(hmax['ncsp'])})
        try: best = fmin(fmin_tunning, space=space, algo=tpe.suggest, max_evals=n_iter//4, trials=trials, points_to_evaluate=[init_vals])
        except: print('Exception raised'); raise
        hmax = hmax.append(pd.Series({'csp_list': [ int(best['csp'+str(i)]) for i in range(hmax['nbands']) ] }))
        # acc_as_tune_cal = trials.best_trial['result']['loss']*(-1)
        del (globals()['init_vals'], globals()['space'], globals()['trials'], globals()['best'], globals()['desvio'], globals()['min_ncsp'], globals()['max_ncsp'])
        # print(f'csp_list: {csp_list}')
        ##%% #########################################################################
        
        ZT, tt = extractEpochs(data1, events1, int(hmax['tmin']*Fs), int(hmax['tmax']*Fs), class_ids)
        ZV, tv = extractEpochs(data2, events2, int(hmax['tmin']*Fs), int(hmax['tmax']*Fs), class_ids)
        # ZV = np.r_[ ZV[np.where(tv == class_ids[0])], ZV[np.where(tv == class_ids[1])] ]              # agrupar epocas de mesmo label
        # tv = np.r_[ class_ids[0]*np.ones(len(ZV)//2), class_ids[1]*np.ones(len(ZV)//2)].astype(int)   # agrupar labels iguais
        
        # acc_as_cal, _, _ = test_sbcsp(ZV, tv, Fs, hmax, hmax['learner'], filtering={'design':'DFT'})    # valida clf treinado na calibração 
        acc_as_max, learner_as_max = sbcsp_approach(ZT, ZV, tt, tv, hmax['nbands'], hmax['fl'], hmax['fh'], hmax['ncsp'], hmax['clf'], Fs=Fs, filtering={'design':'DFT'})
        acc_as_tune, _ = sbcsp_approach(ZT, ZV, tt, tv, hmax['nbands'], hmax['fl'], hmax['fh'], hmax['ncsp'], hmax['clf'], Fs=Fs, filtering={'design':'DFT'}, csp_list=hmax['csp_list'])
        hmax['learner'] = learner_as_max
        
        V, P = [], []
        for i in range(len(H)):
            setup = H.iloc[i]
            ZT, tt = extractEpochs(data1, events1, int(setup['tmin']*Fs), int(setup['tmax']*Fs), class_ids)
            ZV, tv = extractEpochs(data2, events2, int(setup['tmin']*Fs), int(setup['tmax']*Fs), class_ids)
            _, leah = sbcsp_approach(ZT, ZV, tt, tv, setup['nbands'], setup['fl'], setup['fh'], setup['ncsp'], setup['clf'], Fs=Fs, filtering={'design':'DFT'})
            tv = leah['t']
            V.append(leah['y'])
            P.append(leah['yp'])
        H = H.iloc[0:0] # cleaning df
    
        ### Voting
        V = np.asarray(V).T
        yv = np.asarray([mode(V[i])[0][0] for i in range(len(V))], dtype=int)
        acc_as_mode = np.mean(yv == tv)
    
        ### Averaging
        PM = np.mean(np.transpose(P, (1,2,0)), axis=2)
        pv = np.asarray([ class_ids[0] if (PM[p][0]>=PM[p][1]) else class_ids[1] for p in range(len(PM))], dtype=int)
        acc_as_pmean = np.mean(pv == tv)
        del (globals()['V'], globals()['P'], globals()['PM'], globals()['yv'], globals()['pv'], globals()['leah'], globals()['learner_as_max'])
    
        acc_as_best = max(acc_as_tune, acc_as_max, acc_as_mode, acc_as_pmean)
        
        ZT, tt = extractEpochs(data1, events1, int(0.5*Fs), int(2.5*Fs), class_ids)
        ZV, tv = extractEpochs(data2, events2, int(0.5*Fs), int(2.5*Fs), class_ids)
        acc_sb_iir, learner_sb_iir = sbcsp_approach(ZT, ZV, tt, tv, 9, 4, 40, 8, {'model':'SVM','kernel':{'kf':'linear'},'C':-4}, Fs=Fs, filtering={'design':'IIR', 'iir_order':5})
        acc_cla_iir, _ = classic_approach(ZT, ZV, tt, tv, 8, 30, 8, {'model':'LDA'}, Fs=Fs, filtering={'design':'IIR', 'iir_order':5})
        h_sb = pd.Series({'fl':4,'fh':40,'tmin':0.5,'tmax':2.5,'ncsp':8,'nbands':9,'clf':{'model':'SVM','kernel':{'kf':'linear'},'C':-4},'acc':acc_sb_iir,'learner':learner_sb_iir})
        del (globals()['learner_sb_iir'])
        
        print(f"Max: {round(acc_as_max*100,2)} >> {hmax['fl']}-{hmax['fh']}Hz; {hmax['tmin']}-{hmax['tmax']}s; Ns={hmax['nbands']}; R={hmax['ncsp']}; {hmax['clf']}; {hmax['csp_list']}")
        print(f"AS: {round(acc_as_best*100,2)} | SB: {round(acc_sb_iir*100,2)} | CLA: {round(acc_cla_iir*100,2)}\n")
        
        ##%% #########################################################################
        
        smin, smax = int(-2*Fs), int(4*Fs) # inicio da trial (2s antes da dica) : fim da MI
        # trials, labels   = extractEpochs(data, events, smin, smax, class_ids)
        trialsT, labelsT = extractEpochs(data1, events1, smin, smax, class_ids)
        trialsV, labelsV = extractEpochs(data2, events2, smin, smax, class_ids)
        
        # trials2, labels2 = extractEpochs(data2, events2, smin, smax, class_ids)
        # kf = StratifiedShuffleSplit(1, test_size=0.5, random_state=42)
        # acc_cross = []
        # for train, test in kf.split(trials2, labels2l):
        #     trialsT, trialsV, labelsT, labelsV = trials2[train], trials2[test], labels2[train], labels2[test]
        
        # del (globals()['data'], globals()['events'])
        del (globals()['data1'], globals()['data2'], globals()['events1'], globals()['events2'], globals()['smin'], globals()['smax'])
        
        
        # =============================================================================
        # Validação Online Dupla (2 chains)        
        # =============================================================================
        
        # delta_t = 0.2; delta_s = int(delta_t * Fs)
        # for setup, filtering in zip([hmax,h_sb], [{'design':'DFT'},{'design':'IIR', 'iir_order':5}]): 
        #     q = int((setup['tmax']*Fs) - (setup['tmin']*Fs))  # q = largura da época
        #     ZT, tt, ZV, tv = [], [], [], []
        #     for k in range(len(trialsT)): # len(trials_cal)
        #         n = q   # n=localização em amostras (fim) época
        #         m = q/2 # localização em amostra do ponto médio da época
        #         inc = (0.0 * q)  # para ser considerado comando, no min 70% da amostra (m+10%) deve estar contida no periodo de MI
        #         while n <= trialsT.shape[-1]:
        #             ZT.append(trialsT[k, :, n-q:n])
        #             ZV.append(trialsV[k, :, n-q:n])
        #             # m = (n+(n-q))/2  # outra forma de obter a localização em amostra do ponto médio da época
        #             if (m <= (500+inc)) or (m >= (1500-inc)): tt.append(0); tv.append(0); 
        #             else: tt.append(labelsT[k]); tv.append(labelsV[k])
        #             # print(n-q, n, m, tv[-1])
        #             m += delta_s
        #             n += delta_s
        #     ZT, ZV = np.asarray(ZT), np.asarray(ZV)
        #     tt, tv = np.asarray(tt), np.asarray(tv)
            
        #     tt_, tv_ = np.where(tt[:] != 0, 1, 0), np.where(tv[:] != 0, 1, 0)
        #     acc1, learner1 = sbcsp_approach(ZT, ZV, tt_, tv_, setup['nbands'], setup['fl'], setup['fh'], setup['ncsp'], setup['clf'], Fs=Fs, filtering=filtering)
            
        #     Z_, t_ = ZV[np.where(learner1['y'] == 1)], tv[np.where(learner1['y'] == 1)]
            
        #     acc2, y2, _ = test_sbcsp(Z_, t_, Fs, setup, setup['learner'], filtering=filtering)
            
        #     yv = np.copy(learner1['y'])
        #     yv[np.where(yv == 1)] = y2
        #     print('Yes/No:', round(acc1*100,2), '| LH/RH:', round(acc2*100,2), ' | Final:', round(np.mean(yv == tv)*100,2))

        #     acc3, y3, _ = test_sbcsp(ZV, tv, Fs, setup, setup['learner'], filtering=filtering)
        #     print('setup online direct:', round(acc3*100,2))
            

        #     del (globals()['ZT'], globals()['ZV'], globals()['tt'], globals()['tv'], globals()['tt_'], globals()['tv_'], globals()['Z_'], globals()['t_']) 
        #     del (globals()['acc1'], globals()['acc2'], globals()['acc3'], globals()['y2'], globals()['yv'], globals()['y3'], globals()['learner1'], )
        #     del (globals()['n'], globals()['m'], globals()['inc'], globals()['k'], globals()['q'])
            
        
        # =============================================================================
        # RNA   
        # =============================================================================
        delta_t = 0.2; delta_s = int(delta_t * Fs)
        for setup, filtering in zip( [hmax, h_sb], [{'design':'DFT'}, {'design':'IIR', 'iir_order':5}] ):   
        # setup, filtering = hmax, {'design':'DFT'}
            
            q = int((setup['tmax']*Fs) - (setup['tmin']*Fs))  # q = largura da época
            ZT, tt, ZV, tv = [], [], [], []
            for k in range(len(trialsT)): # len(trials_cal)
                n = q   # n=localização em amostras (fim) época
                m = q/2 # localização em amostra do ponto médio da época
                inc = (0.0 * q)  # para ser considerado comando, no min 70% da amostra (m+10%) deve estar contida no periodo de MI
                while n <= trialsT.shape[-1]:
                    ZT.append(trialsT[k, :, n-q:n])
                    ZV.append(trialsV[k, :, n-q:n])
                    # m = (n+(n-q))/2  # outra forma de obter a localização em amostra do ponto médio da época
                    if (m <= (500+inc)) or (m >= (1500-inc)): tt.append(0); tv.append(0); 
                    else: tt.append(labelsT[k]); tv.append(labelsV[k])
                    # print(n-q, n, m, tv[-1])
                    m += delta_s
                    n += delta_s
            ZT, ZV = np.asarray(ZT), np.asarray(ZV)
            tt, tv = np.asarray(tt), np.asarray(tv)
            
            TT, TV = np.zeros((len(ZT), 3)), np.zeros((len(ZV), 3))
            for i in range(3): TT[:,i] = np.where(tt == i, 1, TT[:,i])
            for i in range(3): TV[:,i] = np.where(tv == i, 1, TV[:,i])
                
            del (globals()['n'], globals()['m'], globals()['inc'], globals()['k'], globals()['q'])
                
            # ep0, _ = extractEpochs(data1, events1, -q, 0, class_ids)
            # ep, lb = extractEpochs(data1, events1, 0, q, class_ids)
            # ZT, tt = np.r_[ep0, ep], np.r_[np.zeros(len(ep0)), lb]
            
            # ep0, _ = extractEpochs(data2, events2, -q, 0, class_ids)
            # ep, lb = extractEpochs(data2, events2, 0, q, class_ids)
            # ZV, tv = np.r_[ep0, ep], np.r_[np.zeros(len(ep0)), lb]
                
            step = (setup['fh']-setup['fl']) / (setup['nbands']+1) # n_bins/setup['nbands']+1
            size = step / 0.5 # step/overlap
            sub_bands = []
            for i in range(setup['nbands']):
                fl_sb = i * step + setup['fl']
                fh_sb = i * step + size + setup['fl']
                sub_bands.append([fl_sb, fh_sb])
                
            XT, XV = [], []
            if filtering['design'] == 'DFT':
                filt = Filter(setup['fl'], setup['fh'], Fs, filtering)
                XTF = filt.apply_filter(ZT)
                XVF = filt.apply_filter(ZV)
                for i in range(setup['nbands']):
                    bsize = 2 / ( Fs / ZT.shape[-1] ) # 2 representa sen e cos
                    XT.append(XTF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])
                    bsize = 2 / ( Fs / ZV.shape[-1] )
                    XV.append(XVF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])    
                del (globals()['XTF'], globals()['XVF'], globals()['bsize'])        
            elif filtering['design'] == 'IIR':
                for i in range(setup['nbands']):
                    filt = Filter(sub_bands[i][0], sub_bands[i][1], Fs, filtering)
                    XT.append(filt.apply_filter(ZT))
                    XV.append(filt.apply_filter(ZV))
                
            # cla = [ Pipeline([('CSP', mne.decoding.CSP(n_components=int(setup['ncsp']), reg=None)), ('LDA', LDA())]) for i in range(setup['nbands']) ] # CSP MNE
            cla = [ Pipeline([('CSP', CSP(n_components=int(setup['ncsp']))), ('LDA', LDA())]) for i in range(setup['nbands']) ] # My CSP
                
            # if acc_as_tune > acc_as_max:
            #     print('Yeah')
            #     cla = [ Pipeline([('CSP', mne.decoding.CSP(n_components=hmax['csp_list'][i], reg=None)), ('LDA', LDA())]) for i in range(setup['nbands']) ] # CSP MNE
            
            for i in range(setup['nbands']): cla[i]['CSP'].fit(XT[i], tt)
            FT = [ cla[i]['CSP'].transform(XT[i]) for i in range(setup['nbands']) ]
            FV = [ cla[i]['CSP'].transform(XV[i]) for i in range(setup['nbands']) ]
            
            # for i in range(setup['nbands']): cla[i]['LDA'].fit(FT[i], tt)
            # ST = np.asarray([ np.ravel(cla[i]['LDA'].predict(FT[i])) for i in range(setup['nbands']) ]).T # Score LDA
            # SV = np.asarray([ np.ravel(cla[i]['LDA'].predict(FV[i])) for i in range(setup['nbands']) ]).T # Score LDA
            # # PT = np.transpose([cla[i]['LDA'].predict_proba(FT[i]) for i in range(setup['nbands'])], (1,0,2)) # prob LDA
            # # PV = np.transpose([cla[i]['LDA'].predict_proba(FV[i]) for i in range(setup['nbands'])], (1,0,2)) # prob LDA
            
            del (globals()['filt'], globals()['fl_sb'], globals()['fh_sb'], globals()['size'], globals()['step'], globals()['sub_bands'])
            del (globals()['ZT'], globals()['ZV'], globals()['XT'], globals()['XV'], globals()['cla'])
            
            FT = np.vstack(np.transpose(FT, (0,2,1))).T    
            FV = np.vstack(np.transpose(FV, (0,2,1))).T 
            # FT = normalize(FT, norm='l2')
            # FV = normalize(FV, norm='l2') 
            
            mlp = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, activation='tanh', hidden_layer_sizes=(100,2), random_state=42)
            mlp.out_activation = 'softmax' # 'logistic', 'softmax'
            mlp.fit(FT, TT)
            # print('mlp2:', round((mlp.fit(FT, tt)).score(FV, tv)*100,2)
            YP = mlp.predict_proba(FV)
            Y = mlp.predict(FV); # print('mlp:', np.mean(Y == TV))
            y = np.argmax(YP, axis=1)
            
            lda = LDA(); lda.fit(FT, tt); 
            print('mlp:',round(mlp.score(FV, TV)*100,2), ' | lda:', round(lda.score(FV, tv)*100,2))
            

            # F, t = np.r_[FT, FV], np.r_[tt, tv]
            # kf = StratifiedShuffleSplit(10, test_size=0.3, random_state=42)
            # acc_cross = []
            # for train, test in kf.split(F, t):
            #     FT, FV, tt, tv = F[train], F[test], t[train], t[test]
            #     mlp = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, activation='tanh', hidden_layer_sizes=(10,2)) # random_state=42,
            #     mlp.out_activation = 'softmax' # 'logistic'
            #     mlp.fit(FT, tt)
            #     yv = mlp.predict(FV)
            #     acc_cross.append(np.mean(yv == tv))
            # acc_mlp = np.mean(acc_cross)
            # print('mlp cv:', acc_mlp)
        
            # del (globals()['FT'], globals()['FV']) 
            del (globals()['tt'], globals()['tv'], globals()['TT'], globals()['TV'], globals()['YP'], globals()['Y'], globals()['y']) # globals()['ST'], globals()['SV'], 
        
        del (globals()['trialsT'], globals()['trialsV'], globals()['labelsT'], globals()['labelsV'])
        del (globals()['delta_s'], globals()['delta_t'], globals()['setup'], globals()['filtering'],globals()['Fs'], globals()['i'], globals()['suj'])
        del (globals()['mlp'], globals()['lda'])

    del (globals()['H'], globals()['n_iter'], globals()['class_ids'])
        
        

        
            
        ### Segmentação (treino/teste) pós obtenção de todas as features (não ideal para online)
        # for setup, learner, filt in zip( [hmax, h_sb], [learner_as_max, learner_sb], [{'design':'DFT'}, {'design':'IIR', 'iir_order':5}] ):
        #     q = int((setup['tmax']*Fs) - (setup['tmin']*Fs))  # q = largura da época
        #     Z, t = [], []
        #     for k in range(len(trials)): # len(ensaios)
        #         n = q
        #         m = q/2
        #         inc = (0.0 * q)
        #         while n <= trials.shape[-1]:
        #             Z.append(trials[k, :, n-q:n])
        #             # m = (n+(n-q))/2
        #             if ( m <= 500+inc) or (m >= 1500-inc): t.append(0)
        #             else: t.append(labels[k])
        #             n += delta_s
        #             m += delta_s
        #     Z, t = np.asarray(Z), np.asarray(t)
        
        #     F, _ = get_features(Z, Fs, setup, learner, filtering=filt)
        #     F = np.vstack(np.transpose(F, (1,2,0))).T
        #     # F = normalize(F, norm='l2')
        #     # np.min(F); np.max(F)
        
        #     kf = StratifiedShuffleSplit(1, test_size=0.5, random_state=42)
        #     acc_cross = []
        #     for train, test in kf.split(F, t):
        #         FT, FV, tt, tv = F[train], F[test], t[train], t[test]
        #         mlp = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, activation='tanh', hidden_layer_sizes=(10,2)) # random_state=42,
        #         mlp.out_activation = 'softmax' # 'logistic'
        #         mlp.fit(FT, tt)
        #         yv = mlp.predict(FV)
        #         acc_cross.append(np.mean(yv == tv))
        #     acc_mlp = np.mean(acc_cross)
        #     print('mlp cv:', acc_mlp)
        
        
        # ep0 = np.asarray(Z)[np.where(t == 0)]
        # epa = np.asarray(Z)[np.where(t == class_ids[0])]
        # epb = np.asarray(Z)[np.where(t == class_ids[1])]   


        # =============================================================================
        #     
        # =============================================================================

        # buffer = collections.deque(maxlen=q)
                
        # ZVA = np.asarray(trials2)[np.where(labels2 == class_ids[0])] #[:,:,smin:smax]
        # ZVB = np.asarray(trials2)[np.where(labels2 == class_ids[1])] #[:,:,smin:smax]
    
        # SA, SB = ZVA[0], ZVB[0]
        # for i in range(1, len(ZVA)): SA = np.c_[ SA, ZVA[i] ]
        # for i in range(1, len(ZVB)): SB = np.c_[ SB, ZVB[i] ]
        
        # SA, SB = ZVA[np.random.randint(0, len(ZVA)-1)], ZVB[np.random.randint(0, len(ZVB)-1)] 
        # for i in range(1, len(ZVA)): SA = np.c_[ SA, trials_a[np.random.randint(0, len(ZVA)-1)] ] # ZVA[i]
        # for i in range(1, len(ZVB)): SB = np.c_[ SB, trials_a[np.random.randint(0, len(ZVB)-1)] ] # ZVB[i]
        
        # comm_esperado = 1
        
        # round_time = 20
        # # n_epochs = int((round_time - delay_t)/delta_t)   # número de janelas avaliadas na partida (exemplos/samples para classificação por partida)
        # n_epochs = int((round_time)/delta_t) # ponta do asteroide a vista, asteroide parado por L ()
                
        # if comm_esperado == class_ids[0]: 
        #     slider = np.asarray([ SA[:, n-q:n] for n in range(q, SA.shape[-1], delta_s) ]) # extrai do sinal continuo, épocas com dimensão (q) a cada deslocamento (delta_s)
        #     ZV = slider[:n_epochs] # ZV(acumulador) = buffers/épocas da classe A
        #     tv = class_ids[0] * np.ones(len(ZV))
        
        # elif comm_esperado == class_ids[1]:
        #     slider = np.asarray([ SB[:, n-q:n] for n in range(q, SB.shape[-1], delta_s) ]) # extrai do sinal continuo, épocas com dimensão (q) a cada deslocamento (delta_s)
        
        #     ZV = slider[:n_epochs] # ZV(acumulador) = buffers/épocas da classe A
        #     tv = class_ids[0] * np.ones(len(ZV))
    
        # del (globals()['ZT'], globals()['ZV'], globals()['tt'], globals()['tv'])
        # del (globals()['data_cal'], globals()['data_val'], globals()['data'], globals()['events_cal'], globals()['events_val'], globals()['events'], globals()['info'])


        # ##%% ############################################################################
        # # ============================================================================= #
        # # ON-LINE SIMULATION - BUFFER                                                   #
        # # ============================================================================= #
        # ##%% ############################################################################

        # # raw = mne.io.read_raw_gdf('/mnt/dados/eeg_data/IV2a/A0' + str(suj) + 'E.gdf').load_data()
        # # data_val = raw.get_data()[:22] # [channels x samples]
        # # events_val = mne.events_from_annotations(raw)  # raw.find_edf_events()

        # cond = False
        # for cl in class_ids: cond += (events_val[:,1]==cl)
        # idx = np.ravel(np.where(cond)) # idx mantém os indices das amostras que marcam a apresentação das dicas são apresentadas nas trials ao longo da sessão
        # labels = events_val[idx,1]

        # #### EASY
        # # s0 = events_val[idx,0] # cue moment
        # # sn = s0 + int(4*Fs)    # start break moment

        # #### MODERATE
        # s0 = events_val[idx-1,0] # start trial moment
        # sn = s0 + int(7*Fs)      # min end break moment # int(min(np.diff(s0))) # menor tempo entre os inicios de dois trials = 7.58s ou 1895 amostras

        # #### HARD
        # # s0 = events_val[idx-1,0] # start trial moment
        # # sn = [ s0[i] for i in range(1, len(s0), 1) ] # end break moment
        # # sn.append(data_val.shape[-1] - (Fs*5)) # add last end break moment (5 seg antes do sinal acabar)
        # # sn = np.asarray(sn)

        # trials = [ data_val[:, s0[i]:sn[i]] for i in range(len(labels)) ]
        # trials_a = np.asarray(trials)[np.where(labels == class_ids[0])]
        # trials_b = np.asarray(trials)[np.where(labels == class_ids[1])]


        # SA, SB = trials_a[0], trials_b[0]
        # for i in range(1, len(trials_a)): SA = np.c_[ SA, trials_a[i] ]
        # for i in range(1, len(trials_b)): SB = np.c_[ SB, trials_b[i] ]

        # delta_t = 0.2 # deslocamento em segundos
        # delta_s = int(delta_t * Fs) # deslocamento em amostras
        # delay = int(5/delta_t)  # 2/delta_t == 2 seg == x épocas, representa épocas "perdidas" entre partidas

        # n_rounds = 6        # 6, número de partidas jogadas
        # round_time = 40     # 40, simula tempo (segundos) que o asteroid leva para percorrer a tela

        # # no_asteroidx = [ np.random.choice(class_ids) for i in range(n_rounds) ] # representa os comandos desejados a cada partida (fuga do asteróidd, lado oposto da tela de onde ele surge)
        # no_asteroidx = [1,1,1,2,2,2]; random.shuffle(no_asteroidx)
        # # print(no_asteroidx, '\n')

        # limiar_gatilho = 0.9

        # data_val, events_val = None, None

        # ##%% ###########################################################################
        # #### CONSTRUIDO BUFFER DESLIZANTE - LARGURA OTIMIZADA                       ####
        # ##%% ###########################################################################

        # q = int(tmax*Fs) - int(tmin*Fs) # 500 tamanho do buffer

        # # constrói épocas do sinal continuo usando as dimensões do buffer (q) e o deslocamento (simula o deslizar do buffer)
        # buffers_a = np.asarray([ SA[:, i-q:i] for i in range(q, SA.shape[-1], delta_s) ]) # n_
        # buffers_b = np.asarray([ SB[:, i-q:i] for i in range(q, SB.shape[-1], delta_s) ])

        # ZV, tv = [], [] # acumulador de buffers/épocas e respectivos rótulos para validação
        # cont_a, cont_b = delay, delay # contadores de buffers/épocas A, B adicionadas ao acumulador para validação
        # for i in range(n_rounds):
        #     samples = int(round_time/delta_t)   # 40/0.2 = 200, número de janelas avaliadas na partida
        #     if no_asteroidx[i] == class_ids[0]:
        #         ZV.append(buffers_a[cont_a : cont_a + samples]) # add buffers/épocas da classe A no acumulador
        #         cont_a += (samples + delay) # incrementa o contador de buffers/épocas A usadas
        #     else:
        #         ZV.append(buffers_b[cont_b : cont_b + samples]) # add buffers/épocas da classe A no acumulador
        #         cont_b += (samples + delay) # incrementa o contador de buffers/épocas A usadas
        #     tv.append(no_asteroidx[i] * np.ones(samples))

        # if n_rounds > 1: ZV, tv = np.vstack(ZV), np.ravel(tv).astype(int) # formatando para validação dos modelos para todas as partidas de uma única vez
        # else: ZV, tv = ZV[0], tv[0].astype(int)

        # ################################ AS ON-LINE ##################################

        # ZT, _, tt, _ = half_split_data(data, events, int(tmin*Fs), int(tmax*Fs), class_ids)
        # as_on, LA = sbcsp_approach(ZT, ZV, tt, tv, nbands, fl, fh, ncsp, clf, Fs=info['fs'], filt='DFT')
        # # as_online, y, p = tester(ZV, tv, Fs, class_ids, hmax) # (TREINO SOMENTE E)
        # as_on = round(as_on*100,2); # print(f"ASon : {as_on}")

        # #### GERAÇÃO DE COMANDO (AS)

        # y, p = LA['y'], LA['yp']
        # tta = tmax-tmin # tempo de ação em segundos ### tempo de ação menor == mais possibilidades de comandos durante a partida, e vice-versa
        # nta = int(tta/delta_t) # tempo de ação em número de épocas no buffer circular (amostras passadas)
        # cont_com_a, cont_com_b, cont_no_com = 0, 0, 0
        # as_comlist = []

        # i = nta
        # while (i < len(y)): # for i in range(nta, len(y), 1):
        #     A = y[i-nta:i] #buffer circular externo
        #     U1 = list(A).count(class_ids[0]) # conta as classificações A no buffer
        #     U2 = list(A).count(class_ids[1]) # conta as classificaçòes B no buffer
        #     U1_prop = U1 / nta # proporção de classificações A no buffer
        #     U2_prop = U2 / nta # proporção de classificações B no buffer
        #     if (U1_prop >= limiar_gatilho): #  and (tv[i]==class_ids[0])
        #         as_comlist.append(np.array([class_ids[0], i])) # [ comando, momento do envio (indice época)]
        #         # cont_com_a += 1; rota += '< '; # print([i-nta,i])
        #         i += nta # como o comando foi enviado o buffer SALTA, iniciando a partir da época do comando na próxima iteração
        #     if (U2_prop >= limiar_gatilho): #  and (tv[i]==class_ids[1])
        #         as_comlist.append(np.array([class_ids[1], i]))
        #         # cont_com_b += 1; rota += '> '; # print([i-nta,i])
        #         i += nta # se um comando foi enviado, o próximo A se inicia no instante que o comando foi enviado (sem sobreposição)
        #     if (U1_prop < limiar_gatilho) and (U2_prop < 0.9):
        #         # cont_no_com += 1;
        #         i += 1 # se nenhum comando foi enviado, o próximo A se inicia deslocado de delta_t em relação ao inicio do A anterior (sobrepopsição de nta-1)
        # as_comlist = np.asarray(as_comlist)
        # as_comlist = np.c_[as_comlist, np.zeros(len(as_comlist))].astype(int)

        # for i in range(len(as_comlist)):
        #     if as_comlist[i,0] == tv[as_comlist[i,1]]: as_comlist[i, 2] = 1

        # as_command = round(np.mean(as_comlist[:, 0] == tv[as_comlist[:,1]])*100,2)
        # # taxa de acerto de comando == comando enviado / comando esperado : corretos_cont/len(command_list) ou np.mean(command_list[:, 0] == tv[command_list[:,1]])
        # # print(no_asteroidx, cont_com_a, cont_com_b, cont_no_com)
        # # print(rota); print(corretos_list)
        # # print(f'Comando via Acum. : {as_command} | n_comandos: {cont_com_a + cont_com_b} | n_corretos: {corretos_cont}')

        # # t_ = np.asarray([ tv[i] for i in range(nta, len(y), nta) ]) # tv[0::nta].astype(int)
        # # m_ = np.asarray([ mode(y[i-nta:i])[0][0] for i in range(nta, len(y), nta) ])
        # # p_ = np.asarray([ np.argmax(np.mean((p[i-nta:i]), axis=0)) + 1 for i in range(nta, len(p), nta) ])
        # # print(f'Comando via Moda  : {round(np.mean(m_ == t_)*100,2)} | n_comandos: {len(m_)}')
        # # print(f'Comando via PMedia: {round(np.mean(p_ == t_)*100,2)} | n_comandos: {len(p_)}\n')


        # ##%% ###########################################################################
        # #### CONSTRUIDO BUFFER DESLIZANTE - LARGURA FIXA 2s                         ####
        # ##%% ###########################################################################

        # smin, smax = int(0.5*Fs), int(2.5*Fs)
        # q = smax - smin # tamanho do buffer

        # # constrói épocas do sinal continuo usando as dimensões do buffer (q) e o deslocamento (simula o deslizar do buffer)
        # buffers_a = np.asarray([ SA[:, i-q:i] for i in range(q, SA.shape[-1], delta_s) ]) # n_
        # buffers_b = np.asarray([ SB[:, i-q:i] for i in range(q, SB.shape[-1], delta_s) ])

        # ZV, tv = [], [] # acumulador de buffers/épocas e respectivos rótulos para validação
        # cont_a, cont_b = delay, delay # contadores de buffers/épocas A, B adicionadas ao acumulador para validação
        # for i in range(n_rounds):
        #     samples = int(round_time/delta_t)   # 200, número de janelas avaliadas na partida
        #     if no_asteroidx[i] == class_ids[0]:
        #         ZV.append(buffers_a[cont_a : cont_a + samples]) # add buffers/épocas da classe A no acumulador
        #         cont_a += (samples + delay) # incrementa o contador de buffers/épocas A usadas
        #     else:
        #         ZV.append(buffers_b[cont_b : cont_b + samples]) # add buffers/épocas da classe A no acumulador
        #         cont_b += (samples + delay) # incrementa o contador de buffers/épocas A usadas
        #     tv.append(no_asteroidx[i] * np.ones(samples))

        # if n_rounds > 1: ZV, tv = np.vstack(ZV), np.ravel(tv).astype(int) # formatando para validação dos modelos para todas as partidas de uma única vez
        # else: ZV, tv = ZV[0], tv[0].astype(int)

        # ################################ SBCSP ON-LINE ################################

        # ZT, _, tt, _ = half_split_data(data, events, smin, smax, class_ids)
        # sb_on, LS = sbcsp_approach(ZT, ZV, tt, tv, 9, 4, 40, 8, {'model':'SVM','kernel':{'kf':'linear'},'C':-4}, Fs=info['fs'], filt='IIR')
        # sb_on = round(sb_on*100,2); # print(f"SBon : {sb_on}");

        # #### GERAÇÃO DE COMANDO (SBCSP)

        # ys, ps = LS['y'], LS['yp']
        # tta = tmax-tmin # tempo de ação em segundos ### tempo de ação menor == mais possibilidades de comandos durante a partida, e vice-versa
        # nta = int(tta/delta_t) # tempo de ação em número de épocas no buffer circular (amostras passadas)
        # cont_com_a, cont_com_b, cont_no_com = 0, 0, 0
        # sb_comlist, rota = [], ''

        # i = nta
        # while (i < len(ys)): # for i in range(nta, len(ys), 1):
        #     A = ys[i-nta:i] #buffer circular externo
        #     U1 = list(A).count(class_ids[0]) # conta as classificações A no buffer
        #     U2 = list(A).count(class_ids[1]) # conta as classificaçòes B no buffer
        #     U1_prop = U1 / nta # proporção de classificações A no buffer
        #     U2_prop = U2 / nta # proporção de classificações B no buffer
        #     if (U1_prop >= limiar_gatilho): #  and (tv[i]==class_ids[0])
        #         sb_comlist.append(np.array([class_ids[0], i])) # [ comando, momento do envio (indice época)]
        #         cont_com_a += 1; rota += '< '; # print([i-nta,i])
        #         i += nta # como o comando foi enviado o buffer SALTA, iniciando a partir da época do comando na próxima iteração
        #     if (U2_prop >= limiar_gatilho): #  and (tv[i]==class_ids[1])
        #         sb_comlist.append(np.array([class_ids[1], i]))
        #         cont_com_b += 1; rota += '> '; # print([i-nta,i])
        #         i += nta # se um comando foi enviado, o próximo A se inicia no instante que o comando foi enviado (sem sobreposição)
        #     if (U1_prop < limiar_gatilho) and (U2_prop < 0.9):
        #         cont_no_com += 1; i += 1 # se nenhum comando foi enviado, o próximo A se inicia deslocado de delta_t em relação ao inicio do A anterior (sobrepopsição de nta-1)
        # sb_comlist = np.asarray(sb_comlist)
        # sb_comlist = np.c_[sb_comlist, np.zeros(len(sb_comlist))].astype(int)

        # corretos_cont = 0; corretos_list = ''
        # for i in range(len(sb_comlist)):
        #     if sb_comlist[i,0] == tv[sb_comlist[i,1]]: sb_comlist[i,2] = 1 # corretos_cont += 1; corretos_list += '1 '
        #     # else: corretos_list += '0 '

        # sb_command = round(np.mean(sb_comlist[:, 0] == tv[sb_comlist[:,1]])*100,2)
        # # taxa de acerto de comando == comando enviado / comando esperado : corretos_cont/len(command_list) ou np.mean(command_list[:, 0] == tv[command_list[:,1]])
        # # print(no_asteroidx, cont_com_a, cont_com_b, cont_no_com)
        # # print(rota); print(corretos_list)
        # # print(f'Comando via Acum. : {round(np.mean(command_list[:, 0] == tv[command_list[:,1]])*100,2)} | n_comandos: {cont_com_a + cont_com_b} | n_corretos: {corretos_cont}')

        # # t_ = np.asarray([ tv[i] for i in range(nta, len(ys), nta) ]) # tv[0::nta].astype(int)
        # # m_ = np.asarray([ mode(ys[i-nta:i])[0][0] for i in range(nta, len(ys), nta) ])
        # # p_ = np.asarray([ np.argmax(np.mean((ps[i-nta:i]), axis=0)) + 1 for i in range(nta, len(ps), nta) ])
        # # print(f'Comando via Moda  : {round(np.mean(m_ == t_)*100,2)} | n_comandos: {len(m_)}')
        # # print(f'Comando via PMedia: {round(np.mean(p_ == t_)*100,2)} | n_comandos: {len(p_)}\n')


        # # ########################### CLASSIC ON-LINE ################################

        # ZT, _, tt, _ = half_split_data(data, events, smin, smax, class_ids)
        # cla_on, LC = classic_approach(ZT, ZV, tt, tv, 8, 30, 8, {'model':'LDA'}, Fs=info['fs'], filt='IIR')
        # cla_on = round(cla_on*100,2) # print(f"CLAon: {cla_on}")

        # #### GERAÇÃO DE COMANDO (CLASSIC)

        # yc, pc = LC['y'], LC['yp']
        # tta = 2 # tempo de ação em segundos ### tempo de ação menor == mais possibilidades de comandos durante a partida, e vice-versa
        # nta = int(tta/delta_t) # tempo de ação em número de épocas no buffer circular (amostras passadas)
        # cont_com_a, cont_com_b, cont_no_com = 0, 0, 0
        # cla_comlist, rota_cla_iir = [], ''

        # i = nta
        # while (i < len(yc)): # for i in range(nta, len(yc), 1):
        #     A = yc[i-nta:i] #buffer circular externo
        #     U1 = list(A).count(class_ids[0]) # conta as classificações A no buffer
        #     U2 = list(A).count(class_ids[1]) # conta as classificaçòes B no buffer
        #     U1_prop = U1 / nta # proporção de classificações A no buffer
        #     U2_prop = U2 / nta # proporção de classificações B no buffer
        #     if (U1_prop >= limiar_gatilho): #  and (tv[i]==class_ids[0])
        #         cla_comlist.append(np.array([class_ids[0], i])) # [ comando, momento do envio (indice época)]
        #         cont_com_a += 1; rota_cla_iir += '< '; # print([i-nta,i])
        #         i += nta # como o comando foi enviado o buffer SALTA, iniciando a partir da época do comando na próxima iteração
        #     if (U2_prop >= limiar_gatilho): #  and (tv[i]==class_ids[1])
        #         cla_comlist.append(np.array([class_ids[1], i]))
        #         cont_com_b += 1; rota_cla_iir += '> '; # print([i-nta,i])
        #         i += nta # se um comando foi enviado, o próximo A se inicia no instante que o comando foi enviado (sem sobreposição)
        #     if (U1_prop < limiar_gatilho) and (U2_prop < 0.9):
        #         cont_no_com += 1; i += 1 # se nenhum comando foi enviado, o próximo A se inicia deslocado de delta_t em relação ao inicio do A anterior (sobrepopsição de nta-1)
        # cla_comlist = np.asarray(cla_comlist)
        # cla_comlist = np.c_[cla_comlist, np.zeros(len(cla_comlist))].astype(int)

        # corretos_cont_cla_iir = 0; corretos_list_cla_iir = ''
        # for i in range(len(cla_comlist)):
        #     if cla_comlist[i,0] == tv[cla_comlist[i,1]]: cla_comlist[i,2] = 1 # corretos_cont_cla_iir += 1; corretos_list_cla_iir += '1 '
        #     # else: corretos_list_cla_iir += '0 '

        # # taxa de acerto de comando == comando enviado / comando esperado : corretos_cont/len(command_list) ou np.mean(command_list[:, 0] == tv[command_list[:,1]])
        # # print(no_asteroidx, cont_com_a, cont_com_b, cont_no_com)
        # cla_command = round(np.mean(cla_comlist[:, 0] == tv[cla_comlist[:,1]])*100,2)
        # # print(rota_cla_iir); print(corretos_list_cla_iir)
        # # print(f'Comando via Acum. : {acc_cla_iir_acum} | n_comandos: {cont_com_a + cont_com_b} | n_corretos: {corretos_cont}')

        # # t_ = np.asarray([ tv[i] for i in range(nta, len(yc), nta) ]) # tv[0::nta].astype(int)
        # # m_ = np.asarray([ mode(yc[i-nta:i])[0][0] for i in range(nta, len(yc), nta) ])
        # # p_ = np.asarray([ np.argmax(np.mean((pc[i-nta:i]), axis=0)) + 1 for i in range(nta, len(pc), nta) ])
        # # print(f'Comando via Moda  : {round(np.mean(m_ == t_)*100,2)} | n_comandos: {len(m_)}')
        # # print(f'Comando via PMedia: {round(np.mean(p_ == t_)*100,2)} | n_comandos: {len(p_)}\n')


        # # # z = np.asarray([tv, y, ys, yc]).T

        # R.loc[0] = [suj, class_ids[0], class_ids[1], hmax['fl'], hmax['fh'], hmax['tmin'], hmax['tmax'], hmax['nbands'], hmax['ncsp'], csp_list, hmax['clf']['model'], hmax['clf'],
        #                          acc_as_max, acc_as_tune, acc_as_mode, acc_as_pmean, acc_as_best, acc_sb_iir, acc_cla_iir, as_on, sb_on, cla_on,
        #                          as_command, sb_command, cla_command, as_comlist, sb_comlist, cla_comlist]

        # # pd.to_pickle(R, '/home/vboas/cloud/results/' + ds + '/R_' + ds + '_' + str(suj) + '.pkl')







# epochs, labels = extractEpochs(data, events, 0, int(4*Fs), class_ids)
# kf = StratifiedShuffleSplit(10, test_size=0.5, random_state=42)
# for train, test in kf.split(epochs, labels):
#     trials_cal, labels_cal, trials_val, labels_val = epochs[train], labels[train], epochs[test], labels[test]




