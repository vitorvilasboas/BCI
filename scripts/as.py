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
from scipy.stats import norm
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

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

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
        [D, W] = eigh(S0, S0 + S1)
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
        ind[1::2] = np.arange(0, c // 2)
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
        
        if self.crossval:
            self.cross_scores, self.cross_kappa = [], []
            kf = StratifiedShuffleSplit(self.nfolds, test_size=self.test_perc, random_state=42)
            # kf = StratifiedKFold(self.nfolds, False)
            # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
            # self.cross_scores = cross_val_score(self.chain, XF, self.labels, cv=kf)
            for idx_treino, idx_teste in kf.split(self.epochs, self.labels):
                XT, XV, yT, yV = self.epochs[idx_treino], self.epochs[idx_teste], self.labels[idx_treino], self.labels[idx_teste]
                # print(np.asarray(XT).shape, np.asarray(XV).shape)
                acc_fold, kappa_fold = self.classic_approach(XT, XV, yT, yV) if (self.ap['option'] == 'classic') else self.sbcsp_approach(XT, XV, yT, yV)     
                self.cross_scores.append(acc_fold) # self.cross_scores.append(self.chain.score(XV, yV))
                self.cross_kappa.append(kappa_fold)
            self.acc, self.kappa = np.mean(self.cross_scores), np.mean(self.cross_kappa)
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
                  
            # print(np.asarray(XT).shape, np.asarray(XV).shape)
            yT = np.concatenate([self.class_ids[0] * np.ones(int(len(XT)/2)), self.class_ids[1] * np.ones(int(len(XT)/2))])
            yV = np.concatenate([self.class_ids[0] * np.ones(int(len(XV)/2)), self.class_ids[1] * np.ones(int(len(XV)/2))])
            self.acc, self.kappa = self.classic_approach(XT, XV, yT, yV) if (self.ap['option'] == 'classic') else self.sbcsp_approach(XT, XV, yT, yV)
    
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
        # self.csp_filters = self.csp.filters_
        XT_CSP = self.csp.transform(XTF)
        XV_CSP = self.csp.transform(XVF) 
        self.clf_final.fit(XT_CSP, yT)
        self.scores = self.clf_final.predict(XV_CSP)
        
        # # Option 2:
        # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
        # self.chain.fit(XT, yT)
        # self.csp_filters = self.chain['CSP'].filters_
        # self.scores = self.chain.predict(XV)
        
        acc = np.mean(self.scores == yV)     
        kappa = cohen_kappa_score(self.scores, yV)
        return acc, kappa
        
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
        self.chain = [ Pipeline([('CSP', CSP(n_components=int(self.ncomp))), ('LDA', LDA())]) for i in range(nbands) ]
        # self.chain = [ Pipeline([('CSP', CSP(n_components=self.csp_list[i])), ('LDA', LDA())]) for i in range(nbands) ] # uncomment to tuning ncsp
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
        acc = np.mean(self.scores == yV)
        kappa = cohen_kappa_score(self.scores, yV)
        return acc, kappa

##%% #############################################################################
bci = BCI()
def objective(args):
    # print(args)
    f_low, f_high, bci.tmin, bci.tmax, ncomp, nbands, bci.clf = args
    bci.ap = {'option': 'sbcsp', 'nbands': nbands}
    bci.f_low, bci.f_high, bci.ncomp = int(f_low), int(f_high), int(ncomp)
    while (bci.tmax-bci.tmin)<1: bci.tmax+=0.5 # garante janela minima de 1seg
    bci.evaluate()
    return bci.acc * (-1)

if __name__ == "__main__":
    ds = 'IV2a'
    n_iter = 100
    path_to_setup = '../as_results/sbrt20/IV2a/'
    if not os.path.isdir(path_to_setup): os.makedirs(path_to_setup)
    data_split = 'as_train' # common, as_train, as_test
    overlap = True
    crossval = False
    nfolds = 5
    test_perc = 0.2 if crossval else 0.5  
    subjects = range(1,10) 
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] # 
    filtering = {'design':'DFT'}
    # clf = {'model':'SVM','kernel':{'kf':'linear'},'C':-4}
    # fl, fh = 4, 40
    # tmin, tmax = 0.5, 2.5
    # ncomp = 8
    # approach = {'option':'sbcsp', 'nbands':9}
    
    header = ['subj','A','B','tmin','tmax','fl','fh','ncsp','nbands','clf','clf_details','as_train','as_test','sb_dft','sb_iir','cla_dft','cla_iir']
    R = pd.DataFrame(columns=header)
    
    ##%% ###########################################################################
    for suj in subjects:
        sname = 'A0' + str(suj) + ''
        data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/'+sname+'.npy', allow_pickle=True)
        for class_ids in classes:
            # data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/'+sname+'T.npy', allow_pickle=True)
            print(f'###### {suj} {class_ids} ######')        
            space = (
                hp.uniformint('fl', 0, 20),
                hp.uniformint('fh', 21, 50),
                hp.quniform('tmin', 0, 2, 0.5),
                hp.quniform('tmax', 2, 4, 0.5),
                hp.quniform('ncomp', 2, 22, 2), 
                hp.uniformint('nbands', 1, 50), #
                hp.choice('clf', [
                    {'model':'LDA'},
                    {'model':'SVM','C':hp.quniform('C', -8, 0, 1), 
                     'kernel':hp.choice('kernel',[{'kf':'linear'},{'kf':'poly'},{'kf':'sigmoid'},{'kf':'rbf'}])},
                    {'model':'KNN','neig':hp.uniformint('neig', 2, 50),
                     'metric':hp.choice('metric',['euclidean','manhattan','minkowski','chebyshev'])},
                    {'model':'MLP','eta':hp.quniform('eta', -5, -2, 1), 
                     'n_neurons':hp.quniform('n_neurons', 50, 500, 50),
                     'n_hidden':hp.uniformint('n_hidden', 1, 2), 
                     'activ':hp.choice('activ',[{'af':'logistic'},{'af':'tanh'}])},
                    {'model':'DTree','crit':hp.choice('crit',['gini','entropy'])},
                    {'model':'Bayes'}])
                )
             
            bci.data, bci.events, bci.class_ids, bci.fs, bci.overlap = data, events, class_ids, info['fs'], overlap
            bci.crossval, bci.nfolds, bci.test_perc, bci.split = crossval, nfolds, test_perc, data_split
            bci.filt_info = filtering 
            # bci.clf = clf 
            # bci.ap = approach
            # bci.f_low, bci.f_high = fl, fh
            # bci.tmin, bci.tmax = tmin, tmax
            # bci.ncomp = ncomp
            
            path_to_trials = path_to_setup + sname + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
            acc_train = -1
            # for cont in range(10):
            try:
                trials = pickle.load(open(path_to_trials, 'rb'))
                acc_train = ((-1) * trials.best_trial['result']['loss'])
            except:
                trials = base.Trials()  
            # trials = generate_trials_to_calculate(init_vals)
            init_vals = [{'fl':4,'fh':40,'tmin':0.5,'tmax':2.5,'ncomp':8,'nbands':9,'model':'SVM','C':-4,'kf':'linear'}] 
            if acc_train < 1:
                try:
                    print('N trials: ' + str(len(trials)))
                    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=0, points_to_evaluate=init_vals)
                    pickle.dump(trials, open(path_to_trials, 'wb'))
                except:
                    print('Exception raised')
                    pickle.dump(trials, open(path_to_trials, 'wb'))
                    raise  
            # else: print(suj, class_ids, trials.best_trial['result']['loss'], trials.best_trial['misc']['vals'])
            
            ##%% ###########################################################################
            trials = pickle.load(open(path_to_trials, 'rb'))
            acc_train = (-1) * trials.best_trial['result']['loss']
            best = trials.best_trial['misc']['vals']
                        
            fl = int(best['fl'][0])
            fh = int(best['fh'][0])                       
            ncsp = int(best['ncomp'][0])
            tmin = best['tmin'][0]
            tmax = best['tmax'][0]
            nbands = int(best['nbands'][0])
            
            while (tmax-tmin)<1: # garante janela minima de 1seg
                print(tmax, tmax+0.5)
                tmax+=0.5 
            
            if nbands > (fh-fl): 
                print(nbands, (fh-fl))
                nbands = (fh-fl)
            
            approach = {'option': 'sbcsp', 'nbands': nbands}

            if best['clf'][0] == 0: clf = {'model':'LDA'}
            elif best['clf'][0] == 1: 
                svm_kernel = 'linear' if best['kernel'][0]==0 else 'poly' if best['kernel'][0]==1 else 'sigmoid' if best['kernel'][0]==2  else 'rbf'
                clf = {'model':'SVM','kernel':{'kf':svm_kernel},'C':int(best['C'][0])}
            elif best['clf'][0] == 2: 
                knn_metric = 'euclidean' if best['metric'][0]==0 else 'manhattan' if best['metric'][0]==1 else 'minkowski' if best['metric'][0]==2 else 'chebyshev'
                clf = {'model':'KNN','metric':knn_metric,'neig':int(best['neig'][0]), }
            elif best['clf'][0] == 3:
                mlp_af = 'logistic' if best['activ'][0]==0 else 'tanh'
                clf = {'model':'MLP','eta':best['eta'][0],'activ':{'af':mlp_af},'n_neurons':int(best['n_neurons'][0]),'n_hidden':int(best['n_hidden'][0])}
            elif best['clf'][0] == 4:
                dtree_crit = 'gini' if best['crit'][0]==0 else 'entropy'
                clf = {'model':'DTree','crit':dtree_crit}
            elif best['clf'][0] == 5: clf = {'model':'Bayes'}
            
            # data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/'+sname+'E.npy', allow_pickle=True)
            
            bci_test = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, 
                           crossval=crossval, nfolds=nfolds, test_perc=test_perc, split='as_test',
                           f_low=fl, f_high=fh, tmin=tmin, tmax=tmax, ncomp=ncsp, ap=approach, 
                           filt_info=filtering, clf=clf)
            bci_test.evaluate()
            acc_test = bci_test.acc
            
            ### Fixed SBCSP-DFT
            bci_test = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, 
                           crossval=crossval, nfolds=nfolds, test_perc=test_perc, split='as_test',
                           f_low=4, f_high=40, tmin=0.5, tmax=2.5, ncomp=8, ap={'option':'sbcsp','nbands':9}, 
                           filt_info={'design':'DFT'}, clf={'model':'SVM','kernel':{'kf':'linear'},'C':-4}) 
            bci_test.evaluate()
            sb_dft = bci_test.acc
            
            ### Fixed SBCSP-IIR
            bci_test = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, 
                           crossval=crossval, nfolds=nfolds, test_perc=test_perc, split='as_test',
                           f_low=4, f_high=40, tmin=0.5, tmax=2.5, ncomp=8, ap={'option':'sbcsp','nbands':9}, 
                           filt_info={'design':'IIR','iir_order':5}, clf={'model':'SVM','kernel':{'kf':'linear'},'C':-4}) 
            bci_test.evaluate()
            sb_iir = bci_test.acc
            
            ### Fixed CSP-LDA-DFT
            bci_test = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, 
                           crossval=crossval, nfolds=nfolds, test_perc=test_perc, split='as_test',
                           f_low=8, f_high=30, tmin=0.5, tmax=2.5, ncomp=8, ap={'option':'classic'}, 
                           filt_info={'design':'DFT'}, clf={'model':'LDA'}) 
            bci_test.evaluate()
            cla_dft = bci_test.acc
            
            ### Fixed CSP-LDA-IIR
            bci_test = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, 
                           crossval=crossval, nfolds=nfolds, test_perc=test_perc, split='as_test',
                           f_low=8, f_high=30, tmin=tmin, tmax=tmax, ncomp=8, ap={'option':'classic'},
                           filt_info={'design':'IIR','iir_order':5}, clf={'model':'LDA'}) 
            bci_test.evaluate()
            cla_iir = bci_test.acc
            
            R.loc[len(R)] = [suj, class_ids[0], class_ids[1], tmin, tmax, fl, fh, ncsp, nbands, clf['model'], clf, 
                             acc_train, acc_test, sb_dft, sb_iir, cla_dft, cla_iir]
            
            # print(f"Best: {best}")
            print(f"AS(tr):{round(acc_train*100,2)} | AS(te):{round(acc_test*100,2)} | SBDFT:{round(sb_dft*100,2)} | SBIIR:{round(sb_iir*100,2)} | CLADFT:{round(cla_dft*100,2)} | CLAIIR:{round(cla_iir*100,2)}\n")

    print(f">>> AS(tr):{round(R['as_train'].mean()*100, 2)} | AS(te):{round(R['as_test'].mean()*100, 2)} | SBDFT:{round(R['sb_dft'].mean()*100,2)} | SBIIR:{round(R['sb_iir'].mean()*100,2)} | CLADFT:{round(R['cla_dft'].mean()*100,2)} | CLAIIR:{round(R['cla_iir'].mean()*100,2)} <<<")
    
    ##%% PLOT GRAFIC #####################################################################
    # acc_as = R['as_test']*100
    # ref = ['sb_dft','sb_iir']
    # plt.rcParams.update({'font.size':12})
    # plt.figure(3, facecolor='mintcream')
    # plt.subplots(figsize=(10, 12), facecolor='mintcream')
    # for i in range(2):
    #     acc_ref = R[ref[i]]*100
    #     plt.subplot(2, 1, i+1)
    #     plt.scatter(np.asarray(acc_ref).reshape(-1,1), np.asarray(acc_as).reshape(-1,1), facecolors = 'c', marker = 'o', s=50, alpha=.9, edgecolors='firebrick', zorder=3)
    #     plt.scatter(round(acc_ref.mean(),2), round(acc_as.mean(),2), facecolors = 'dodgerblue', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label=r'Acurácia Média', zorder=5)
    #     plt.plot(np.linspace(40, 110, 1000), np.linspace(40, 110, 1000), color='dimgray', linewidth=1, linestyle='--', zorder=0) #linha pontilhada diagonal - limiar 
    #     plt.ylim((48, 102))
    #     plt.xlim((48, 102))
    #     plt.xticks(np.arange(50, 102, 5))
    #     plt.yticks(np.arange(50, 102, 5)) 
    #     plt.plot(np.ones(1000)*round(acc_ref.mean(),2), np.linspace(40, round(acc_as.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) # linha pontilhada verical - acc média auto setup
    #     plt.plot(np.linspace(40, round(acc_ref.mean(),2), 1000), np.ones(1000)*round(acc_as.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) # linha pontilhada horizontal - acc média ref
    #     plt.xlabel('Acurácia ' + ('SBCSP DFT' if i==0 else 'SBCSP IIR' ) + ' (configuração única) (%)', fontsize=12)
    #     plt.ylabel('Acurácia Auto Setup (%)', fontsize=12)
    #     plt.legend(loc='lower right', fontsize=12)
    # # plt.savefig('../as_results/sbrt20/IV2a/scatter_y.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        
    ##%% SAVE RESULTS PICKLE FILE ########################################################
    # pd.to_pickle(R, path_to_setup + 'RESULTS.pkl')           
    # loaded = pd.read_pickle("../as_results/sbrt20/IV2a/RESULTS.pkl")