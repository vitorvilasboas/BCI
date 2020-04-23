# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import re
import os
import mne
import math
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time, sleep
from sklearn.svm import SVC
from scipy.io import loadmat, savemat
from scipy.stats import norm
from datetime import datetime
from scipy.fftpack import fft
from scipy.linalg import eigh
from sklearn.pipeline import Pipeline
from functools import reduce
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter, decimate, welch
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import cohen_kappa_score 
from hyperopt import base, fmin, tpe, hp
from bci_utils import extractEpochs, nanCleaner, Filter


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
        return XVAR


class BCI():
    
    def __init__(self, data, events, class_ids, overlap, fs, crossval, nfolds, test_perc,
                 f_low=None, f_high=None, tmin=None, tmax=None, ncomp=None, ap=None, filt_info=None, clf=None):
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
        self.ap = ap
        self.filt_info = filt_info
        self.clf = clf
        self.acc = None
        self.kappa = None
        self.ncsp_list = None
        while (self.tmax-self.tmin)<1: self.tmax+=0.5 # garante janela minima de 1seg
        
    
    def objective(self, args):
        # print(self.class_ids, self.overlap, self.fs, self.crossval, self.nfolds, self.test_perc, 
        #       self.f_low, self.f_high, self.tmin, self.tmax, self.ncomp, self.ap, self.filt_info, self.clf)
        self.csp_list = list(map(lambda x: int(x), args))
        # self.f_low, self.f_high, self.ncomp = int(self.f_low), int(self.f_high), int(self.ncomp)
        # self.acc, self.kappa = self.evaluate()
        self.evaluate()
        return self.acc * (-1)
        # return np.random.random(1)[0] * (-1)
    
    
    def evaluate(self):
        
        if self.clf['model'] == 'LDA': 
            # if clf_dict['lda_solver'] == 'svd': lda_shrinkage = None
            # else:
            #     lda_shrinkage = self.clf['shrinkage'] if self.clf['shrinkage'] in [None,'auto'] else self.clf['shrinkage']['shrinkage_float']
            self.clf_final = LDA(solver=self.clf['lda_solver'], shrinkage=None)
        
        if self.clf['model'] == 'Bayes': 
            self.clf_final = GaussianNB()
        
        if self.clf['model'] == 'SVM': 
            # degree = self.clf['kernel']['degree'] if self.clf['kernel']['kf'] == 'poly' else 3
            # gamma = self.clf['gamma'] if self.clf['gamma'] in ['scale', 'auto'] else 10 ** (self.clf['gamma']['gamma_float'])
            self.clf_final = SVC(kernel=self.clf['kernel']['kf'], C=10 ** (self.clf['C']), 
                                 gamma='scale', degree=3, probability=True) # degree=3,
        
        if self.clf['model'] == 'KNN':
            self.clf_final = KNeighborsClassifier(n_neighbors=int(self.clf['neig']), 
                                                  metric=self.clf['metric'], p=3) # p=self.clf['p'] #metric='minkowski', p=3)  # minkowski,p=2 -> distancia euclidiana padrão
                                                  
        if self.clf['model'] == 'DTree': 
            # print(self.clf['min_split'])
            # if self.clf['min_split'] == 1.0: self.clf['min_split'] += 1
            # max_depth = self.clf['max_depth'] if self.clf['max_depth'] is None else int(self.clf['max_depth']['max_depth_int'])
            self.clf_final = DecisionTreeClassifier(criterion=self.clf['crit'], random_state=0,
                                                    max_depth=None, # max_depth=max_depth,
                                                    min_samples_split=2 # min_samples_split=self.clf['min_split'], #math.ceil(self.clf['min_split']),
                                                    ) # None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
                
        if self.clf['model'] == 'MLP':   
            self.clf_final = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4,
                                           learning_rate_init=10**self.clf['eta'],
                                           # alpha=10**self.clf['alpha'],
                                           activation=self.clf['activ']['af'],
                                           hidden_layer_sizes=(int(self.clf['n_neurons']), int(self.clf['n_hidden'])),
                                           learning_rate='constant', # self.clf['eta_type'], 
                                           solver=self.clf['mlp_solver'],
                                           )
        
        # cortex_ch = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
        # self.data = self.data[cortex_ch]
        
        smin = math.floor(self.tmin * self.fs)
        smax = math.floor(self.tmax * self.fs)
        self.buffer_len = smax - smin
        
        self.dft_rf = self.fs/self.buffer_len # resolução em frequência fft
        self.dft_size_band = round(2/self.dft_rf) # 2 representa sen e cos que foram separados do componente complexo da fft intercalados
        
        self.epochs, self.labels = extractEpochs(self.data, self.events, smin, smax, self.class_ids)
        self.epochs = nanCleaner(self.epochs)
        # self.epochs = np.asarray([ nanCleaner(ep) for ep in self.epochs ])
        
        # self.epochs, self.labels = self.epochs[:int(len(self.epochs)/2)], self.labels[:int(len(self.labels)/2)] # Lee19 somente sessão 1
        # self.epochs, self.labels = self.epochs[int(len(self.epochs)/2):], self.labels[int(len(self.labels)/2):] # Lee19 somente sessão 2
        
        self.filt = Filter(self.f_low, self.f_high, self.buffer_len, self.fs, self.filt_info)
        
        self.csp = CSP(n_components=self.ncomp)
            
        if self.crossval:
            
            self.cross_scores = []
            self.cross_kappa = []
            
            kf = StratifiedShuffleSplit(self.nfolds, test_size=self.test_perc, random_state=42)
            #kf = StratifiedKFold(self.nfolds, False)
            
            if self.ap['option'] == 'classic':
                
                XF = self.filt.apply_filter(self.epochs)
        
                if self.filt_info['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
                    bmin = self.f_low * self.dft_size_band
                    bmax = self.f_high * self.dft_size_band
                    XF = XF[:, :, bmin:bmax]
                    # print(bmin, bmax)
                
                # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
                # self.cross_scores = cross_val_score(self.chain, XF, self.labels, cv=kf)
                
                for idx_treino, idx_teste in kf.split(XF, self.labels):
                    XT, XV, yT, yV = XF[idx_treino], XF[idx_teste], self.labels[idx_treino], self.labels[idx_teste]
                    
                    
                    # Option 1
                    self.csp.fit(XT, yT)
                    XT_CSP = self.csp.transform(XT)
                    XV_CSP = self.csp.transform(XV) 
                    self.clf_final.fit(XT_CSP, yT)
                    self.scores = self.clf_final.predict(XV_CSP)
                    # self.csp_filters = self.csp.filters_
                    
                    # Option 2
                    # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
                    # self.chain.fit(XT, yT)
                    # self.scores = self.chain.predict(XV)
                    # self.csp_filters = self.chain['CSP'].filters_
                    
                    acc_fold = np.mean(self.scores == yV) # or self.cross_scores.append(self.chain.score(XV, yV)) 
                    kappa_fold = cohen_kappa_score(self.scores, yV)     
                    self.cross_scores.append(acc_fold)
                    self.cross_kappa.append(kappa_fold)
                
            
            elif self.ap['option'] == 'sbcsp':
                
                for idx_treino, idx_teste in kf.split(self.epochs, self.labels):
                    acc_fold, kappa_fold = self.sbcsp_approach(self.epochs[idx_treino], self.epochs[idx_teste], 
                                                               self.labels[idx_treino], self.labels[idx_teste])
                    self.cross_scores.append(acc_fold)
                    self.cross_kappa.append(kappa_fold)
            
            self.acc = np.mean(self.cross_scores) 
            self.kappa = np.mean(self.cross_kappa) 
            
            
        else: # NO crossval
            
            test_size = int(len(self.epochs) * self.test_perc)
            train_size = int(len(self.epochs) - test_size)
            train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
            epochsT, labelsT = self.epochs[:train_size], self.labels[:train_size] 
            epochsV, labelsV = self.epochs[train_size:], self.labels[train_size:]
            
            XT = [ epochsT[np.where(labelsT == i)] for i in self.class_ids ] # Extrair épocas de cada classe
            XV = [ epochsV[np.where(labelsV == i)] for i in self.class_ids ]
            
            XT = np.concatenate([XT[0],XT[1]]) # Train data classes A + B
            XV = np.concatenate([XV[0],XV[1]]) # Test data classes A + B        
            yT = np.concatenate([self.class_ids[0] * np.ones(int(len(XT)/2)), self.class_ids[1] * np.ones(int(len(XT)/2))])
            yV = np.concatenate([self.class_ids[0] * np.ones(int(len(XV)/2)), self.class_ids[1] * np.ones(int(len(XV)/2))])
            # print(XT.shape, XV.shape)
           
            if self.ap['option'] == 'classic':
                
                XTF = self.filt.apply_filter(XT)
                XVF = self.filt.apply_filter(XV)
                
                if self.filt_info['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
                    bmin = self.f_low * self.dft_size_band
                    bmax = self.f_high * self.dft_size_band
                    XTF = XTF[:, :, bmin:bmax]
                    XVF = XVF[:, :, bmin:bmax]
                    # print(bmin, bmax)
                
                # Option 1
                self.csp.fit(XTF, yT)
                XT_CSP = self.csp.transform(XTF)
                XV_CSP = self.csp.transform(XVF) 
                self.clf_final.fit(XT_CSP, yT)
                self.scores = self.clf_final.predict(XV_CSP)
                # self.csp_filters = self.csp.filters_
                
                # Option 2
                # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
                # self.chain.fit(XTF, yT)           
                # self.scores = self.chain.predict(XVF)
                # self.csp_filters = self.chain['CSP'].filters_
                
                self.acc = np.mean(self.scores == yV) # or chain.score(XVF, yV)     
                self.kappa = cohen_kappa_score(self.scores, yV)
                
            elif self.ap['option'] == 'sbcsp': 
                self.acc, self.kappa = self.sbcsp_approach(XT, XV, yT, yV)
        
    
    def sbcsp_approach(self, XT, XV, yT, yV):
        
        nbands = int(self.ap['nbands'])
        if nbands > (self.f_high-self.f_low): nbands = (self.f_high-self.f_low)
    
        
        n_bins = self.f_high - self.f_low
        overlap = 0.5 if self.overlap else 1
        step = n_bins / nbands
        size = step / overlap
        
        sub_bands = []
        for i in range(nbands):
            fl_sb = round(i * step + self.f_low)
            fh_sb = round(i * step + size + self.f_low)
            # if fl_sb == 0: fl_sb = 0.001
            if fh_sb <= self.f_high: sub_bands.append([fl_sb, fh_sb]) 
            # se ultrapassar o limite superior da banda total, desconsidera a última sub-banda
            # ... para casos em que a razão entre a banda total e n_bands não é exata 
        
        # print(sub_bands)
        nbands = len(sub_bands)
        
        XTF, XVF = [], []
        if self.filt_info['design'] == 'DFT':
            
            XT_FFT = self.filt.apply_filter(XT)
            XV_FFT = self.filt.apply_filter(XV)
            for i in range(nbands):
                bmin = sub_bands[i][0] * self.dft_size_band
                bmax = sub_bands[i][1] * self.dft_size_band
                XTF.append(XT_FFT[:, :, bmin:bmax])
                XVF.append(XV_FFT[:, :, bmin:bmax]) 
                # print(bmin, bmax)
        
        elif self.filt_info['design'] in ['IIR' or 'FIR']:
            
            for i in range(nbands):
                filt_sb = Filter(sub_bands[i][0], sub_bands[i][1], len(XT[0,0,:]), self.fs, self.filt_info)
                XTF.append(filt_sb.apply_filter(XT))
                XVF.append(filt_sb.apply_filter(XV))
        
        self.chain = [ Pipeline([('CSP', CSP(n_components=self.csp_list[i])), ('LDA', LDA())]) for i in range(nbands) ]
        
        for i in range(nbands): self.chain[i]['CSP'].fit(XTF[i], yT)
            
        XT_CSP = [ self.chain[i]['CSP'].transform(XTF[i]) for i in range(nbands) ]
        XV_CSP = [ self.chain[i]['CSP'].transform(XVF[i]) for i in range(nbands) ]
        
        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        for i in range(len(sub_bands)):
            # print(XT_CSP[i].shape)
            self.chain[i]['LDA'].fit(XT_CSP[i], yT)
            SCORE_T[:, i] = np.ravel(self.chain[i]['LDA'].transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(self.chain[i]['LDA'].transform(XV_CSP[i]))
        
        # csp_filters_sblist = [self.chain[i]['CSP'].filters_ for i in range(nbands)]
        # lda_sblist = [self.chain[i]['LDA'] for i in range(nbands)] 
            

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
    

if __name__ == "__main__":
    ds = 'IV2a' # III3a, III4a, IV2a, IV2b, Lee19, LINCE
    auto_setup = True
    n_iter = 250
    crossval = False
    nfolds = 10
    test_perc = 0.1 if crossval else 0.5 
    overlap = True
    
    if ds == 'III3a':
        subjects = ['K3','K6','L1'] 
        classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    
    elif ds == 'III4a':
        subjects = ['aa','al','av','aw','ay']
        classes = [[1, 3]]
    
    elif ds == 'IV2a':        
        subjects = range(1,10) 
        classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] 
    
    elif ds == 'IV2b': 
        subjects = range(1,10)
        classes = [[1, 2]]
       
    elif ds == 'LINCE':
        subjects = ['CL_LR','CL_LF','TL_S1','TL_S2','WL_S1','WL_S2']
        classes = [[1, 2]]
    
    elif ds == 'Lee19':
        subjects = range(1, 55) 
        classes = [[1, 2]]
        cortex_only = True # True if only cortex channels is used
    
    R = pd.read_pickle('./asetup_trials/' + ds + '/results_v1_' + ds + '.pkl')  
    # print(ds, R['acc'].mean(), R['acc'].median(), R['acc'].std(), R['acc'].max(), R['acc'].min())
    
    df = []
    # subjects = [1] # uncomment to run one subject only
    for suj in subjects:
        # path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + '' + 'S' + str(suj) + 'sess2' + '.npy' #> ENTER THE PATH TO DATASET HERE  
        path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + '' + 'A0' + str(suj) + '.npy' #> ENTER THE PATH TO DATASET HERE  
        data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))
        
        if ds=='Lee19' and cortex_only:
            cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
            data = data[cortex]   
            info['eeg_channels'] = len(cortex)
            info['ch_labels'] = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
        
        for class_ids in classes: # [[1,2]]
            
            args = R.loc[(R['subj'] == suj) & (R['A'] == class_ids[0]) & (R['B'] == class_ids[1])]
            args.index = [0]
            # print(args['clf_details'])
            
            if int(args['nbands'])==1:
                df.append([suj, class_ids[0], class_ids[1], int(args['ncsp']), int(args['nbands']), args['acc'][0], args['acc'][0], 
                                 args['tmin'][0], args['tmax'][0], args['fl'][0], args['fh'][0], args['clf'][0], args['clf_details'][0]])
            else:
                desvio = 4 # desvio em torno do ncsp ótimo (deve ser par)
                min_ncsp = (int(args['ncsp']) - desvio) if (int(args['ncsp']) - desvio) > 2 else 2
                max_ncsp = (int(args['ncsp']) + desvio) if (int(args['ncsp']) + desvio) < info['eeg_channels'] else info['eeg_channels']
    
                space = []
                for i in range(int(args['nbands'])):
                    space.append(hp.quniform('csp'+str(i), min_ncsp, max_ncsp, 2))
                tuple(space)
                
                approach = {'option':'sbcsp', 'nbands':int(args['nbands'])}
               
                bci = BCI(data, events, class_ids, overlap, info['fs'], crossval, nfolds, test_perc,
                          f_low=int(args['fl']), f_high=int(args['fh']), tmin=float(args['tmin']), tmax=float(args['tmax']), 
                          ncomp=int(args['ncsp']), ap=approach, filt_info={'design':'DFT'}, clf=args['clf_details'][0])
                
                path_to_trials = './asetup_trials/tuning_ncsp/' + ds + '/'
                if not os.path.isdir(path_to_trials): os.makedirs(path_to_trials)
                
                path_to_trials2 = path_to_trials  + ds + '_' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + \
                    ('_cv' if crossval else '') + '.pkl'
                
                trials = base.Trials()
                try:
                    print('Trying to pickle file')
                    trials = pickle.load(open(path_to_trials2, 'rb'))
                except:
                    print('No trial file at specified path, creating new one')
                    trials = base.Trials()
                else:
                    print('File found')
                
                try:
                    print('Size of object: ' + str(len(trials)))
                    best = fmin(bci.objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=1)
                    pickle.dump(trials, open(path_to_trials2, 'wb'))
                    # print(suj, class_ids, best)
                except:
                    print('Exception raised')
                    pickle.dump(trials, open(path_to_trials2, 'wb'))
                    # print('\n', suj, class_ids, trials.best_trial['misc']['vals'])
                    raise
                
                acc = (-1) * trials.best_trial['result']['loss']
                # print(suj, class_ids, str(round(acc*100,2))+'%')
                
                best = trials.best_trial['misc']['vals']
    
                df.append(np.r_[[suj, class_ids[0], class_ids[1], int(args['ncsp']), int(args['nbands']), args['acc'][0], acc, 
                                 args['tmin'][0], args['tmax'][0], args['fl'][0], args['fh'][0], args['clf'][0], args['clf_details'][0]], 
                                [ int(best['csp'+str(i)][0]) for i in range(int(args['nbands']))]])
    
    header = ['subj', 'A', 'B', 'ncsp', 'nbands', 'acc', 'acc_tune', 'tmin', 'tmax', 'fl', 'fh', 'clf', 'clf_details']#, 
              # 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n17', 
              # 'n18', 'n19', 'n20', 'n21', 'n22', 'n23', 'n24', 'n25', 'n26', 'n27', 'n28', 'n29', 'n30', 'n31', 'n32', 
              # 'n33', 'n34', 'n35', 'n36', 'n37', 'n38', 'n39', 'n40', 'n41', 'n42', 'n43', 'n44', 'n45', 'n46', 'n47']
    
    
    header_tune = ['n'+str(i+1) for i in range( max(list(map(lambda l: len(l), df))) - len(header) )]
    
    FINAL = pd.DataFrame(df, columns=header+header_tune) 
    
          
    pd.to_pickle(FINAL, path_to_trials + 'results_tuning_' + ds + '.pkl')
    FINAL.to_csv(path_to_trials + 'results_tuning_' + ds + '.csv', index=False)
    # del globals()['events'] del globals()['data'] del globals()['best'] del globals()['trials'] del globals()['space']


np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)


 



