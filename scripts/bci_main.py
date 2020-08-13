"""
-*- coding: utf-8 -*-
Created on Sat Mar 14 18:08:50 2020
@author: Vitor Vilas-Boas
"""
import os
import numpy as np
import pandas as pd
from time import time
from scripts.bci_utils import BCI 

if __name__ == "__main__": 
    ds = 'IV2a' # III3a, III4a, IV2a, IV2b, LINCE, Lee19
    overlap = True
    crossval = False
    nfolds = 10 
    test_perc = 0.1 if crossval else 0.5
    cortex_only = False # used when ds == Lee19 - True to used only cortex channels
    
    is_sbcsp = True
    fl, fh, ncsp, tmin, tmax, nbands = 10, 40, 22, 0.5, 2.5, 8
    ncsp_list = None # [6, 6, 2, 6] 
    
    clf = {'model':'LR'}
    # clf = {'model':'Bayes'}
    # clf = {'model':'LDA', 'lda_solver':'svd'} # 'lda_solver': 'svd','lsqr','eigen'
    # clf = {'model':'KNN', 'metric':'chebyshev', 'neig':14} # 'metric': 'euclidean','manhattan','minkowski','chebyshev'
    # clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-3} # 'kernel': 'linear', 'poly', 'sigmoid', 'rbf'
    # clf = {'model':'MLP', 'eta':-4, 'activ':{'af':'tanh'}, 'alpha':-1, 'n_neurons':465, 'n_hidden':2, 'mlp_solver':'adam'} # 'mlp_solver':'adam', 'lbfgs', 'sgd' # 'af':'identity', 'logistic', 'tanh', 'relu'
    # clf = {'model':'DTree', 'crit':'gini'} # 'crit': 'entropy' or 'gini'
    
    # approach = {'option':'classic'}
    # approach = {'option':'sbcsp', 'nbands':8}
    
    filtering = {'design':'DFT'}
    # filtering = {'design':'IIR', 'iir_order':5}
    # filtering = {'design':'FIR', 'fir_order':5} 
    
    if not is_sbcsp: ncsp_list = None
    
    prefix, suffix = '', ''
    if ds == 'III3a':
        subjects = ['K3','K6','L1'] 
        classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]  
    elif ds == 'III4a':
        subjects = ['aa','al','av','aw','ay']
        classes = [[2, 3]]
    elif ds == 'IV2a':        
        subjects = range(1,10) 
        classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
        prefix = 'A0'
    elif ds == 'IV2b': 
        subjects = range(1,10)
        classes = [[1, 2]]
        prefix = 'B0'
    elif ds == 'LINCE':
        subjects = ['CL_LR','CL_LF','TL_S1','TL_S2','WL_S1','WL_S2']
        classes = [[1, 2]] 
    elif ds == 'Lee19':
        subjects = range(1, 55) 
        classes = [[1, 2]]
        prefix = 'S'
        suffix = '' # 'sess1' or 'sess2'
        
    subjects = [4] # uncomment to run one subject only
    classes = [[1, 2]] # uncomment to run LH x RH classification only
    
    R = pd.DataFrame(columns=['acc','kpa','cost'])
    for suj in subjects:
        sname = prefix + str(suj) + suffix
        # data, events, info = labeling(path='/mnt/dados/eeg_data/+ds+'/', ds=ds, session='T', subj=suj, channels=None, save=False)
        path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + sname + '.npy' # PATH TO DATASET
        data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))
        
        if ds=='LINCE' and suj == 'CL_LF': classes = [[1, 3]]
        if ds=='Lee19' and cortex_only:
            cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
            data = data[cortex]
        for class_ids in classes:       
            bci = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc, 
                      f_low=fl, f_high=fh, tmin=tmin, tmax=tmax, ncsp=ncsp, nbands=nbands, filtering=filtering, clf=clf, is_sbcsp=is_sbcsp,
                      ncsp_list=ncsp_list)        
            st = time()
            bci.evaluate()
            cost = time() - st
            # print(suj, class_ids, str(round(bci.acc*100,2))+'%', str(round(bci.kpa,3)), str(round(cost, 2))+'s')
            print(suj, round(bci.acc*100,2))
            # if crossval: print(bci.cross_scores)
            learner = bci.learner
            R.loc[len(R)] = [bci.acc, bci.kpa, cost]        
    # print(R.mean())