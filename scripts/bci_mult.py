#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:29:45 2020
@author: vboas
"""
import numpy as np
from time import time
from bci_utils import BCI

if __name__ == "__main__": 
    ds = 'IV2a' # III3a, III4a, IV2a, IV2b, Lee19, LINCE
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
    
    subjects = [1] # uncomment to run one subject only
    classes = [[1,2]]
    for suj in subjects:
        # path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + '' + 'S' + str(suj) + 'sess2' + '.npy' #> ENTER THE PATH TO DATASET HERE (Lee19 default)
        path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + '' + 'A0' + str(suj)  + '.npy' #> ENTER THE PATH TO DATASET HERE  
        data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))
        # data = data[[7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]]
        for class_ids in classes:       
            fl, fh, ncsp, tmin, tmax = 0, 40, 2, 0.5, 2.5

            # clf = {'model':'Bayes'}
            # clf = {'model':'LDA', 'lda_solver':'svd'} # 'lda_solver': 'svd','lsqr','eigen'
            # clf = {'model':'KNN', 'metric':'manhattan', 'neig':105} # 'metric': 'euclidean','manhattan','minkowski','chebyshev'
            clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4} # 'kernel': 'linear', 'poly', 'sigmoid', 'rbf'
            # clf = {'model':'MLP', 'eta':-4, 'activ':{'af':'tanh'}, 'alpha':-1, 'n_neurons':465, 'n_hidden':2, 'mlp_solver':'adam'} # 'mlp_solver':'adam', 'lbfgs', 'sgd' # 'af':'identity', 'logistic', 'tanh', 'relu'
            # clf = {'model':'DTree', 'crit':'gini'} # 'crit': 'entropy' or 'gini'
            
            # approach = {'option':'classic'}
            approach = {'option':'sbcsp','nbands':10}
            
            # filtering = {'design':'DFT'}
            filtering = {'design':'IIR', 'iir_order':5}
            # filtering = {'design':'FIR', 'fir_order':5}
            
            bci = BCI(data, events, class_ids, overlap, info['fs'], crossval, nfolds, test_perc, fl, fh, tmin, tmax, ncsp, approach, filtering, clf)  
            st = time()
            bci.evaluate()
            cost = time() - st
            
            print(str(round(bci.acc*100,2))+'%', str(round(bci.kappa,3)), str(round(cost, 2))+'s')
            if crossval: print(bci.cross_scores)
