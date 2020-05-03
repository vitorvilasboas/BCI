"""
-*- coding: utf-8 -*-
Created on Sat Mar 14 18:08:50 2020
@author: Vitor Vilas-Boas
"""
import numpy as np
from time import time
from scripts.bci_utils import BCI 

# III3a, III4a, IV2a, IV2b, Lee19   ||   'S1', 'A01', 'B01', 'S1sess1', 'S1sess2'
data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01.npy', allow_pickle=True)
class_ids = [1, 2]

# lee19_cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
# data = data[lee19_cortex]

crossval = False
nfolds = 10
test_perc = 0.1 if crossval else 0.5 
overlap = True

fl, fh, ncsp, tmin, tmax = 0, 40, 8, 0.5, 2.5

# clf = {'model':'Bayes'}
# clf = {'model':'LDA', 'lda_solver':'svd'} # 'lda_solver': 'svd','lsqr','eigen'
# clf = {'model':'KNN', 'metric':'manhattan', 'neig':105} # 'metric': 'euclidean','manhattan','minkowski','chebyshev'
clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4} # 'kernel': 'linear', 'poly', 'sigmoid', 'rbf'
# clf = {'model':'MLP', 'eta':-4, 'activ':{'af':'tanh'}, 'alpha':-1, 'n_neurons':465, 'n_hidden':2, 'mlp_solver':'adam'} # 'mlp_solver':'adam', 'lbfgs', 'sgd' # 'af':'identity', 'logistic', 'tanh', 'relu'
# clf = {'model':'DTree', 'crit':'gini'} # 'crit': 'entropy' or 'gini'

# approach = {'option':'classic'}
approach = {'option':'sbcsp', 'nbands':9}

filtering = {'design':'DFT'}
# filtering = {'design':'IIR', 'iir_order':5}
# filtering = {'design':'FIR', 'fir_order':5}

bci = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc,
          f_low=fl, f_high=fh, tmin=tmin, tmax=tmax, ncomp=ncsp, ap=approach, filt_info=filtering, clf=clf)        
st = time()
bci.evaluate()
cost = time() - st

print(str(round(bci.acc*100,2))+'%', str(round(bci.kappa,3)), str(round(cost, 2))+'s')
if crossval: print(bci.cross_scores)