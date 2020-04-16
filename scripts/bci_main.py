"""
-*- coding: utf-8 -*-
Created on Sat Mar 14 18:08:50 2020
@author: Vitor Vilas-Boas
"""
import math
import pickle
import numpy as np
from time import time
from scipy.stats import norm
from sklearn.svm import SVC
from scipy.io import loadmat
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from bci_utils import extractEpochs, nanCleaner, Filter, CSP
from sklearn.metrics import cohen_kappa_score
from scripts.bci_utils import BCI 

# III3a, III4a, IV2a, IV2b, Lee19   ||   'S1', 'A01', 'B01', 'S1sess1', 'S1sess2'
data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/A01.npy', allow_pickle=True)
class_ids = [1, 2]

# cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
# data = data[cortex]
# info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
# info['eeg_channels'] = len(cortex)

crossval = False
nfolds = 10
test_perc = 0.1 if crossval else 0.5 
overlap = True

fl, fh, ncsp, tmin, tmax = 8, 30, 8, 0.5, 2.5

# clf = {'model':'Bayes'}
clf = {'model':'LDA', 'lda_solver':'svd'} # 'lda_solver': 'svd','lsqr','eigen'
# clf = {'model':'KNN', 'metric':'manhattan', 'neig':105} # 'metric': 'euclidean','manhattan','minkowski','chebyshev'
# clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4} # 'kernel': 'linear', 'poly', 'sigmoid', 'rbf'
# clf = {'model':'MLP', 'eta':-4, 'activ':{'af':'tanh'}, 'alpha':-1, 'n_neurons':465, 'n_hidden':2, 'mlp_solver':'adam'} # 'mlp_solver':'adam', 'lbfgs', 'sgd' # 'af':'identity', 'logistic', 'tanh', 'relu'
# clf = {'model':'DTree', 'crit':'gini'} # 'crit': 'entropy' or 'gini'

approach = {'option':'classic'}
# approach = {'option':'sbcsp', 'nbands':10}

# filtering = {'design':'DFT'}
filtering = {'design':'IIR', 'iir_order':5}
# filtering = {'design':'FIR', 'fir_order':5}

bci = BCI(data, events, class_ids, overlap, info['fs'], crossval, nfolds, test_perc, fl, fh, tmin, tmax, ncsp, approach, filtering, clf)  
st = time()
bci.evaluate()
cost = time() - st

print(str(round(bci.acc*100,2))+'%', str(round(bci.kappa,3)), str(round(cost, 2))+'s')
if crossval: print(bci.cross_scores)




# smin = math.floor(tmin * info['fs'])
# smax = math.floor(tmax * info['fs'])
# buffer_len = smax - smin

# filt = Filter(f_low, f_high, buffer_len, info['fs'], filtering)
# csp = CSP(n_components=ncomp)
# clf_final = LDA(solver=clf['lda_solver'], shrinkage=None)

# epochs, labels = extractEpochs(data, events, smin, smax, class_ids)
# # epochs = nanCleaner(epochs)
# # epochs, labels = epochs[:int(len(epochs)/2)], labels[:int(len(labels)/2)] # somente sessão 1
# # epochs, labels = epochs[int(len(epochs)/2):], labels[int(len(labels)/2):] # somente sessão 2

# test_size = int(len(epochs) * 0.5)
# train_size = int(len(epochs) - test_size)
# train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
# epochsT, labelsT = epochs[:train_size], labels[:train_size] 
# epochsV, labelsV = epochs[train_size:], labels[train_size:]

# XT = [ epochsT[np.where(labelsT == i)] for i in class_ids ] # Extrair épocas de cada classe
# XV = [ epochsV[np.where(labelsV == i)] for i in class_ids ]
# XT = np.concatenate([XT[0],XT[1]]) # Train data classes A + B
# XV = np.concatenate([XV[0],XV[1]]) # Test data classes A + B        
# yT = np.concatenate([class_ids[0] * np.ones(int(len(XT)/2)), class_ids[1] * np.ones(int(len(XT)/2))])
# yV = np.concatenate([class_ids[0] * np.ones(int(len(XV)/2)), class_ids[1] * np.ones(int(len(XV)/2))])

# XTF = filt.apply_filter(XT)
# XVF = filt.apply_filter(XV)

# csp.fit(XTF, yT)
# XT_CSP = csp.transform(XTF)
# XV_CSP = csp.transform(XVF) 
# clf_final.fit(XT_CSP, yT)
# scores = clf_final.predict(XV_CSP)
# csp_filters = csp.filters_

# acc = np.mean(scores == yV) # or chain.score(XVF, yV)     
# kappa = cohen_kappa_score(scores, yV)

# print(str(round(acc*100,2))+'%', str(round(kappa,3)))