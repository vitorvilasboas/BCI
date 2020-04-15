# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import pickle
import numpy as np
from time import time
from hyperopt import base, fmin, tpe, hp
from bci_utils import BCI
# import matplotlib.pyplot as plt

ds = 'IV2a' # III3a, III4a, IV2a, IV2b, Lee19, CL, TWL

auto_setup = True
n_iter = 10

crossval = False
nfolds = 10
test_perc = 0.1 if crossval else 0.5 

overlap = True

path = '/mnt/dados/eeg_data/' + ds + '/npy/'   ## >>> ENTER THE PATH TO THE DATASET HERE

if ds == 'III3a':
    subjects = ['K3','K6','L1'] # 
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    n_eeg_channels = 60
    max_ncomp = int(n_eeg_channels/2) if subjects == ['K3'] else n_eeg_channels
    Tcue, Tpause, mi_time = 3, 7, 4
    trials_per_class = 90 if subjects == ['K3'] else 60 # K3=90, K6=60, L1=60
    fs = 250.0

elif ds == 'III4a':
    subjects = ['aa','al','av','aw','ay']
    classes = [[1, 3]]
    n_eeg_channels = 118
    max_ncomp = int(n_eeg_channels/2)
    Tcue, Tpause, mi_time = 0, 4, 4
    trials_per_class = 140
    fs = 100.0
  
elif ds == 'IV2a':        
    subjects = range(1,10) 
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] 
    n_eeg_channels = 22
    max_ncomp = n_eeg_channels
    Tcue, Tpause, mi_time = 2, 6, 4
    trials_per_class = 144
    path += 'A0'
    fs = 250.0

elif ds == 'IV2b': 
    subjects = range(1,10)
    classes = [[1, 2]]
    n_eeg_channels = 3
    max_ncomp = n_eeg_channels 
    Tcue, Tpause, mi_time = 3, 7, 4
    trials_per_class = 360 
    path += 'B0'
    fs = 250.0

elif ds == 'Lee19':
    cortex_only = True # True if only cortex channels is used
    one_session = True # True if only session is used
    lee_session = 1
    
    lee_option = ('_s' + str(lee_session) + '_cortex') if one_session and cortex_only else '_cortex' if cortex_only else ''    
    subjects = range(1, 55) 
    classes = [[1, 2]]
    Tcue, Tpause, mi_time  = 3, 7, 4
    trials_per_class = 100 if one_session else 200
    n_eeg_channels = 20 if cortex_only else 62
    max_ncomp = n_eeg_channels
    fs = 250.0
    path += 'S'
       
elif ds == 'CL':
    subjects = ['CL_LR', 'CL_LF']
    classes = [[1, 2]]
    n_eeg_channels = 16
    max_ncomp = n_eeg_channels - 4
    Tcue, Tpause, mi_time = 2, 9, 7
    trials_per_class = 50 if subjects == ['CL_LR'] else 24 # LR=50, LF=24
    fs = 125.0
    
elif ds == 'TWL':
    subjects = ['TL', 'WL'] 
    classes = [[1, 2]]
    n_eeg_channels = 8
    max_ncomp = n_eeg_channels - 2
    Tcue, Tpause, mi_time  = 3, 8, 5
    trials_per_class = 40 #20
    fs = 250.0


path_out = './asetup_trials/' + ds + ((lee_option + '/') if ds=='Lee19' else '/')
if not os.path.isdir(path_out): os.makedirs(path_out)

for suj in subjects:
    
    if ds=='Lee19' and one_session: path_to_data = path + str(suj) + '_sess' + str(lee_session) + '.npy' # '.omi' 
    else: path_to_data = path + str(suj) + '.npy'
    
    data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))
    
    if ds=='Lee19' and cortex_only:
        cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
        data = data[cortex]   
        info['eeg_channels'] = len(cortex)
        info['ch_labels'] = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
    
    for class_ids in classes:       
        if auto_setup:
            max_knn_neig = int((trials_per_class * 2) * test_perc)
            # print(trials_per_class, max_knn_neig)
                                
            space = (
                hp.uniformint('fl', 0, 20), # hp.quniform('fl', 1, 20, 1),
                hp.uniformint('fh', 30, 49),  # hp.quniform('fh', 30, 49, 1),
                hp.quniform('tmin', 0, 2, 0.5),
                hp.quniform('tmax', 2, mi_time, 0.5),
                hp.quniform('ncomp', 2, max_ncomp, 2),
                hp.choice('approach', [
                    {'option':'classic',},
                    {'option':'sbcsp', 'nbands': hp.uniformint('nbands', 2, 48)} # hp.quniform('nbands', 2, max_nbands, 1)}
                    ]),
                hp.choice('filt', [
                    {'design':'DFT'},
                    # {'design':'IIR', 'iir_order': hp.uniformint('iir_order', 1, 8)}, #hp.quniform('iir_order', 1, 8, 1)},
                    #{'design':'FIR', 'fir_order': hp.uniformint('fir_order', 1, 8)}, #hp.quniform('fir_order', 1, 8, 1)}
                    ]),
                hp.choice('clf', [
                    {'model':'Bayes'},
                    {'model':'LDA',
                      'lda_solver': hp.choice('lda_solver', ['svd','lsqr','eigen']),
                      # 'shrinkage': hp.choice('shrinkage', [None, 'auto', {'shrinkage_float':  hp.uniform('shrinkage_float', 0, 1)}]) #np.logspace(-4, 0, 1)
                      },
                    {'model':'KNN', 
                      'neig': hp.uniformint('neig', 2, max_knn_neig), # hp.quniform('neig', 2, trials_per_class, 1)
                      'metric': hp.choice('metric', ['euclidean','manhattan','minkowski','chebyshev']), #{'mf':'cityblock'}, {'mf':'cosine'}, {'mf':'l1'}, {'mf':'l2'},
                      # 'p': hp.quniform('p', 2, 50, 1)
                      },
                    {'model':'SVM', 
                      'C': hp.quniform('C', -8, 4, 1), 
                      'kernel': hp.choice('kernel', [{'kf':'linear'}, {'kf':'poly'}, {'kf':'sigmoid'}, {'kf':'rbf'}]), #'degree': hp.uniformint('degree', 2, 4)    #, 'width': hp.lognormal('width', 0, 1)
                      # 'gamma': hp.choice('gamma', ['scale', 'auto', {'gamma_float': hp.quniform('gamma_float', -9, 4, 1)}]), # hp.loguniform('gamma_float', -9, 3)  np.logspace(-9, 3, 13)),
                      },
                    {'model':'MLP', 
                      'eta': hp.quniform('eta', -5, -2, 1), 
                      # 'alpha': hp.quniform('alpha', -8, 1, 1), 
                      'n_neurons' : hp.quniform('n_neurons', 50, 500, 50), # hp.uniformint('n_neurons', 50, 500),
                      'n_hidden': hp.uniformint('n_hidden', 1, 2), # hp.quniform('n_hidden', 1, 4, 1),
                      'activ': hp.choice('activ', [{'af':'identity'},{'af':'logistic'},{'af':'tanh'},{'af':'relu'}]),
                      # 'eta_type': hp.choice('eta_type', ['constant', 'invscaling', 'adaptive']),
                      'mlp_solver': hp.choice('mlp_solver', ['adam', 'lbfgs', 'sgd']),
                      },
                    {'model':'DTree', 
                      'crit': hp.choice('crit', ['gini', 'entropy']),
                      # 'max_depth': hp.choice('max_depth', [None, {'max_depth_int': hp.qlognormal('max_depth_int', 3, 1, 1)}]), # np.random.lognormal(3, 1, 1) ]),
                      # 'min_split': hp.uniform('min_split', 0.0001, 1), #  np.random.lognormal(2, 1, 1) # hp.qlognormal('min_split', 2, 1, 1)
                      }
                    ])
                )
              
            bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc)
            path_to_trials = path_out  + ds + '_' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1])+ ('_cv' if crossval else '') + '.pkl'
            
            trials = base.Trials()
            try:
                print('Trying to pickle file')
                trials = pickle.load(open(path_to_trials, 'rb'))
            except:
                print('No trial file at specified path, creating new one')
                trials = base.Trials()
            else:
                print('File found')
            
            try:
                print('Size of object: ' + str(len(trials)))
                best = fmin(bci.objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=1)
                pickle.dump(trials, open(path_to_trials, 'wb'))
                print(suj, class_ids, best)
            except:
                print('Exception raised')
                pickle.dump(trials, open(path_to_trials, 'wb'))
                print('\n', suj, class_ids, trials.best_trial['misc']['vals'])
                raise
            
            acc = (-1) * trials.best_trial['result']['loss']
            # print(suj, class_ids, str(round(acc*100,2))+'%')
                    
        else: # NO auto-setup
            
            fl, fh, ncsp, tmin, tmax = 8, 30, 6, 0.5, 4.
            
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
        
            bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, fl, fh, tmin, tmax, ncsp, approach, filtering, clf)  
            
            st = time()
            bci.evaluate()
            cost = time() - st
            
            print(suj, class_ids, str(round(bci.acc*100,2))+'%', str(round(bci.kappa,3)), str(round(cost, 2))+'s')
            if crossval: print(bci.cross_scores)
      
        
# del globals()['events'] del globals()['data'] del globals()['best'] del globals()['trials'] del globals()['space']
