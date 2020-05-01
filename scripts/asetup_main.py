# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import pickle
import numpy as np
import pandas as pd
from time import time
from hyperopt import base, fmin, tpe, hp
from bci_utils import BCI
# import matplotlib.pyplot as plt

ds = 'IV2a' # III3a, III4a, IV2a, IV2b, Lee19, LINCE
auto_setup = True
n_iter = 1000
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

# header = ['subj', 'A', 'B', 'tmin', 'tmax', 'acc', 'acc2', 'kappa', 'cost', 'fl', 'fh', 'f_type', 'forder', 'ncsp', 'approach', 'nbands', 'clf', 'clf details', 'acc_cla_dft', 
#           'acc_cla_iir', 'acc_sb_dft', 'acc_sb_iir', 'kpa_cla_dft', 'kpa_cla_iir', 'kpa_sb_dft', 'kpa_sb_iir', 'cost_cla_dft', 'cost_cla_iir', 'cost_sb_dft', 'cost_sb_iir', 
#           'lda_solver', 'knn_metric', 'knn_neig', 'svm_kernel', 'svm_clog', 'dtree_crit', 'mlp_n_hidden', 'mlp_n_neurons', 'mlp_eta', 'mlp_af', 'mlp_solver']

header = ['subj', 'A', 'B', 'tmin', 'tmax', 'fl', 'fh', 'ncsp', 'nbands', 'clf', 'clf_details', 'acc', 'cla_dft', 'cla_iir', 'sb_dft', 'sb_iir']
df = pd.DataFrame(columns=header)

# subjects = [1] # uncomment to run one subject only
for suj in subjects:
    # path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + '' + 'S' + str(suj) + 'sess2' + '.npy' #> ENTER THE PATH TO DATASET HERE (Lee19 default)
    path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + '' + 'A0' + str(suj)  + '.npy' #> ENTER THE PATH TO DATASET HERE  
    data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))
    
    if ds=='Lee19' and cortex_only:
        cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
        data = data[cortex]   
        info['eeg_channels'] = len(cortex)
        info['ch_labels'] = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
    
    for class_ids in classes:       
        max_knn_neig = int((info['trials_per_class'] * 2) * test_perc)
        # print(trials_per_class, max_knn_neig)
                            
        space = (
            hp.uniformint('fl', 0, 20), # hp.quniform('fl', 1, 20, 1),
            hp.uniformint('fh', 30, 49),  # hp.quniform('fh', 30, 49, 1),
            hp.quniform('tmin', 0, 2, 0.5),
            hp.quniform('tmax', 2, info['trial_mi_time'], 0.5),
            hp.quniform('ncomp', 2, info['eeg_channels'], 2),
            hp.choice('approach', [
                {'option':'classic',},
                {'option':'sbcsp', 'nbands': hp.uniformint('nbands', 2, 48)} # hp.quniform('nbands', 2, max_nbands, 1)}
                ]),
            hp.choice('filt', [
                {'design':'DFT'},
                # {'design':'IIR', 'iir_order': hp.uniformint('iir_order', 1, 8)}, #hp.quniform('iir_order', 1, 8, 1)},
                # {'design':'FIR', 'fir_order': hp.uniformint('fir_order', 1, 8)}, #hp.quniform('fir_order', 1, 8, 1)}
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
                  'C': hp.quniform('C', -8, 0, 1), 
                  'kernel': hp.choice('kernel', [{'kf':'linear'}, {'kf':'poly'}, {'kf':'sigmoid'}, {'kf':'rbf'}]), #'degree': hp.uniformint('degree', 2, 4)    #, 'width': hp.lognormal('width', 0, 1)
                  # 'gamma': hp.choice('gamma', ['scale', 'auto', {'gamma_float': hp.quniform('gamma_float', -9, 4, 1)}]), # hp.loguniform('gamma_float', -9, 3)  np.logspace(-9, 3, 13)),
                  },
                {'model':'MLP', 
                  'eta': hp.quniform('eta', -5, -2, 1), 
                  'n_neurons' : hp.quniform('n_neurons', 50, 500, 50), # hp.uniformint('n_neurons', 50, 500),
                  'n_hidden': hp.uniformint('n_hidden', 1, 2), # hp.quniform('n_hidden', 1, 4, 1),
                  'activ': hp.choice('activ', [{'af':'identity'},{'af':'logistic'},{'af':'tanh'},{'af':'relu'}]),
                  'mlp_solver': hp.choice('mlp_solver', ['adam', 'lbfgs', 'sgd'])
                  # 'alpha': hp.quniform('alpha', -8, 1, 1),
                  # 'eta_type': hp.choice('eta_type', ['constant', 'invscaling', 'adaptive']),,
                  },
                {'model':'DTree', 
                  'crit': hp.choice('crit', ['gini', 'entropy']),
                  # 'max_depth': hp.choice('max_depth', [None, {'max_depth_int': hp.qlognormal('max_depth_int', 3, 1, 1)}]), # np.random.lognormal(3, 1, 1) ]),
                  # 'min_split': hp.uniform('min_split', 0.0001, 1), #  np.random.lognormal(2, 1, 1) # hp.qlognormal('min_split', 2, 1, 1)
                  }
                ])
            )
          
        bci = BCI(data, events, class_ids, overlap, info['fs'], crossval, nfolds, test_perc)
        
        path_to_trials = './asetup_trials/' + ds + '/'
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
            # pickle.dump(trials, open(path_to_trials2, 'wb'))
            print(suj, class_ids, best)
        except:
            print('Exception raised')
            # pickle.dump(trials, open(path_to_trials2, 'wb'))
            print('\n', suj, class_ids, trials.best_trial['misc']['vals'])
            raise
        
        acc = (-1) * trials.best_trial['result']['loss']
        # print(suj, class_ids, str(round(acc*100,2))+'%')
        
        best = trials.best_trial['misc']['vals']
                 
        fl = int(best['fl'][0])
        fh = int(best['fh'][0])                       
        ncsp = int(best['ncomp'][0])
        tmin = best['tmin'][0]
        tmax = best['tmax'][0]
        filt_type = 'DFT' if best['filt'][0]==0 else 'IIR' if best['filt'][0]==1 else 'FIR'
        filt_order = None if best['filt'][0]==0 else int(best['iir_order'][0]) if best['filt'][0]==1 else int(best['fir_order'][0])
        ap = 'classic' if best['approach'][0]==0 else 'sbcsp'
        nbands = int(best['nbands'][0]) if best['approach'][0]==1 else 1
        while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
        if nbands > (fh-fl): nbands = (fh-fl)
                  
        clf_type, clf_details = '', ''
        lda_solver = knn_metric = knn_neig = svm_kernel = svm_clog = mlp_n_hidden = mlp_n_neurons = mlp_eta = mlp_af = mlp_solver = dtree_crit = None
        
        
        if best['clf'][0] == 0: 
            clf_type += 'Bayes'
            clf = {'model': clf_type}
        
        elif best['clf'][0] == 1: 
            clf_type += 'LDA'
            lda_solver = 'svd' if best['lda_solver'][0] == 0 else 'lsqr' if best['lda_solver'][0] == 1 else 'eigen'
            clf_details += "solver={}".format(lda_solver)
            
            clf = {'model': clf_type, 'lda_solver': lda_solver}
        
        elif best['clf'][0] == 2: 
            clf_type += 'KNN'
            knn_metric = 'euclidean' if best['metric'][0] == 0 else 'manhattan' if best['metric'][0] == 1 else 'minkowski' if best['metric'][0] == 2 else 'chebyshev'
            knn_neig = int(best['neig'][0])
            clf_details += 'neig={}, metric={}'.format(knn_neig, knn_metric)
            
            clf = {'model': clf_type, 'metric': knn_metric, 'neig': knn_neig, }
               
        elif best['clf'][0] == 3: 
            clf_type += 'SVM'
            svm_kernel = 'linear' if best['kernel'][0]==0 else 'poly' if best['kernel'][0]==1 else 'sigmoid' if best['kernel'][0]==2  else 'rbf'
            svm_clog = int(best['C'][0])
            clf_details += 'k={}, C=10**({})'.format(svm_kernel, svm_clog)
            
            clf = {'model': clf_type, 'kernel': {'kf': svm_kernel}, 'C': svm_clog}
        
        elif best['clf'][0] == 4:
            clf_type += 'MLP'
            mlp_af = 'identity' if best['activ'][0]==0 else 'logistic' if best['activ'][0]==1 else 'tanh' if best['activ'][0]==2  else 'relu'
            mlp_solver = 'adam' if best['mlp_solver'][0] == 0 else 'lbfgs' if best['mlp_solver'][0] == 1 else 'sgd'
            mlp_n_hidden = int(best['n_hidden'][0])
            mlp_n_neurons = int(best['n_neurons'][0])
            mlp_eta = best['eta'][0]
            clf_details += '({}, {}), af={}, eta={}, solver={}'.format(mlp_n_neurons, mlp_n_hidden, mlp_af, mlp_eta, mlp_solver)
            
            clf = {'model': clf_type, 'eta': mlp_eta, 'activ': {'af': mlp_af}, 'n_neurons': mlp_n_neurons, 'n_hidden': mlp_n_hidden, 'mlp_solver': mlp_solver}
        
        elif best['clf'][0] == 5:
            clf_type += 'DTree'
            dtree_crit = 'gini' if best['crit'][0]==0 else 'entropy'
            clf_details += 'criterion={}'.format(dtree_crit)
            
            clf = {'model': clf_type, 'crit' : dtree_crit}
        
        ### Compute kappa, cost and confirm acc
        # clf = {'model': clf_type, 'lda_solver': lda_solver, 'metric': knn_metric, 'neig': knn_neig, 'kernel': {'kf': svm_kernel}, 'C': svm_clog, 'crit' : dtree_crit, 
               # 'eta': mlp_eta, 'activ': {'af': mlp_af}, 'n_neurons': mlp_n_neurons, 'n_hidden': mlp_n_hidden, 'mlp_solver': mlp_solver}
        filtering = {'design':filt_type} if filt_type=='DFT' else {'design':filt_type, 'iir_order':filt_order} if filt_type=='IIR' else {'design':filt_type, 'fir_order':filt_order}
        approach = {'option': ap, 'nbands': nbands}
        # print(path_data + str(suj) + '.npy', fl, fh, tmin, tmax, ncsp, approach, filtering, clf)
        bci = BCI(data, events, class_ids, overlap, info['fs'], crossval, nfolds, test_perc, fl, fh, tmin, tmax, ncsp, approach, filtering, clf) 
        st = time()
        bci.evaluate()
        cost = round(time()-st,4)
        acc2 = bci.acc
        kappa = bci.kappa
        
        ### Fixed CSP-LDA-DFT
        bci = BCI(data, events, class_ids, overlap, info['fs'], crossval, nfolds, test_perc, 8, 30, tmin, tmax, 8, 
                  {'option':'classic'}, {'design':'DFT'}, {'model':'LDA', 'lda_solver':'svd'}) 
        st = time()
        bci.evaluate()
        cost_cla_dft = round(time()-st,4)
        acc_cla_dft = bci.acc
        kpa_cla_dft = bci.kappa
        
        ### Fixed CSP-LDA-IIR
        bci = BCI(data, events, class_ids, overlap, info['fs'], crossval, nfolds, test_perc, 8, 30, tmin, tmax, 8, 
                  {'option':'classic'}, {'design':'IIR', 'iir_order':5}, {'model':'LDA', 'lda_solver':'svd'}) 
        st = time()
        bci.evaluate()
        cost_cla_iir = round(time()-st,4)
        acc_cla_iir = bci.acc
        kpa_cla_iir = bci.kappa
        
        ### Fixed SBCSP-DFT
        bci = BCI(data, events, class_ids, overlap, info['fs'], crossval, nfolds, test_perc, 4, 40, tmin, tmax, 8, 
                  {'option':'sbcsp', 'nbands':9}, {'design':'DFT'}, {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4}) 
        st = time()
        bci.evaluate()
        cost_sb_dft = round(time()-st,4)
        acc_sb_dft = bci.acc
        kpa_sb_dft = bci.kappa
        
        ### Fixed SBCSP-IIR
        bci = BCI(data, events, class_ids, overlap, info['fs'], crossval, nfolds, test_perc, 4, 40, tmin, tmax, 8, 
                  {'option':'sbcsp', 'nbands':9}, {'design':'IIR', 'iir_order':5}, {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4}) 
        st = time()
        bci.evaluate()
        cost_sb_iir = round(time()-st,4)
        acc_sb_iir = bci.acc
        kpa_sb_iir = bci.kappa
        
        # df.loc[len(df)] = [suj, class_ids[0], class_ids[1], tmin, tmax, acc, acc2, kappa, cost, fl, fh, filt_type, filt_order, ncsp, approach, nbands, clf_type, clf_details,
        #                    acc_cla_dft, acc_cla_iir, acc_sb_dft, acc_sb_iir, kpa_cla_dft, kpa_cla_iir, kpa_sb_dft, kpa_sb_iir, cost_cla_dft, cost_cla_iir, cost_sb_dft, cost_sb_iir,
        #                    lda_solver , knn_metric, knn_neig, svm_kernel, svm_clog, dtree_crit, mlp_n_hidden, mlp_n_neurons, mlp_eta, mlp_af, mlp_solver]   
        df.loc[len(df)] = [suj, class_ids[0], class_ids[1], tmin, tmax, fl, fh, ncsp, nbands, clf_type, clf, acc, acc_cla_dft, acc_cla_iir, acc_sb_dft, acc_sb_iir] #clf_details
      
pd.to_pickle(df, path_to_trials + 'results_' + ds + '.pkl')
# del globals()['events'] del globals()['data'] del globals()['best'] del globals()['trials'] del globals()['space']
