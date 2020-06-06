# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import pickle
import numpy as np
import pandas as pd
from time import time
from hyperopt import base, fmin, tpe, hp
from bci_utils import BCI

bci = BCI()
def objective(args):
    # print(args)
    f_low, f_high, bci.tmin, bci.tmax, ncomp, bci.ap, bci.filt_info, bci.clf = args
    bci.f_low, bci.f_high, bci.ncomp = int(f_low), int(f_high), int(ncomp)
    while (bci.tmax-bci.tmin)<1: bci.tmax+=0.5 # garante janela minima de 1seg
    bci.evaluate()
    return bci.acc * (-1)


if __name__ == "__main__":
    ds = 'IV2a' # III3a, III4a, IV2a, IV2b, LINCE, Lee19
    n_iter = 100
    scenario = '' # '_s1_cortex' or '_s2_cortex'
    
    # fl, fh, tmin, tmax = 0, 50, 0.5, 2.5  # fl,fh=None to option 3      
    # approach = {'option':'classic'}
    # approach = {'option':'sbcsp', 'nbands':24} # nbands=None to option 2 ou 3
    # filtering = {'design':'DFT'}
    # filtering = {'design':'IIR', 'iir_order': 5}
    # clf = {'model':'LDA', 'lda_solver':'svd'} 
    # clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':None}
    
    path_to_setup = '../as_results/sbrt20/' + ds + '' + scenario + '/' # PATH TO AUTO SETUP RESULTS AND TRIALS
    
    overlap = True
    crossval = False
    nfolds = 10
    test_perc = 0.2 if crossval else 0.5 
    cortex_only = True # used when ds == Lee19 - True to used only cortex channels 
    
    if not os.path.isdir(path_to_setup): os.makedirs(path_to_setup)
    
    prefix, suffix = '', ''
    if ds == 'III3a':
        subjects = ['K3','K6','L1'] 
        classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]]
    
    elif ds == 'III4a':
        subjects = ['aa','al','av','aw','ay']
        classes = [[1, 3]]
    
    elif ds == 'IV2a':        
        subjects = range(1,10) 
        classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] # 
        prefix = 'A0'
        suffix = '' # '', 'T', 'E'
    
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
    
        
    ###################
    # import numpy as np
    # idx = np.arange(0, 144, 1, dtype=int)
    # idx_teste = np.random.choice(144, size=44, replace=False)
    # idx = np.asarray(list(set(idx).difference(idx_teste)))
    # idx_treino = np.random.choice(idx, size=50, replace=False)
    # idx_valida = np.asarray(list(set(idx).difference(idx_treino)))
    # idx_teste.sort()
    # idx_treino.sort()
    # idx_valida.sort()
    ####################

    
    # subjects = [1] # uncomment to run one subject only
    # classes = [[1, 2]] # uncomment to run LH x RH classification only
    
#%% ###########################################################################
    
    # for suj in subjects:
    #     sname = prefix + str(suj) + suffix
    #     path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + sname  + '.npy' # PATH TO DATASET  
    #     data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))
        
    #     if ds=='LINCE' and suj == 'CL_LF': classes = [[1, 3]]
    #     if ds=='Lee19' and cortex_only:
    #         cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
    #         data = data[cortex]   
    #         info['eeg_channels'] = len(cortex)
    #         info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
        
    #     for class_ids in classes:       
    #         # max_knn_neig = int((info['trials_per_class'] * 2) * test_perc)
    #         max_knn_neig = int((100 * 2) * test_perc)
    #         # print(trials_per_class, max_knn_neig)
                                
    #         space = (
    #             hp.uniformint('fl', 0, 20), # hp.quniform('fl', 1, 20, 1),
    #             hp.uniformint('fh', 30, 49),  # hp.quniform('fh', 30, 49, 1),
    #             hp.quniform('tmin', 0, 2, 0.5),
    #             hp.quniform('tmax', 2, info['trial_mi_time'], 0.5),
    #             hp.quniform('ncomp', 2, info['eeg_channels'], 2),
    #             hp.choice('approach', [
    #                 {'option':'classic',},
    #                 {'option':'sbcsp', 'nbands': hp.uniformint('nbands', 2, 48)} # hp.quniform('nbands', 2, max_nbands, 1)}
    #                 ]),
    #             hp.choice('filt', [
    #                 {'design':'DFT'},
    #                 # {'design':'IIR', 'iir_order': hp.uniformint('iir_order', 1, 8)}, #hp.quniform('iir_order', 1, 8, 1)},
    #                 # {'design':'FIR', 'fir_order': hp.uniformint('fir_order', 1, 8)}, #hp.quniform('fir_order', 1, 8, 1)}
    #                 ]),
    #             hp.choice('clf', [
    #                 {'model':'Bayes'},
    #                 {'model':'LDA',
    #                   'lda_solver': hp.choice('lda_solver', ['svd','lsqr','eigen']),
    #                   # 'shrinkage': hp.choice('shrinkage', [None, 'auto', {'shrinkage_float':  hp.uniform('shrinkage_float', 0, 1)}]) #np.logspace(-4, 0, 1)
    #                   },
    #                 {'model':'KNN', 
    #                   'neig': hp.uniformint('neig', 2, max_knn_neig), # hp.quniform('neig', 2, trials_per_class, 1)
    #                   'metric': hp.choice('metric', ['euclidean','manhattan','minkowski','chebyshev']), #{'mf':'cityblock'}, {'mf':'cosine'}, {'mf':'l1'}, {'mf':'l2'},
    #                   # 'p': hp.quniform('p', 2, 50, 1)
    #                   },
    #                 {'model':'SVM', 
    #                   'C': hp.quniform('C', -8, 0, 1), 
    #                   'kernel': hp.choice('kernel', [{'kf':'linear'}, {'kf':'poly'}, {'kf':'sigmoid'}, {'kf':'rbf'}]), #'degree': hp.uniformint('degree', 2, 4)    #, 'width': hp.lognormal('width', 0, 1)
    #                   # 'gamma': hp.choice('gamma', ['scale', 'auto', {'gamma_float': hp.quniform('gamma_float', -9, 4, 1)}]), # hp.loguniform('gamma_float', -9, 3)  np.logspace(-9, 3, 13)),
    #                   },
    #                 {'model':'MLP', 
    #                   'eta': hp.quniform('eta', -5, -2, 1), 
    #                   'n_neurons' : hp.quniform('n_neurons', 50, 500, 50), # hp.uniformint('n_neurons', 50, 500),
    #                   'n_hidden': hp.uniformint('n_hidden', 1, 2), # hp.quniform('n_hidden', 1, 4, 1),
    #                   'activ': hp.choice('activ', [{'af':'identity'},{'af':'logistic'},{'af':'tanh'},{'af':'relu'}]),
    #                   'mlp_solver': hp.choice('mlp_solver', ['adam', 'lbfgs', 'sgd'])
    #                   # 'alpha': hp.quniform('alpha', -8, 1, 1),
    #                   # 'eta_type': hp.choice('eta_type', ['constant', 'invscaling', 'adaptive']),,
    #                   },
    #                 {'model':'DTree', 
    #                   'crit': hp.choice('crit', ['gini', 'entropy']),
    #                   # 'max_depth': hp.choice('max_depth', [None, {'max_depth_int': hp.qlognormal('max_depth_int', 3, 1, 1)}]), # np.random.lognormal(3, 1, 1) ]),
    #                   # 'min_split': hp.uniform('min_split', 0.0001, 1), #  np.random.lognormal(2, 1, 1) # hp.qlognormal('min_split', 2, 1, 1)
    #                   }
    #                 ])
    #             )
              
    #         bci.data, bci.events, bci.class_ids, bci.fs = data, events, class_ids, info['fs']
    #         bci.overlap, bci.crossval, bci.nfolds, bci.test_perc = overlap, crossval, nfolds, test_perc
    #         # bci.f_low, bci.f_high, bci.tmin, bci.tmax, bci.ap, bci.filt_info, bci.clf = fl, fh, tmin, tmax, approach, filtering, clf        
    #         path_to_trials = path_to_setup + sname + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
            
    #         trials = base.Trials()
    #         found = False
    #         try:
    #             print('Trying to pickle file')
    #             trials = pickle.load(open(path_to_trials, 'rb'))
    #             found = True
    #         except:
    #             print('No trial file at specified path, creating new one')
    #             trials = base.Trials()
    #         else:
    #             print('File found')
            
            
    #         acc = ((-1) * trials.best_trial['result']['loss']) if found else -1
    #         if acc < 1:
    #             try:
    #                 print('Size of object: ' + str(len(trials)))
    #                 best = fmin(objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=1)
    #                 pickle.dump(trials, open(path_to_trials, 'wb'))
    #                 print(suj, class_ids, best)
    #             except:
    #                 print('Exception raised')
    #                 pickle.dump(trials, open(path_to_trials, 'wb'))
    #                 print('\n', suj, class_ids, trials.best_trial['misc']['vals'])
    #                 raise
    #         else: print('\n', suj, class_ids, acc, trials.best_trial['misc']['vals'])
                
        
#%% ###########################################################################
    # # header = ['subj', 'A', 'B', 'tmin', 'tmax', 'acc', 'acc2', 'kappa', 'cost', 'fl', 'fh', 'f_type', 'forder', 'ncsp', 'approach', 'nbands', 'clf', 'clf details', 'acc_cla_dft', 
    # #           'acc_cla_iir', 'acc_sb_dft', 'acc_sb_iir', 'kpa_cla_dft', 'kpa_cla_iir', 'kpa_sb_dft', 'kpa_sb_iir', 'cost_cla_dft', 'cost_cla_iir', 'cost_sb_dft', 'cost_sb_iir', 
    # #           'lda_solver', 'knn_metric', 'knn_neig', 'svm_kernel', 'svm_clog', 'dtree_crit', 'mlp_n_hidden', 'mlp_n_neurons', 'mlp_eta', 'mlp_af', 'mlp_solver']

    header = ['subj', 'A', 'B', 'tmin', 'tmax', 'fl', 'fh', 'ncsp', 'nbands', 'clf', 'clf_details', 'acc', 'cla_dft', 'cla_iir', 'sb_dft', 'sb_iir']
    df = pd.DataFrame(columns=header)
    
    for suj in subjects:
        sname = prefix + str(suj) + suffix
        path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + sname  + '.npy' # PATH TO DATASET  
        data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))
        
        if ds=='LINCE' and suj == 'CL_LF': classes = [[1, 3]]
        if ds=='Lee19' and cortex_only:
            cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
            data = data[cortex]   
            info['eeg_channels'] = len(cortex)
            info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
        
        for class_ids in classes:
            path_to_trials = path_to_setup + sname + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
            trials = pickle.load(open(path_to_trials, 'rb'))
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
            # while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
            # if nbands > (fh-fl): nbands = (fh-fl)
                      
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
            bci0 = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc,
                        f_low=fl, f_high=fh, tmin=tmin, tmax=tmax, ncomp=ncsp, ap=approach, filt_info=filtering, clf=clf)
            st = time()
            bci0.evaluate()
            cost = round(time()-st,4)
            acc2 = bci0.acc
            kappa = bci0.kappa
            
            print(round(acc*100, 1), round(acc2*100, 1))
            
            ### Fixed CSP-LDA-DFT
            bci0 = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc,
                        f_low=8, f_high=30, tmin=tmin, tmax=tmax, ncomp=8, 
                        ap={'option':'classic'}, filt_info={'design':'DFT'}, clf={'model':'LDA', 'lda_solver':'svd'}) 
            st = time()
            bci0.evaluate()
            cost_cla_dft = round(time()-st,4)
            acc_cla_dft = bci0.acc
            kpa_cla_dft = bci0.kappa
            
            ### Fixed CSP-LDA-IIR
            bci0 = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc, 
                        f_low=8, f_high=30, tmin=tmin, tmax=tmax, ncomp=8, 
                        ap={'option':'classic'}, filt_info={'design':'IIR', 'iir_order':5}, clf={'model':'LDA', 'lda_solver':'svd'}) 
            st = time()
            bci0.evaluate()
            cost_cla_iir = round(time()-st,4)
            acc_cla_iir = bci0.acc
            kpa_cla_iir = bci0.kappa
            
            ### Fixed SBCSP-DFT
            bci0 = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc,
                        f_low=4, f_high=40, tmin=tmin, tmax=tmax, ncomp=8, 
                        ap={'option':'sbcsp', 'nbands':9}, filt_info={'design':'DFT'}, clf={'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4}) 
            st = time()
            bci0.evaluate()
            cost_sb_dft = round(time()-st,4)
            acc_sb_dft = bci0.acc
            kpa_sb_dft = bci0.kappa
            
            ### Fixed SBCSP-IIR
            bci0 = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc, 
                        f_low=4, f_high=40, tmin=tmin, tmax=tmax, ncomp=8, 
                        ap={'option':'sbcsp', 'nbands':9}, filt_info={'design':'IIR', 'iir_order':5}, clf={'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4}) 
            st = time()
            bci0.evaluate()
            cost_sb_iir = round(time()-st,4)
            acc_sb_iir = bci0.acc
            kpa_sb_iir = bci0.kappa
            
            # df.loc[len(df)] = [suj, class_ids[0], class_ids[1], tmin, tmax, acc, acc2, kappa, cost, fl, fh, filt_type, filt_order, ncsp, approach, nbands, clf_type, clf_details,
            #                    acc_cla_dft, acc_cla_iir, acc_sb_dft, acc_sb_iir, kpa_cla_dft, kpa_cla_iir, kpa_sb_dft, kpa_sb_iir, cost_cla_dft, cost_cla_iir, cost_sb_dft, cost_sb_iir,
            #                    lda_solver , knn_metric, knn_neig, svm_kernel, svm_clog, dtree_crit, mlp_n_hidden, mlp_n_neurons, mlp_eta, mlp_af, mlp_solver]   
            df.loc[len(df)] = [suj, class_ids[0], class_ids[1], tmin, tmax, fl, fh, ncsp, nbands, clf_type, clf, acc2, acc_cla_dft, acc_cla_iir, acc_sb_dft, acc_sb_iir] #clf_details
          
    pd.to_pickle(df, path_to_setup + 'RESULTS.pkl')
                 
    # del globals()['events'] del globals()['data'] del globals()['best'] del globals()['trials'] del globals()['space']
    
    sbrt = pd.read_pickle("../as_results/sbrt20/IV2a/RESULTS.pkl")
    sbrt_old = pd.read_pickle("../as_results/sbrt20/IV2a_old/RESULTS.pkl")
    # tuned = pd.read_pickle('../as_results/ncsp_tuning/IV2a/RESULTS.pkl')
    
    print(round(sbrt['acc'].mean()*100, 2), round(sbrt['acc'].std()*100, 3),'\n',
          round(sbrt['sb_iir'].mean()*100,2), round(sbrt['sb_iir'].std()*100, 3), '\n', 
          round(sbrt['cla_iir'].mean()*100,2), round(sbrt['cla_iir'].std()*100, 3))
    
    # print(sbrt['acc'].mean(), sbrt['sb_iir'].mean(), sbrt['cla_iir'].mean(), '\n',
    #       sbrt_old['acc'].mean(), sbrt_old['sb_iir'].mean(), sbrt_old['cla_iir'].mean())

