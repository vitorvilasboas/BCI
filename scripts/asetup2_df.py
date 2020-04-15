# -*- coding: utf-8 -*-
""" @author: Vitor Vilas-Boas """
import os
import pickle
import numpy as np
import pandas as pd
from time import time
from bci_utils import BCI 

ds = 'IV2a' # III3a, III4a, IV2a, IV2b, Lee19, CL, TW
path = '/mnt/dados/eeg_data/' + ds + '/npy/' + ('S' if ds=='Lee19' else 'A0' if ds=='IV2a' else 'B0' if ds=='IV2b' else '')
## >>> ENTER THE PATH TO THE DATASET HERE

fs = 250 if ds=='Lee19' else 100 if ds=='III4a' else 125 if ds=='CL' else 250
subjects = range(1,55) if ds=='Lee19' else ['aa','al','av','aw','ay'] if ds=='III4a' else ['K3','K6','L1'] if ds=='III4a' else range(1,10)
classes = [[1,3]] if ds=='III4a' else [[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]] if ds in ['IV2a','III3a'] else [[1,2]]

if ds=='Lee19':
    cortex_only = True
    one_session = True
    lee_session = 2
    lee_option = ('_s' + str(lee_session) + '_cortex') if one_session and cortex_only else '_cortex' if cortex_only else ''

overlap = True
crossval = False
nfolds = 10
test_perc = 0.1 if crossval else 0.5

# header = ['subj', 'A', 'B', 'tmin', 'tmax', 'acc', 'acc2', 'kappa', 'cost', 'fl', 'fh', 
#           'f_type', 'forder', 'ncsp', 'approach', 'nbands', 'clf', 'clf details',
#           'acc_cla_dft', 'acc_cla_iir', 'acc_sb_dft', 'acc_sb_iir', 
#           'kpa_cla_dft', 'kpa_cla_iir', 'kpa_sb_dft', 'kpa_sb_iir', 
#           'cost_cla_dft', 'cost_cla_iir', 'cost_sb_dft', 'cost_sb_iir', 
#           'lda_solver', 'knn_metric', 'knn_neig', 'svm_kernel', 'svm_clog', 'dtree_crit', 
#           'mlp_n_hidden', 'mlp_n_neurons', 'mlp_eta', 'mlp_af', 'mlp_solver']
header = ['subj', 'A', 'B', 'tmin', 'tmax', 'fl', 'fh', 'ncsp', 'nbands', 'clf', 'clf_details', 'acc', 'cla_dft', 'cla_iir', 'sb_dft', 'sb_iir']
df = pd.DataFrame(columns=header)

path_to_results = './asetup_results/'
if not os.path.isdir(path_to_results): os.makedirs(path_to_results)

for class_ids in classes:
    for suj, i in zip(subjects, range(len(subjects))):
        path_to_trials = './asetup_trials/' + ds + ((lee_option + '/') if ds=='Lee19' else '/') + ds + '_' + str(suj) + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
        trials = pickle.load(open(path_to_trials, 'rb'))
        # print(path_to_trials)
        
        acc = trials.best_trial['result']['loss'] * -1
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
        
        elif best['clf'][0] == 1: 
            clf_type += 'LDA'
            lda_solver = 'svd' if best['lda_solver'][0] == 0 else 'lsqr' if best['lda_solver'][0] == 1 else 'eigen'
            clf_details += "solver={}".format(lda_solver)
        
        elif best['clf'][0] == 2: 
            clf_type += 'KNN'
            knn_metric = 'euclidean' if best['metric'][0] == 0 else 'manhattan' if best['metric'][0] == 1 else 'minkowski' if best['metric'][0] == 2 else 'chebyshev'
            knn_neig = int(best['neig'][0])
            clf_details += 'neig={}, metric={}'.format(knn_neig, knn_metric)
               
        elif best['clf'][0] == 3: 
            clf_type += 'SVM'
            svm_kernel = 'linear' if best['kernel'][0]==0 else 'poly' if best['kernel'][0]==1 else 'sigmoid' if best['kernel'][0]==2  else 'rbf'
            svm_clog = int(best['C'][0])
            clf_details += 'k={}, C=10**({})'.format(svm_kernel, svm_clog)
        
        elif best['clf'][0] == 4:
            clf_type += 'MLP'
            mlp_af = 'identity' if best['activ'][0]==0 else 'logistic' if best['activ'][0]==1 else 'tanh' if best['activ'][0]==2  else 'relu'
            mlp_solver = 'adam' if best['mlp_solver'][0] == 0 else 'lbfgs' if best['mlp_solver'][0] == 1 else 'sgd'
            mlp_n_hidden = int(best['n_hidden'][0])
            mlp_n_neurons = int(best['n_neurons'][0])
            mlp_eta = best['eta'][0]
            clf_details += '({}, {}), af={}, eta={}, solver={}'.format(mlp_n_neurons, mlp_n_hidden, mlp_af, mlp_eta, mlp_solver)
        
        elif best['clf'][0] == 5:
            clf_type += 'DTree'
            dtree_crit = 'gini' if best['crit'][0]==0 else 'entropy'
            clf_details += 'criterion={}'.format(dtree_crit)
        
        # print(path + str(suj) + '.npy')
        if ds=='Lee19' and one_session: path_to_data = path + str(suj) + '_sess' + str(lee_session) + '.npy' # '.omi' 
        else: path_to_data = path + str(suj) + '.npy'
        
        data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))
        
        if ds=='Lee19' and cortex_only:
            cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
            data = data[cortex]   
            info['eeg_channels'] = len(cortex)
            info['ch_labels'] = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
        
        
        ### Compute kappa, cost and confirm acc
        clf = {'model': clf_type, 'lda_solver': lda_solver, 'metric': knn_metric, 'neig': knn_neig, 'kernel': {'kf': svm_kernel}, 'C': svm_clog, 'crit' : dtree_crit, 
               'eta': mlp_eta, 'activ': {'af': mlp_af}, 'n_neurons': mlp_n_neurons, 'n_hidden': mlp_n_hidden, 'mlp_solver': mlp_solver}
        filtering = {'design':filt_type} if filt_type=='DFT' else {'design':filt_type, 'iir_order':filt_order} if filt_type=='IIR' else {'design':filt_type, 'fir_order':filt_order}
        approach = {'option': ap, 'nbands': nbands}
        # print(path_data + str(suj) + '.npy', fl, fh, tmin, tmax, ncsp, approach, filtering, clf)
        bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, fl, fh, tmin, tmax, ncsp, approach, filtering, clf) 
        st = time()
        bci.evaluate()
        cost = round(time()-st,4)
        acc2 = bci.acc
        kappa = bci.kappa
        
        ### Fixed CSP-LDA-DFT
        bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, 8, 30, tmin, tmax, 8, 
                  {'option':'classic'}, {'design':'DFT'}, {'model':'LDA', 'lda_solver':'svd'}) 
        st = time()
        bci.evaluate()
        cost_cla_dft = round(time()-st,4)
        acc_cla_dft = bci.acc
        kpa_cla_dft = bci.kappa
        
        ### Fixed CSP-LDA-IIR
        bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, 8, 30, tmin, tmax, 8, 
                  {'option':'classic'}, {'design':'IIR', 'iir_order':5}, {'model':'LDA', 'lda_solver':'svd'}) 
        st = time()
        bci.evaluate()
        cost_cla_iir = round(time()-st,4)
        acc_cla_iir = bci.acc
        kpa_cla_iir = bci.kappa
        
        ### Fixed SBCSP-DFT
        bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, 4, 40, tmin, tmax, 8, 
                  {'option':'sbcsp', 'nbands':9}, {'design':'DFT'}, {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4}) 
        st = time()
        bci.evaluate()
        cost_sb_dft = round(time()-st,4)
        acc_sb_dft = bci.acc
        kpa_sb_dft = bci.kappa
        
        ### Fixed SBCSP-IIR
        bci = BCI(data, events, class_ids, overlap, fs, crossval, nfolds, test_perc, 4, 40, tmin, tmax, 8, 
                  {'option':'sbcsp', 'nbands':9}, {'design':'IIR', 'iir_order':5}, {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':-4}) 
        st = time()
        bci.evaluate()
        cost_sb_iir = round(time()-st,4)
        acc_sb_iir = bci.acc
        kpa_sb_iir = bci.kappa
        
        # df.loc[len(df)] = [suj, class_ids[0], class_ids[1], tmin, tmax, acc, acc2, kappa, cost, fl, fh, 
        #                       filt_type, filt_order, ncsp, approach, nbands, clf_type, clf_details,
        #                       acc_cla_dft, acc_cla_iir, acc_sb_dft, acc_sb_iir, 
        #                       kpa_cla_dft, kpa_cla_iir, kpa_sb_dft, kpa_sb_iir, 
        #                       cost_cla_dft, cost_cla_iir, cost_sb_dft, cost_sb_iir,
        #                       lda_solver , knn_metric, knn_neig, svm_kernel, svm_clog, dtree_crit, 
        #                       mlp_n_hidden, mlp_n_neurons, mlp_eta, mlp_af, mlp_solver]   
        df.loc[len(df)] = [suj, class_ids[0], class_ids[1], tmin, tmax, fl, fh, ncsp, nbands, clf_type, clf_details, acc, acc_cla_dft, acc_cla_iir, acc_sb_dft, acc_sb_iir]
        

pd.to_pickle(df, path_to_results + '/RES_' + ds + ('.pkl' if ds!='Lee19' else (lee_option+'.pkl')))
