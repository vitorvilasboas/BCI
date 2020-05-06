# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import pickle
import numpy as np
import pandas as pd
from time import time
from hyperopt import base, tpe, fmin, hp
from scripts.bci_utils import BCI

bci = BCI()
def objective(args):
    # print(args)
    if bci.ap['option'] == 'classic': bci.ncomp = args
    else:
        # bci.ncomp, bci.clf['C'] = args # option 1
        # bci.ncomp, bci.ap['nbands'], bci.clf['C'] = args # option 2
        bci.f_low, bci.f_high, bci.ncomp, bci.ap['nbands'], bci.clf['C'] = args # option 3
    bci.evaluate()
    return bci.acc * (-1)

if __name__ == "__main__": 
    ds = 'IV2a' # III3a, III4a, IV2a, IV2b, LINCE, Lee19
    scenario = 'sbcsp_free' # 'classic_8-30Hz' or 'sbcsp_8-30Hz_9sb' or 'sbcsp_0-50Hz_9sb' or 'sbcsp_0-50Hz_24sb' or 'sbcsp_free'
    n_iter = 300
    
    fl, fh, tmin, tmax = None,None, 0.5, 2.5  # fl,fh=None to option 3
            
    # approach = {'option':'classic'}
    approach = {'option':'sbcsp', 'nbands':None} # nbands=None to option 2 ou 3
    
    filtering = {'design':'DFT'}
    # filtering = {'design':'IIR', 'iir_order': 5}
    
    # clf = {'model':'LDA', 'lda_solver':'svd'} 
    clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':None}
    
    path_to_setup = '../as_results/dft_cost/' + ds + '_' + scenario + '/' # PATH TO AUTO SETUP RESULTS AND TRIALS
     
    overlap = True
    crossval = False
    nfolds = 10 
    test_perc = 0.1 if crossval else 0.5
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
        
    # subjects = [1] # uncomment to run one subject only
    # classes = [[1, 2]] # uncomment to run LH x RH classification only
    
    ###########################################################################
    # for suj in subjects:
    #     sname = prefix + str(suj) + suffix
    #     path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + sname + '.npy' # PATH TO DATASET  
    #     data, events, info = np.load(path_to_data, allow_pickle=True)

    #     if ds=='LINCE' and suj == 'CL_LF': classes = [[1, 3]]
    #     if ds=='Lee19' and cortex_only:
    #         cortex = [7,32,8,9,33,10,34,12,35,13,36,14,37,17,38,18,39,19,40,20]
    #         data = data[cortex]   
    #         info['eeg_channels'] = len(cortex)
    #         info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
        
    #     for class_ids in classes:
    #         bci.data, bci.events, bci.class_ids, bci.fs = data, events, class_ids, info['fs']
    #         bci.overlap, bci.crossval, bci.nfolds, bci.test_perc = overlap, crossval, nfolds, test_perc
    #         bci.f_low, bci.f_high, bci.tmin, bci.tmax, bci.ap, bci.filt_info, bci.clf = fl, fh, tmin, tmax, approach, filtering, clf 
            
    #         if approach['option'] == 'classic':
    #             space = (hp.quniform('ncomp', 2, info['eeg_channels'], 2))
    #         else:
    #             space = (
    #                 hp.uniformint('fl', 0, 20), # option 3
    #                 hp.uniformint('fh', 30, 49), # option 3
    #                 hp.quniform('ncomp', 2, info['eeg_channels'], 2),
    #                 hp.uniformint('nbands', 2, 50), # option 2 ou 3
    #                 hp.quniform('svm_clog', -8, 0, 1)
    #                 )
                
    #         path_to_trials = path_to_setup + sname + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
            
    #         trials = base.Trials()
    #         try:
    #             # print('Trying to pickle file')
    #             trials = pickle.load(open(path_to_trials, 'rb'))
    #         except:
    #             print('No trial file at specified path, creating new one')
    #             trials = base.Trials()
    #         else: print('File found')
            
    #         try:
    #             print('Size of object: ' + str(len(trials)))
    #             best = fmin(objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=0)
    #             pickle.dump(trials, open(path_to_trials, 'wb'))
    #             # print(suj, class_ids, best)
    #         except:
    #             print('Exception raised')
    #             pickle.dump(trials, open(path_to_trials, 'wb'))
    #             raise
    
    
    ###########################################################################
    R = pd.DataFrame(columns=['subj','A','B','fl','fh','tmin','tmax','nbands','ncsp','clog','acc_dft','acc_iir','cost_dft','cost_iir','kpa_dft','kpa_iir'])        
    for suj in subjects:
        sname = prefix + str(suj) + suffix
        path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + sname + '.npy' # PATH TO DATASET  
        data, events, info = np.load(path_to_data, allow_pickle=True)
        
        if ds=='LINCE' and suj == 'CL_LF': classes = [[1, 3]]
        if ds=='Lee19' and cortex_only:
            cortex = [7,32,8,9,33,10,34,12,35,13,36,14,37,17,38,18,39,19,40,20]
            data = data[cortex]   
            info['eeg_channels'] = len(cortex)
            info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
        
        for class_ids in classes:
            path_to_trials = path_to_setup + sname + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
            trials = pickle.load(open(path_to_trials, 'rb'))
            acc = (-1) * trials.best_trial['result']['loss']
            best = trials.best_trial['misc']['vals']
            
            ncsp = best['ncomp'][0]
            if clf['model'] == 'SVM': clf['C'] = best['svm_clog'][0]
            if approach['option'] == 'sbcsp': approach['nbands'] = best['nbands'][0] # option 2 ou 3
            fl, fh = best['fl'][0], best['fh'][0] # option 3
            
            cost_dft, cost_iir = [], []
            for i in range(10):
                bci_dft = BCI(data, events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc, 
                              f_low=fl, f_high=fh, tmin=tmin, tmax=tmax, ncomp=ncsp, ap=approach, filt_info={'design':'DFT'}, clf=clf)
                st = time()
                bci_dft.evaluate()
                cost_dft.append(round(time()-st,4))
                acc_dft, kappa_dft = bci_dft.acc, bci_dft.kappa
            
                bci_iir = BCI(data, events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc, 
                              f_low=fl, f_high=fh, tmin=tmin, tmax=tmax, ncomp=ncsp, ap=approach, filt_info={'design':'IIR', 'iir_order': 5}, clf=clf)
                st = time()
                bci_iir.evaluate()
                cost_iir.append(round(time()-st,4))
                acc_iir, kappa_iir = bci_iir.acc, bci_iir.kappa
            
            R.loc[len(R)] = [suj, class_ids[0], class_ids[1], fl, fh, tmin, tmax, approach['nbands'] if approach['option'] == 'sbcsp' else None, ncsp, 
                              clf['C'] if approach['option'] == 'sbcsp' else None, acc_dft, acc_iir, np.mean(cost_dft), np.mean(cost_iir), kappa_dft, kappa_iir]

    pd.to_pickle(R, path_to_setup + 'RESULTS.pkl')
    print('Mean:', R['acc_dft'].mean(), R['acc_iir'].mean()) 
    
RL = pd.read_pickle(path_to_setup + 'RESULTS.pkl')
print(scenario)
print(RL.iloc[:,10:].describe())