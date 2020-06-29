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
    print(args)
    if bci.ap['option'] == 'classic': bci.ncomp = args
    else:
        bci.ncomp, bci.clf['C'] = args 
        # bci.f_low, bci.f_high, bci.ncomp, bci.ap['nbands'], bci.clf['C'] = args # free space
    bci.evaluate()
    return bci.acc * (-1)

if __name__ == "__main__": 
    ds = 'IV2a' # III3a, III4a, IV2a, IV2b, LINCE, Lee19
    crossval = False
    n_iter = 80
    fl, fh, tmin, tmax = 8, 30, 0.5, 2.5  # fl,fh=None to free space       
    # approach = {'option':'classic'}
    approach = {'option':'sbcsp', 'nbands':9} # nbands=None to free space
    
    filtering = {'design':'DFT'}
    # filtering = {'design':'IIR', 'iir_order': 5}
    
    overlap = True
    nfolds = 10 
    test_perc = 0.1 if crossval else 0.5
    cortex_only = True # used when ds == Lee19 - True to used only cortex channels
    
    clf = {'model':'LDA', 'lda_solver':'svd'} if approach['option']=='classic' else {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':None}
    
    if fl == None: scenario = approach['option'] + '_free'
    else: scenario = approach['option'] + '_' + str(fl) + '-' + str(fh) + 'Hz' + \
        (('_' + str(approach['nbands']) + 'sb') if approach['option']=='sbcsp' else '')
    scenario += ('_' + filtering['design'])
    
    path_to_setup = '../as_results/dft_cost/' + ds + '_' + scenario + '/' # PATH TO AUTO SETUP RESULTS AND TRIALS
     
    # if not os.path.isdir(path_to_setup): os.makedirs(path_to_setup)

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
        
    # subjects = ['aa'] # uncomment to run one subject only
    # classes = [[1, 2]] # uncomment to run LH x RH classification only
    
######### AUTO SETUP ##########################################################
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
            bci.data, bci.events, bci.class_ids, bci.fs = data, events, class_ids, info['fs']
            bci.overlap, bci.crossval, bci.nfolds, bci.test_perc = overlap, crossval, nfolds, test_perc
            bci.f_low, bci.f_high, bci.tmin, bci.tmax, bci.ap, bci.filt_info, bci.clf = fl, fh, tmin, tmax, approach, filtering, clf 
            
            if approach['option'] == 'classic':
                space = (hp.quniform('ncomp', 2, info['eeg_channels'], 2))
            else:
                space = (
                    # hp.uniformint('fl', 0, 20), # free space
                    # hp.uniformint('fh', 30, 49), # free space
                    hp.quniform('ncomp', 2, info['eeg_channels'], 2),
                    # hp.uniformint('nbands', 2, 50), # free space
                    hp.quniform('svm_clog', -8, 0, 1)
                    )
                
            path_to_trials = path_to_setup + sname + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
            
            trials = base.Trials()
            try:
                # print('Trying to pickle file')
                trials = pickle.load(open(path_to_trials, 'rb'))
            except:
                print('No trial file at specified path, creating new one')
                trials = base.Trials()
            else: print('File found')
            
            try:
                print('Size of object: ' + str(len(trials)))
                best = fmin(objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=0)
                pickle.dump(trials, open(path_to_trials, 'wb'))
                # print(suj, class_ids, best)
            except:
                print('Exception raised')
                pickle.dump(trials, open(path_to_trials, 'wb'))
                raise
    
######### COMPILE RESULTS #####################################################
    R = pd.DataFrame(columns=['subj','A','B','fl','fh','nbands','ncsp','clog','acc','kpa','cost'])        
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
            acc0 = (-1) * trials.best_trial['result']['loss']
            best = trials.best_trial['misc']['vals']
            print(path_to_trials, acc0)
            
            ncsp = best['ncomp'][0]
            if clf['model'] == 'SVM': clf['C'] = best['svm_clog'][0]
            
            if scenario == (approach['option'] + '_free' + '_' + filtering['design']):
                if approach['option'] == 'sbcsp': approach['nbands'] = best['nbands'][0] # free space
                fl, fh = best['fl'][0], best['fh'][0] # free space
            
            cost = []
            for i in range(5):
                bci1 = BCI(data, events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc, 
                           f_low=fl, f_high=fh, tmin=tmin, tmax=tmax, ncomp=ncsp, ap=approach, filt_info=filtering, clf=clf)
                st = time()
                bci1.evaluate()
                cost.append(round(time()-st,4))
                acc, kpa = bci1.acc, bci1.kappa
            
            nbands = approach['nbands'] if approach['option'] == 'sbcsp' else None
            clog = clf['C'] if approach['option'] == 'sbcsp' else None
            
            R.loc[len(R)] = [suj,class_ids[0],class_ids[1],fl,fh,nbands,ncsp,clog,acc,kpa,np.mean(cost)]

    pd.to_pickle(R, '../as_results/dft_cost/RES_' + ds + '_' + scenario + '.pkl')
    print('Mean:', R['acc'].mean(), R['cost'].mean(), R['kpa'].mean()) 
    
########### ANALYSIS #########################################################

    ds = 'III4a' # III4a, IV2a
    scenario = '_classic_8-30Hz' # '_classic_8-30Hz' or '_sbcsp_8-30Hz_9sb' or '_sbcsp_0-50Hz_9sb' or '_sbcsp_0-50Hz_24sb' or '_sbcsp_free'
    DFT = pd.read_pickle('../as_results/dft_cost/RES_' + ds + scenario + '_DFT.pkl')
    IIR = pd.read_pickle('../as_results/dft_cost/RES_' + ds + scenario + '_IIR.pkl')
    
    print('Mean DFT:', round(DFT['acc'].mean()*100, 2), round(DFT['kpa'].mean(),4), round(DFT['cost'].mean(),4))
    print('Mean IIR:', round(IIR['acc'].mean()*100, 2), round(IIR['kpa'].mean(),4), round(IIR['cost'].mean(),4))
    print('>> ' + ds + scenario)
    print(f"Custo DFT: {round(((IIR['cost'].mean() - DFT['cost'].mean()) / IIR['cost'].mean())*100,2)}%")
    print(f"Ganho Acc DFT: {round((DFT['acc'].mean() - IIR['acc'].mean())*100,2)}%")
    
    print('================ DFT ================')
    print(DFT.iloc[:,8:].describe())
    print('================ IIR ================')
    print(IIR.iloc[:,8:].describe())
    
    