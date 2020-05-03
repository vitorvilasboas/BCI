# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import numpy as np
import pandas as pd
from time import time
from hyperopt import base, tpe, fmin, hp
from scripts.bci_utils import BCI

bci = BCI()
def objective(args):
    if bci.ap['option'] == 'classic': 
        bci.ncomp = args
        bci.clf = {'model':'LDA', 'lda_solver':'svd'}
    else: 
        bci.ncomp, nbands, svm_clog = args
        bci.clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':svm_clog}
        bci.ap = {'option':'sbcsp', 'nbands':nbands}
    while (bci.tmax - bci.tmin) < 1: bci.tmax += 0.5 # garante janela minima de 1seg
    bci.evaluate()
    # print(args)
    return bci.acc * (-1)

if __name__ == "__main__": 
    ds = 'IV2a' # III3a, III4a, IV2a, IV2b
    n_iter = 10    
    
    overlap = True
    crossval = False
    nfolds = 10 
    test_perc = 0.1 if crossval else 0.5
    
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
    
    R = pd.DataFrame(columns=['subj', 'A', 'B', 'fl', 'fh', 'tmin', 'tmax', 'ncsp', 'nbands', 'clog', 
                              'acc_dft', 'acc_iir', 'cost_dft', 'cost_iir', 'kpa_dft', 'kpa_iir'])

    subjects = [1] # uncomment to run one subject only
    # classes = [[1, 2]] 
    for suj in subjects:
        # path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + '' + 'S' + str(suj) + 'sess2' + '.npy' #> ENTER THE PATH TO DATASET HERE 
        path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + '' + 'A0' + str(suj)  + '.npy' #> ENTER THE PATH TO DATASET HERE  
        data, events, info = np.load(path_to_data, allow_pickle=True)
        
        if ds=='Lee19' and cortex_only:
            cortex = [7,32,8,9,33,10,34,12,35,13,36,14,37,17,38,18,39,19,40,20]
            data = data[cortex]   
            info['eeg_channels'] = len(cortex)
            info['ch_labels'] = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
        
        for class_ids in classes:
            fl, fh, tmin, tmax = 0, 40, 0.5, 2.5
            
            # approach = {'option':'classic'}
            approach = {'option':'sbcsp', 'nbands':None}
            
            filtering = {'design':'DFT'}
            # filtering = {'design':'IIR', 'iir_order': 5}
            
            # clf = {'model':'LDA', 'lda_solver':'svd'} 
            clf = {'model':'SVM', 'kernel':{'kf':'linear'}, 'C':None}

            bci.data, bci.events, bci.class_ids, bci.fs = data, events, class_ids, info['fs']
            bci.overlap, bci.crossval, bci.nfolds, bci.test_perc = overlap, crossval, nfolds, test_perc
            bci.f_low, bci.f_high, bci.tmin, bci.tmax = fl, fh, tmin, tmax 
            bci.ap, bci.filt_info, bci.clf = approach, filtering, clf
            
            if approach['option'] == 'classic':
                space = (hp.quniform('ncomp', 2, info['eeg_channels'], 2))
            else:
                space = (
                    hp.quniform('ncomp', 2, info['eeg_channels'], 2),
                    hp.uniformint('nbands', 2, 40), 
                    hp.quniform('svm_clog', -8, 0, 1)
                    )
            trials = base.Trials()
            try:
                # print('Size of object: ' + str(len(trials)))
                best = fmin(objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=0)
                # print(suj, class_ids, best)
                # pickle.dump(trials, open(path_out, 'wb'))
            except:
                print('Exception raised')
                raise
            
            cost_dft, cost_iir = [], []
            for i in range(10):
                if clf['model'] == 'SVM': clf['C'] = best['svm_clog']
                if approach['option'] == 'sbcsp': approach['nbands'] = best['nbands']

                bci_dft = BCI(data, events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc, 
                              f_low=fl, f_high=fh, tmin=tmin, tmax=tmax, ncomp=best['ncomp'], ap=approach, filt_info={'design':'DFT'}, clf=clf)
                st = time()
                bci_dft.evaluate()
                cost_dft.append(round(time()-st,4))
                acc_dft, kappa_dft = bci_dft.acc, bci_dft.kappa
            
                bci_iir = BCI(data, events, class_ids=class_ids, fs=info['fs'], overlap=overlap, crossval=crossval, nfolds=nfolds, test_perc=test_perc, 
                              f_low=fl, f_high=fh, tmin=tmin, tmax=tmax, ncomp=best['ncomp'], ap=approach, filt_info={'design':'IIR', 'iir_order': 5}, clf=clf)
                st = time()
                bci_iir.evaluate()
                cost_iir.append(round(time()-st,4))
                acc_iir, kappa_iir = bci_iir.acc, bci_iir.kappa
            
            R.loc[len(R)] = [suj, class_ids[0], class_ids[1], fl, fh, tmin, tmax, best['ncomp'], best['nbands'] if approach['option'] == 'sbcsp' else 1, 
                             best['svm_clog'] if approach['option'] == 'sbcsp' else None, acc_dft, acc_iir, np.mean(cost_dft), np.mean(cost_iir), kappa_dft, kappa_iir]
    
    pd.to_pickle(R, '/home/vboas/Desktop/dft_sbcsp/R_' + ds + '_' + approach['option'] + '_' + str(fl) + '-' + str(fh) + 'Hz.pkl')        
    print('Mean:', R['acc_dft'].mean(), R['acc_iir'].mean())   