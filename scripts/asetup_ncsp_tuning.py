# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import pickle
import numpy as np
import pandas as pd
from hyperopt import base, fmin, tpe, hp
from bci_utils import BCI

bci = BCI()
def objective(args):
    bci.csp_list = list(map(lambda x: int(x), args))
    bci.evaluate()
    return bci.acc * (-1)
        
if __name__ == "__main__":
    ds = 'IV2a' # III3a, III4a, IV2a, IV2b, LINCE, Lee19
    scenario = ''
    n_iter = 10
    
    path_to_setup = '../as_results/ncsp_tuning/' + ds + scenario + '/' # PATH TO AUTO SETUP RESULTS AND TRIALS
                
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
        
    R = pd.read_pickle('../as_results/sbrt20/' + ds + scenario + '/RESULTS.pkl')  
    # print(ds, R['acc'].mean(), R['acc'].median(), R['acc'].std(), R['acc'].max(), R['acc'].min())
    
    # subjects = [8, 9] # uncomment to run one subject only
    # classes = [[1, 2]] # uncomment to run LH x RH classification only
    
    ###########################################################################
    df = []
    for suj in subjects:
        sname = prefix + str(suj) + suffix
        path_to_data = '/mnt/dados/eeg_data/' + ds + '/npy/' + sname + '.npy' # PATH TO DATASET 
        data, events, info = np.load(path_to_data, allow_pickle=True) # pickle.load(open(path_to_data, 'rb'))
        
        if ds=='LINCE' and suj == 'CL_LF': classes = [[1, 3]]
        if ds=='Lee19' and cortex_only:
            cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
            data = data[cortex]   
            info['eeg_channels'] = len(cortex)
            info['ch_labels'] = ['FC5','FC3','FC1','FC2','FC4','FC6','C5','C3','C1','Cz','C2','C4','C6','CP5','CP3','CP1','CPz','CP2','CP4','CP6']
        
        for class_ids in classes: # [[1,2]] 
            args = R.loc[(R['subj'] == suj) & (R['A'] == class_ids[0]) & (R['B'] == class_ids[1])]
            args.index = [0]
            # print(args['clf_details'])
            
            if int(args['nbands'])==1:
                df.append([suj, class_ids[0], class_ids[1], int(args['ncsp']), int(args['nbands']), args['acc'][0], args['acc'][0], 
                            args['tmin'][0], args['tmax'][0], args['fl'][0], args['fh'][0], args['clf'][0], args['clf_details'][0]])

            else:
                desvio = 4 # desvio em torno do ncsp ótimo (deve ser par)
                min_ncsp = (int(args['ncsp']) - desvio) if (int(args['ncsp']) - desvio) > 2 else 2
                max_ncsp = (int(args['ncsp']) + desvio) if (int(args['ncsp']) + desvio) < info['eeg_channels'] else info['eeg_channels']
    
                space = tuple([ hp.quniform('csp'+str(i), min_ncsp, max_ncsp, 2) for i in range(int(args['nbands'])) ])
                
                approach = {'option':'sbcsp', 'nbands':int(args['nbands'])}
                filtering = {'design':'DFT'}
               
                bci.data, bci.events, bci.class_ids, bci.fs = data, events, class_ids, info['fs']
                bci.overlap, bci.crossval, bci.nfolds, bci.test_perc = overlap, crossval, nfolds, test_perc
                bci.f_low, bci.f_high, bci.tmin, bci.tmax, bci.ncomp = int(args['fl']), int(args['fh']), float(args['tmin']), float(args['tmax']), int(args['ncsp'])
                bci.ap, bci.filt_info, bci.clf = approach, filtering, args['clf_details'][0]
                
                path_to_trials = path_to_setup + sname + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
                
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
                    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=1)
                    pickle.dump(trials, open(path_to_trials, 'wb'))
                    # print(suj, class_ids, best)
                except:
                    print('Exception raised')
                    pickle.dump(trials, open(path_to_trials, 'wb'))
                    # print('\n', suj, class_ids, trials.best_trial['misc']['vals'])
                    raise
                
                acc = (-1) * trials.best_trial['result']['loss']
                # print(suj, class_ids, str(round(acc*100,2))+'%')
                
                best = trials.best_trial['misc']['vals']
    
                df.append(np.r_[[suj, class_ids[0], class_ids[1], int(args['ncsp']), int(args['nbands']), args['acc'][0], acc, 
                                  args['tmin'][0], args['tmax'][0], args['fl'][0], args['fh'][0], args['clf'][0], args['clf_details'][0]], 
                                [ int(best['csp'+str(i)][0]) for i in range(int(args['nbands']))]])
    
    header = ['subj', 'A', 'B', 'ncsp', 'nbands', 'acc', 'acc_tune', 'tmin', 'tmax', 'fl', 'fh', 'clf', 'clf_details']#, 
              # 'n1', 'n2', 'n3', 'n4', 'n5', 'n6', 'n7', 'n8', 'n9', 'n10', 'n11', 'n12', 'n13', 'n14', 'n15', 'n16', 'n17', 
              # 'n18', 'n19', 'n20', 'n21', 'n22', 'n23', 'n24', 'n25', 'n26', 'n27', 'n28', 'n29', 'n30', 'n31', 'n32', 
              # 'n33', 'n34', 'n35', 'n36', 'n37', 'n38', 'n39', 'n40', 'n41', 'n42', 'n43', 'n44', 'n45', 'n46', 'n47']
    
    header_tune = [('n'+str(i+1)) for i in range( max(list(map(lambda l: len(l), df))) - len(header) )]
    
    FINAL = pd.DataFrame(df, columns=header+header_tune) 
        
    pd.to_pickle(FINAL, path_to_setup + 'RESULTS.pkl')
    FINAL.to_csv(path_to_setup + 'RESULTS.csv', index=False)
    # del globals()['events'] del globals()['data'] del globals()['best'] del globals()['trials'] del globals()['space']
    
    
    # sbrt = pd.read_pickle("../as_results/sbrt20/IV2a/RESULTS.pkl")
    # tuned = pd.read_pickle('../as_results/ncsp_tuning/IV2a/RESULTS.pkl')
    # tuned = pd.read_pickle(path_to_setup + 'RESULTS.pkl')