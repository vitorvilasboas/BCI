# -*- coding: utf-8 -*-
# @author: Vitor Vilas Boas
import os
import pickle
import numpy as np
import pandas as pd
from time import time
from hyperopt import base, fmin, tpe, rand, hp, space_eval
from bci_utils import BCI
import matplotlib.pyplot as plt

bci = BCI()
def objective(args):
    # print(args)
    f_low, f_high, bci.tmin, bci.tmax, ncomp, nbands, bci.clf = args
    bci.ap = {'option': 'sbcsp', 'nbands': nbands}
    bci.f_low, bci.f_high, bci.ncomp = int(f_low), int(f_high), int(ncomp)
    while (bci.tmax-bci.tmin)<1: bci.tmax+=0.5 # garante janela minima de 1seg
    bci.evaluate()
    return bci.acc * (-1)


if __name__ == "__main__":
    ds = 'IV2a'
    n_iter = 100
    path_to_setup = '../as_results/sbrt20/IV2a/'
    data_split = 'as_train' # common, as_train, as_test
    overlap = True
    crossval = False
    nfolds = 5
    test_perc = 0.2 if crossval else 0.5 
    if not os.path.isdir(path_to_setup): os.makedirs(path_to_setup)  
    subjects = range(1,10) 
    classes = [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]] # 
    filtering = {'design':'DFT'}
    # clf = {'model':'SVM','kernel':{'kf':'linear'},'C':-4}
    # fl, fh = 4, 40
    # tmin, tmax = 0.5, 2.5
    # ncomp = 8
    # approach = {'option':'sbcsp', 'nbands':9}
    
    header = ['subj','A','B','tmin','tmax','fl','fh','ncsp','nbands','clf','clf_details','acc_train','acc_test','sb_dft','sb_iir']
    R = pd.DataFrame(columns=header)
    
    ##%% ###########################################################################
    for suj in subjects:
        sname = 'A0' + str(suj) + ''
        data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/'+sname+'.npy', allow_pickle=True)
        for class_ids in classes:
            # data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/'+sname+'T.npy', allow_pickle=True)
            print(f'###### {suj} {class_ids} ######')                          
            space = (
                hp.uniformint('fl', 0, 20),
                hp.uniformint('fh', 21, 50),
                hp.quniform('tmin', 0, 2, 0.5),
                hp.quniform('tmax', 2, 4, 0.5),
                hp.quniform('ncomp', 2, 22, 2),
                hp.uniformint('nbands', 1, 50),
                hp.choice('clf', [
                    {'model':'LDA'},
                    {'model':'KNN','neig': hp.uniformint('neig', 2, 50),
                      'metric':hp.choice('metric', ['euclidean','manhattan','minkowski'])},
                    {'model':'SVM','C': hp.quniform('C', -8, 0, 1), 
                      'kernel':hp.choice('kernel',[{'kf':'linear'},{'kf':'sigmoid'}])}])
            )
             
            bci.data, bci.events, bci.class_ids, bci.fs, bci.overlap = data, events, class_ids, info['fs'], overlap
            bci.crossval, bci.nfolds, bci.test_perc, bci.split = crossval, nfolds, test_perc, data_split
            bci.filt_info = filtering 
            # bci.clf = clf 
            # bci.ap = approach
            # bci.f_low, bci.f_high = fl, fh
            # bci.tmin, bci.tmax = tmin, tmax
            # bci.ncomp = ncomp
            
            path_to_trials = path_to_setup + sname + '_' + str(class_ids[0]) + 'x' + str(class_ids[1]) + '.pkl'
            acc_train = -1
            # for cont in range(10):
            try:
                trials = pickle.load(open(path_to_trials, 'rb'))
                acc_train = ((-1) * trials.best_trial['result']['loss'])
            except:
                trials = base.Trials()
            #     print('No trial file at specified path, creating new one')
            # else: print('Trials file found')

            if acc_train < 1:
                try:
                    print('N trials: ' + str(len(trials)))
                    best = fmin(objective, space=space, algo=tpe.suggest, max_evals=len(trials) + n_iter, trials=trials, verbose=0)
                    pickle.dump(trials, open(path_to_trials, 'wb'))
                except:
                    print('Exception raised')
                    pickle.dump(trials, open(path_to_trials, 'wb'))
                    raise  
            # else: print(suj, class_ids, trials.best_trial['result']['loss'], trials.best_trial['misc']['vals'])
            
            ##%% ###########################################################################
            trials = pickle.load(open(path_to_trials, 'rb'))
            acc_train = (-1) * trials.best_trial['result']['loss']
            best = trials.best_trial['misc']['vals']
            # print(f'Acc: {round(acc_train*100, 2)} %')
            # print('Best:', best,'\n')
            
            fl = int(best['fl'][0])
            fh = int(best['fh'][0])                       
            ncsp = int(best['ncomp'][0])
            tmin = best['tmin'][0]
            tmax = best['tmax'][0]
            nbands = int(best['nbands'][0])
            while (tmax-tmin)<1: tmax+=0.5 # garante janela minima de 1seg
            if nbands > (fh-fl): nbands = (fh-fl)
            approach = {'option': 'sbcsp', 'nbands': nbands}

            if best['clf'][0] == 0: 
                clf_type = 'LDA'
                clf = {'model': clf_type}
            elif best['clf'][0] == 1: 
                clf_type = 'KNN'
                knn_metric = 'euclidean' if best['metric'][0] == 0 else 'manhattan' if best['metric'][0] == 1 else 'minkowski'
                clf = {'model': clf_type, 'metric': knn_metric, 'neig': int(best['neig'][0]), }     
            else: 
                clf_type = 'SVM'
                svm_kernel = 'linear' if best['kernel'][0]==0 else 'sigmoid' 
                clf = {'model': clf_type, 'kernel': {'kf': svm_kernel}, 'C':  int(best['C'][0])}
            
            # data, events, info = np.load('/mnt/dados/eeg_data/IV2a/npy/'+sname+'E.npy', allow_pickle=True)
            
            bci_test = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, 
                           crossval=crossval, nfolds=nfolds, test_perc=test_perc, split='as_test',
                           f_low=fl, f_high=fh, tmin=tmin, tmax=tmax, ncomp=ncsp, ap=approach, filt_info=filtering, clf=clf)
            bci_test.evaluate()
            acc_test = bci_test.acc
            
            ### Fixed SBCSP-DFT
            bci_test = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, 
                           crossval=crossval, nfolds=nfolds, test_perc=test_perc, split='as_test',
                           f_low=4, f_high=40, tmin=0.5, tmax=2.5, ncomp=8, ap={'option':'sbcsp','nbands':9}, 
                           filt_info={'design':'DFT'}, clf={'model':'SVM','kernel':{'kf':'linear'},'C':-4}) 
            bci_test.evaluate()
            sb_dft = bci_test.acc
            
            ### Fixed SBCSP-IIR
            bci_test = BCI(data=data, events=events, class_ids=class_ids, fs=info['fs'], overlap=overlap, 
                           crossval=crossval, nfolds=nfolds, test_perc=test_perc, split='as_test',
                           f_low=4, f_high=40, tmin=0.5, tmax=2.5, ncomp=8, ap={'option':'sbcsp','nbands':9}, 
                           filt_info={'design':'IIR','iir_order':5}, clf={'model':'SVM','kernel':{'kf':'linear'},'C':-4}) 
            bci_test.evaluate()
            sb_iir = bci_test.acc
            
            R.loc[len(R)] = [suj, class_ids[0], class_ids[1], tmin, tmax, fl, fh, ncsp, nbands, clf_type, clf, 
                             acc_train, acc_test, sb_dft, sb_iir]

    print(f">>> AS(tr):{round(R['acc_train'].mean()*100, 2)} | AS(te):{round(R['acc_test'].mean()*100, 2)} | FSBDFT:{round(R['sb_dft'].mean()*100,2)} | FSBIIR:{round(R['sb_iir'].mean()*100,2)} <<<")
    
    ##%% ###########################################################################
    # ### PLOT GRAFIC
    # plt.rcParams.update({'font.size':12})
    # acc_as = R['acc']*100
    # plt.figure(3, facecolor='mintcream')
    # plt.subplots(figsize=(10, 12), facecolor='mintcream')
    # ref = ['sb_dft','sb_iir']
    # for i in range(2):
    #     acc_ref = R[ref[i]]*100
    #     plt.subplot(2, 1, i+1)
    #     plt.scatter(np.asarray(acc_ref).reshape(-1,1), np.asarray(acc_as).reshape(-1,1), facecolors = 'c', marker = 'o', s=50, alpha=.9, edgecolors='firebrick', zorder=3)
    #     plt.scatter(round(acc_ref.mean(),2), round(acc_as.mean(),2), facecolors = 'dodgerblue', marker = 'o', s=100, alpha=1, edgecolors='darkblue', label=r'Acurácia Média', zorder=5)
    #     plt.plot(np.linspace(40, 110, 1000), np.linspace(40, 110, 1000), color='dimgray', linewidth=1, linestyle='--', zorder=0) #linha pontilhada diagonal - limiar 
    #     plt.ylim((48, 102))
    #     plt.xlim((48, 102))
    #     plt.xticks(np.arange(50, 102, 5))
    #     plt.yticks(np.arange(50, 102, 5)) 
    #     plt.plot(np.ones(1000)*round(acc_ref.mean(),2), np.linspace(40, round(acc_as.mean(),2), 1000), color='dimgray', linewidth=.7, linestyle=':', zorder=0) # linha pontilhada verical - acc média auto setup
    #     plt.plot(np.linspace(40, round(acc_ref.mean(),2), 1000), np.ones(1000)*round(acc_as.mean(),2), color='dimgray', linewidth=.7, linestyle=':', zorder=0) # linha pontilhada horizontal - acc média ref
    #     plt.xlabel('Acurácia ' + ('SBCSP DFT' if i==0 else 'SBCSP IIR' ) + ' (configuração única) (%)', fontsize=12)
    #     plt.ylabel('Acurácia Auto Setup (%)', fontsize=12)
    #     plt.legend(loc='lower right', fontsize=12)
    # # plt.savefig('../as_results/sbrt20/IV2a/scatter_y.png', format='png', dpi=300, transparent=True, bbox_inches='tight')
        
    ##%% ###########################################################################
    # ### SAVE RESULTS PICKLE FILE
    # pd.to_pickle(R, path_to_setup + 'RESULTS.pkl')           
    # R = pd.read_pickle("../as_results/sbrt20/IV2a/RESULTS.pkl")
    # print(round(R['acc'].mean()*100, 2), round(R['sb_iir'].mean()*100,2))