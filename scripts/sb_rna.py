#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 16:25:04 2020
@author: vboas
"""
import mne
import warnings
import numpy as np
from time import time
from sklearn.svm import SVC
from scipy.io import loadmat
from scipy.stats import norm
from sklearn.preprocessing import normalize
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from bci_utils import nanCleaner, extractEpochs, Filter, CSP

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)


if __name__ == "__main__":
    suj = 1
    class_ids = [1,2]
    path_gdf = '/mnt/dados/eeg_data/IV2a/A0'
    path_trues = '/mnt/dados/eeg_data/IV2a/true_labels/A0'
    tmin, tmax, fl, fh, ncsp, nbands = 0.5, 2.5, 4, 40, 8, 9
    filtering = {'design':'DFT'} 
    # filtering = {'design':'IIR', 'iir_order':5}

    #######
    
    Fs = 250
    smin, smax = int(tmin*Fs), int(tmax*Fs)
    
    ### to gdf file
    eeg = mne.io.read_raw_gdf(path_gdf + str(suj) + 'T.gdf').load_data()
    data1 = eeg.get_data()[:22] # [channels x samples]
    events1 = mne.events_from_annotations(eeg) # raw.find_edf_events()
    ch_names = eeg.ch_names
    
    eeg = mne.io.read_raw_gdf(path_gdf + str(suj) + 'E.gdf').load_data()
    data2 = eeg.get_data()[:22] # [channels x samples]
    events2 = mne.events_from_annotations(eeg) # raw.find_edf_events()
    
    # for k,v in events1[1].items(): print(f'{k}:: {v}')
    # for k,v in events2[1].items(): print(f'{k}:: {v}')
    lb_utils = [4,5,6,7]
    
    events1 = np.delete(events1[0], 1, axis=1)
    events2 = np.delete(events2[0], 1, axis=1)
    
    ZT, tt = extractEpochs(data1, events1, smin, smax, lb_utils)
    for i,k in zip(lb_utils, range(1, 5)): tt = np.where(tt == i, k, tt)
    ZT = np.vstack([ ZT[np.where(tt == k)] for k in class_ids ])
    tt = np.hstack([ np.ones(len(ZT)//2)*k for k in class_ids ]).astype(int) 
    
    ZV, _ = extractEpochs(data2, events2, smin, smax, [4])
    tv = np.ravel(loadmat(path_trues + str(suj) + 'E.mat' )['classlabel'])
    ZV = np.vstack([ ZV[np.where(tv == k)] for k in class_ids ])
    tv = np.hstack([ np.ones(len(ZV)//2)*k for k in class_ids ]).astype(int) 

    # ### to npy file
    # # data1, events1, _  = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'T.npy', allow_pickle=True) # sessão 1
    # # data2, events2, _  = np.load('/mnt/dados/eeg_data/IV2a/npy/A0' + str(suj) + 'E.npy', allow_pickle=True) # sessão 2
    # # ZT, tt = extractEpochs(data1, events1, smin, smax, class_ids)
    # # ZV, tv = extractEpochs(data2, events2, smin, smax, class_ids)
    
    # t0 = time()
    # step = (fh-fl) / (nbands+1) # n_bins/nbands+1
    # size = step / 0.5 # step/overlap
    # sub_bands = []
    # for i in range(nbands):
    #     fl_sb = i * step + fl
    #     fh_sb = i * step + size + fl
    #     sub_bands.append([fl_sb, fh_sb])
    
    # XT, XV = [], []
    # if filtering['design'] == 'DFT':
    #     filt = Filter(fl, fh, Fs, filtering)
    #     XTF = filt.apply_filter(ZT)
    #     XVF = filt.apply_filter(ZV)
    #     for i in range(nbands):
    #         bsize = 2/(Fs/ZT.shape[-1]) # 2 == sen/cos
    #         XT.append(XTF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])
    #         bsize = 2/(Fs/ZV.shape[-1])
    #         XV.append(XVF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])         
    # elif filtering['design'] == 'IIR':
    #     for i in range(nbands):
    #         filt = Filter(sub_bands[i][0], sub_bands[i][1], Fs, filtering)
    #         XT.append(filt.apply_filter(ZT))
    #         XV.append(filt.apply_filter(ZV))
            
    # csp = [ CSP(n_components=ncsp) for i in range(nbands) ] # mne.decoding.CSP()
    # for i in range(nbands): csp[i].fit(XT[i], tt)
    # FT = [ csp[i].transform(XT[i]) for i in range(nbands) ]
    # FV = [ csp[i].transform(XV[i]) for i in range(nbands) ]
    
    # ldas = [ LDA() for i in range(nbands) ]
    # for i in range(nbands): ldas[i].fit(FT[i], tt)
    # ST = np.asarray([ np.ravel(ldas[i].transform(FT[i])) for i in range(nbands)]).T # Score LDA
    # SV = np.asarray([ np.ravel(ldas[i].transform(FV[i])) for i in range(nbands)]).T 
        
    # p0 = norm(np.mean(ST[tt == class_ids[0], :], axis=0), np.std(ST[tt == class_ids[0], :], axis=0))
    # p1 = norm(np.mean(ST[tt == class_ids[1], :], axis=0), np.std(ST[tt == class_ids[1], :], axis=0))
    # META_ST = np.log(p0.pdf(ST) / p1.pdf(ST))
    # META_SV = np.log(p0.pdf(SV) / p1.pdf(SV))
    
    # svm = SVC(kernel='linear', C=1e-4, probability=True)
    # svm.fit(META_ST, tt)
    # y, yp = svm.predict(META_SV), svm.predict_proba(META_SV); 
    # acc_svm = round(np.mean(y == tv)*100,2) # round(svm.score(META_SV, tv)*100,2)
    # print('Off SVM acc:', acc_svm)
    
    # lda = LDA()
    # lda.fit(META_ST, tt)
    # y, yp = lda.predict(META_SV), lda.predict_proba(META_SV)
    # acc_lda = round(np.mean(y == tv)*100,2) # round(lda.score(META_SV, tv)*100,2)
    # print('Off LDA acc:', acc_lda)
    
    # TT, TV = np.zeros((len(ZT), 2)), np.zeros((len(ZV), 2))
    # for i in range(2): TT[:,i] = np.where(tt == i+1, 1, TT[:,i])
    # for i in range(2): TV[:,i] = np.where(tv == i+1, 1, TV[:,i])
    
    # mlp = MLPClassifier(hidden_layer_sizes=(100,2), max_iter=10000, activation='tanh', verbose=False) #, random_state=42)
    # mlp.out_activation = 'softmax' # 'logistic', 'softmax', # mlp.outputs = 3
    # mlp.fit(META_ST, TT)
    # Y, YP = mlp.predict(META_SV), mlp.predict_proba(META_SV)
    # y = np.argmax(YP, axis=1)+1
    # acc_mlp = round(np.mean(y == tv)*100,2) # round(mlp.score(META_SV, TV)*100,2)
    # print('Off MLP acc:', acc_mlp)

    # print('runtime:', round(time()-t0,3), '\n')
    
    # ### =============================================================================
    # ### FULL TRIAL (3 classes)
    # ### =============================================================================

    # ####################################
    # # trialsT, labelsT = extractEpochs(data1, events1, int(0.5*Fs), int(2.5*Fs), lb_utils)
    # # for i,k in zip(lb_utils, range(1, 5)): labelsT = np.where(labelsT == i, k, labelsT)
    # # trialsT = [ trialsT[np.where(labelsT == k)] for k in class_ids ]
    # # ZTa, ZTb = trialsT[0], trialsT[1] 
    
    # # trialsT, labelsT = extractEpochs(data1, events1, int(-2*Fs), 0, lb_utils)
    # # for i,k in zip(lb_utils, range(1, 5)): labelsT = np.where(labelsT == i, k, labelsT)
    # # ZT0 = np.vstack([ trialsT[np.where(labelsT == k)] for k in class_ids ])
    # # ZT0 = np.vstack([ZT0[:36], ZT0[108:]])
    # # ZT = np.vstack([ZT0, ZTa, ZTb])
    # # tt = np.hstack([np.ones(len(ZT)//3)*k for k in [0,1,2]])
    
    # # #######
    
    # # trialsV, _ = extractEpochs(data2, events2, int(0.5*Fs), int(2.5*Fs), [4])
    # # labelsV = np.ravel(loadmat(path_trues + str(suj) + 'E.mat' )['classlabel'])
    # # trialsV = [ trialsV[np.where(labelsV == k)] for k in class_ids ]
    # # ZVa, ZVb = trialsV[0], trialsV[1] 
    
    # # trialsV, _ = extractEpochs(data2, events2, int(0.5*Fs), int(2.5*Fs), [4])
    # # labelsV = np.ravel(loadmat(path_trues + str(suj) + 'E.mat' )['classlabel'])
    # # ZV0 = np.vstack([ trialsV[np.where(labelsV == k)] for k in class_ids ])
    # # ZV0 = np.vstack([ZV0[:36], ZV0[108:]])
    # # ZV = np.vstack([ZV0, ZVa, ZVb])
    # # tv = np.hstack([np.ones(len(ZV)//3)*k for k in [0,1,2]])
    # ####################################
    
    # smin, smax = int(-2*Fs), int(5*Fs)
    
    # ### to gdf file
    # trialsT, labelsT = extractEpochs(data1, events1, smin, smax, lb_utils)
    # for i,k in zip(lb_utils, range(1, 5)): labelsT = np.where(labelsT == i, k, labelsT)
    # trialsT = np.vstack([ trialsT[np.where(labelsT == k)] for k in class_ids ])
    # labelsT = np.hstack([ np.ones(len(trialsT)//2)*k for k in class_ids ]).astype(int) 
    
    # trialsV, _ = extractEpochs(data2, events2, smin, smax, [4])
    # labelsV = np.ravel(loadmat(path_trues + str(suj) + 'E.mat' )['classlabel'])
    # trialsV = np.vstack([ trialsV[np.where(labelsV == k)] for k in class_ids ])
    # labelsV = np.hstack([ np.ones(len(trialsV)//2)*k for k in class_ids ]).astype(int) 
    
    # ### to npy file
    # # trialsT, labelsT = extractEpochs(data1, events1, smin, smax, class_ids)
    # # trialsV, labelsV = extractEpochs(data2, events2, smin, smax, class_ids)
    
    # delta_t = 0.2; delta_s = int(delta_t * Fs)
    # q = int((tmax*Fs)-(tmin*Fs))  # q = largura em amostras
    # ZT, tt, ZV, tv = [], [], [], []
    # for k in range(len(trialsT)): # len(trials_cal)
    #     n = q   # n=localização em amostras (fim)
    #     m = q/2 # localização em amostra do ponto médio da época
    #     inc = (0.0 * q)  # para ser considerado comando, no min 70% da amostra (m+10%) deve estar contida no periodo de MI
    #     while n <= trialsT.shape[-1]:
    #         ZT.append(trialsT[k, :, n-q:n])
    #         ZV.append(trialsV[k, :, n-q:n])
    #         if (m <= (500+inc)) or (m >= (1500-inc)): tt.append(0); tv.append(0); 
    #         else: tt.append(labelsT[k]); tv.append(labelsV[k])
    #         m += delta_s
    #         n += delta_s
    # ZT, ZV = np.asarray(ZT), np.asarray(ZV)
    # tt, tv = np.asarray(tt), np.asarray(tv)
    
    # t0 = time()
    # step = (fh-fl) / (nbands+1) # n_bins/setup['nbands']+1
    # size = step / 0.5 # step/overlap
    # sub_bands = []
    # for i in range(nbands):
    #     fl_sb = i * step + fl
    #     fh_sb = i * step + size + fl
    #     sub_bands.append([fl_sb, fh_sb])
        
    # XT, XV = [], []
    # if filtering['design'] == 'DFT':
    #     filt = Filter(fl, fh, Fs, filtering)
    #     XTF = filt.apply_filter(ZT)
    #     XVF = filt.apply_filter(ZV)
    #     for i in range(nbands):
    #         bsize = 2 / ( Fs / ZT.shape[-1] ) # 2 representa sen e cos
    #         XT.append(XTF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])
    #         bsize = 2 / ( Fs / ZV.shape[-1] )
    #         XV.append(XVF[:, :, round(sub_bands[i][0]*bsize):round(sub_bands[i][1]*bsize)])         
    # elif filtering['design'] == 'IIR':
    #     for i in range(nbands):
    #         filt = Filter(sub_bands[i][0], sub_bands[i][1], Fs, filtering)
    #         XT.append(filt.apply_filter(ZT))
    #         XV.append(filt.apply_filter(ZV))
            
    # csp = [ CSP(n_components=ncsp) for i in range(nbands) ] # mne.decoding.CSP
    # for i in range(nbands): csp[i].fit(XT[i], tt)
    # FT = [ csp[i].transform(XT[i]) for i in range(nbands) ]
    # FV = [ csp[i].transform(XV[i]) for i in range(nbands) ]
    
    # ldas = [ LDA() for i in range(nbands) ]
    # for i in range(nbands): ldas[i].fit(FT[i], tt)
    # ST = np.asarray([ np.ravel(ldas[i].predict(FT[i])) for i in range(nbands) ]).T # Score LDA 
    # SV = np.asarray([ np.ravel(ldas[i].predict(FV[i])) for i in range(nbands) ]).T # transform=2cls; predict=3cls
    
    # FT = np.vstack(np.transpose(FT, (0,2,1))).T    
    # FV = np.vstack(np.transpose(FV, (0,2,1))).T 
    # # FT = normalize(FT, norm='l2')
    # # FV = normalize(FV, norm='l2')
    
    # svm = SVC(kernel='linear', C=1e-4, probability=True, decision_function_shape='ovo')
    # svm.fit(FT, tt)
    # y, yp = svm.predict(FV), svm.predict_proba(FV); 
    # acc_svm_on = round(np.mean(y == tv)*100,2) # round(svm.score(FV, tv)*100,2)
    # print('On SVM acc:', acc_svm_on)
    
    # lda = LDA()
    # lda.fit(FT, tt)
    # y, yp = lda.predict(FV), lda.predict_proba(FV)
    # acc_lda_on = round(np.mean(y == tv)*100,2) # round(lda.score(FV, tv)*100,2)
    # print('On LDA acc:', acc_lda_on)
     
    # TT, TV = np.zeros((len(ZT), 3)), np.zeros((len(ZV), 3))
    # for i in range(3): TT[:,i] = np.where(tt == i, 1, TT[:,i])
    # for i in range(3): TV[:,i] = np.where(tv == i, 1, TV[:,i])
    
    # mlp = MLPClassifier(hidden_layer_sizes=(100,2), max_iter=1000, activation='tanh', verbose=False) #, random_state=42)
    # mlp.out_activation = 'softmax' # 'logistic', 'softmax', # mlp.outputs = 3
    # mlp.fit(FT, TT)
    # Y, YP = mlp.predict(FV), mlp.predict_proba(FV)
    # y = np.argmax(YP, axis=1)
    # acc_mlp_on = round(np.mean(y == tv)*100,2) # round(mlp.score(FV, TV)*100,2) 
    # print('On MLP acc:', acc_mlp_on) 
    # # print(confusion_matrix(tv, y))
    
    # print('runtime:', round(time()-t0,3))