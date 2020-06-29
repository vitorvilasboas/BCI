#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 14:06:44 2020
@author: vboas
"""
import mne
import os
import pickle
import warnings
import numpy as np
from time import time
from datetime import datetime
from scipy.signal import decimate, resample
from scipy.io import loadmat
from bci_utils import CSP
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter, decimate, welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def extractEpochs(data, events, smin, smax, class_ids):
    cond = False
    for i in range(len(class_ids)): cond += (events[:, 1] == class_ids[i]) 
    idx = np.where(cond)[0]
    s0 = events[idx, 0] + smin
    sn = events[idx, 0] + smax
    labels = events[idx, 1]
    epochs = np.asarray([ data[:, s0[i]:sn[i]] for i in range(len(s0)) ])
    return epochs, labels

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

factor = 10
cortex = [7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 40]
mi_start, mi_end = 0, 4 # -3, 7

for suj in range(1,55):
    
    Fs = 1000
    suj_in = str(suj) if suj >= 10 else ('0' + str(suj))
    #### Session 1: Load and segment 
    S = loadmat('/mnt/dados/eeg_data/Lee19/session1/sess01_subj' + suj_in + '_EEG_MI.mat')
    
    data = (S['EEG_MI_train']['x'][0,0].T)
    events = np.r_[ S['EEG_MI_train']['t'][0,0], S['EEG_MI_train']['y_dec'][0,0] ].T
    events[:, 1] = np.where(events[:, 1] == 2, 1, 2) # troca class_ids 1=LH, 2=RH
    ZT1, tt1 = extractEpochs(data[cortex], events, int(Fs*mi_start), int(Fs*mi_end), [1,2])
    
    data = (S['EEG_MI_test']['x'][0,0].T)
    events = np.r_[ S['EEG_MI_test']['t'][0,0],  S['EEG_MI_test']['y_dec'][0,0]  ].T
    events[:, 1] = np.where(events[:, 1] == 2, 1, 2) # troca class_ids 1=LH, 2=RH
    ZV1, tv1 = extractEpochs(data[cortex], events, int(Fs*mi_start), int(Fs*mi_end), [1,2])

    #### Session 2: Load and segment 
    S = loadmat('/mnt/dados/eeg_data/Lee19/session2/sess02_subj' + suj_in + '_EEG_MI.mat')
    
    data = (S['EEG_MI_train']['x'][0,0].T)
    events = np.r_[ S['EEG_MI_train']['t'][0,0], S['EEG_MI_train']['y_dec'][0,0] ].T
    events[:, 1] = np.where(events[:, 1] == 2, 1, 2) # troca class_ids 1=LH, 2=RH
    ZT2, tt2 = extractEpochs(data[cortex], events, int(Fs*mi_start), int(Fs*mi_end), [1,2])
    
    data = (S['EEG_MI_test']['x'][0,0].T)
    events = np.r_[ S['EEG_MI_test']['t'][0,0],  S['EEG_MI_test']['y_dec'][0,0]  ].T
    events[:, 1] = np.where(events[:, 1] == 2, 1, 2) # troca class_ids 1=LH, 2=RH
    ZV2, tv2 = extractEpochs(data[cortex], events, int(Fs*mi_start), int(Fs*mi_end), [1,2])
    
    ### Downsampling 
    Fs = Fs/factor
    ZT1, ZV1 = resample(ZT1, ZT1.shape[-1]//factor, axis=-1), resample(ZV1, ZV1.shape[-1]//factor, axis=-1) # decimate(ZT1, factor), decimate(ZV1, factor) # 
    ZT2, ZV2 = resample(ZT2, ZT2.shape[-1]//factor, axis=-1), resample(ZV2, ZV2.shape[-1]//factor, axis=-1) # decimate(ZT2, factor), decimate(ZV2, factor)
    
    np.save('/mnt/dados/eeg_data/Lee19/npy_epochs/S' + str(suj), [ZT1, ZV1, ZT2, ZV2, tt1, tv1, tt2, tv2, {'fs':Fs}])
    
    del (globals()['S'], globals()['data'], globals()['events'], globals()['suj_in'])
    del (globals()['ZT1'], globals()['ZV1'], globals()['ZT2'], globals()['ZV2'], globals()['tt1'], globals()['tv1'], globals()['tt2'], globals()['tv2'])
   
    ##########

    class_ids = [1, 2]
    tmin, tmax = 1, 3.5
    fl, fh, ncsp = 8, 30, 4
    
    ZT1, ZV1, ZT2, ZV2, tt1, tv1, tt2, tv2, info = np.load('/mnt/dados/eeg_data/Lee19/npy_epochs/S' + str(suj) + '.npy', allow_pickle=True)
    Fs = info['fs']
    
    smin, smax = int(Fs*tmin), int(Fs*tmax)
    ZT1 = ZT1[:,:,smin:smax]
    ZV1 = ZV1[:,:,smin:smax]
    ZT2 = ZT2[:,:,smin:smax]
    ZV2 = ZV2[:,:,smin:smax]
    
    ### Combine Sessions 1 e 2 
    ZT, tt = np.r_[ZT1, ZT2], np.r_[tt1, tt2]
    ZV, tv = np.r_[ZV1, ZV2], np.r_[tv1, tv2]
    
    # ZT, tt = np.copy(ZT1), np.copy(tt1)
    # ZV, tv = np.copy(ZV1), np.copy(tv1)
    
    del (globals()['ZT1'], globals()['ZV1'], globals()['ZT2'], globals()['ZV2'], globals()['tt1'], globals()['tv1'], globals()['tt2'], globals()['tv2']) 
    
    #### BCI 
    st = time()
    
    b, a = butter(5, [fl/(Fs/2),fh/(Fs/2)], btype='bandpass')
    XT = lfilter(b, a, ZT)
    XV = lfilter(b, a, ZV)
    del (globals()['a'], globals()['b'])
    
    csp = mne.decoding.CSP(n_components=int(ncsp), reg=None) # csp = CSP(n_components=int(ncsp))
    
    # W = csp.filters_[:int(ncsp)]
    # XT = np.asarray([ np.dot(W, ep) for ep in XT ])
    # XV = np.asarray([ np.dot(W, ep) for ep in XV ])
    # FT = np.log(np.var(XT, axis=2)) # np.log(np.mean(ET**2, axis=2))
    # FV = np.log(np.var(XV, axis=2)) # np.log(np.mean(EV**2, axis=2))
    
    csp.fit(XT, tt)
    FT = csp.transform(XT) 
    
    lda = LDA()
    lda.fit(FT, tt)
    
    FV = csp.transform(XV)
    y = lda.predict(FV) # lda.predict_proba(XV_CSP)
    acc = lda.score(FV, tv) # np.mean(y == tv)
    
    print(acc*100)




