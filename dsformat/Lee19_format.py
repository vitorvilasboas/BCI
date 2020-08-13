# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:27:46 2020
@author: Vitor Vilas-Boas
    'EEG_MI_train' and 'EEG_MI_test': training and test data
     'x':       continuous EEG signals (data points × channels)
     't':       stimulus onset times of each trial
     'fs':      sampling rates
     'y_dec':   class labels in integer types 
     'y_logic': class labels in logical types
     'y_class': class definitions
     'chan':    channel information

     Protocol: Tcue=3s, Tpause=7s, mi_time=4s, min_pause_time=6s, endTrial=13~14.5s
     100 trials (50 LH + 50 RH) * 2 phases (train, test) * 2 sessions = 400 trials (200 LH + 200 RH)
     62 channels
     54 subjects
     fs = 1000 Hz
"""

import os
import pickle
import numpy as np
from datetime import datetime
from scipy.signal import decimate, resample
from scipy.io import loadmat

path = '/mnt/dados/eeg_data/Lee19/' ## >>> ENTER THE PATH TO THE DATASET HERE
downsampling = True
factor = 10
ch_grid = 'openbci_16' # openbci_16, openbci_8, m1 

path_out = path + 'npy_m1/' #npy
if not os.path.isdir(path_out): os.makedirs(path_out)

for suj in range(1, 55):
    Fs = 1000
    suj_in = str(suj) if suj >= 10 else ('0' + str(suj))
    S1 = loadmat(path + 'session1/sess01_subj' + suj_in + '_EEG_MI.mat') ## >>> ENTER THE SESSION 1 PATH
    S2 = loadmat(path + 'session2/sess02_subj' + suj_in + '_EEG_MI.mat') ## >>> ENTER THE SESSION 2 PATH
    T1 = S1['EEG_MI_train']
    V1 = S1['EEG_MI_test']
    T2 = S2['EEG_MI_train']
    V2 = S2['EEG_MI_test']
    del (globals()['S1'], globals()['S2'])
    dataT1 = T1['x'][0,0].T
    dataV1 = V1['x'][0,0].T
    dataT2 = T2['x'][0,0].T
    dataV2 = V2['x'][0,0].T
    eventsT1 = np.r_[T1['t'][0,0], T1['y_dec'][0,0]].T
    eventsV1 = np.r_[V1['t'][0,0], V1['y_dec'][0,0]].T
    eventsT2 = np.r_[T2['t'][0,0], T2['y_dec'][0,0]].T
    eventsV2 = np.r_[V2['t'][0,0], V2['y_dec'][0,0]].T
    del (globals()['T1'], globals()['T2'], globals()['V1'], globals()['V2'])
    
    # eventsV1[:,0] += dataT1.shape[-1]
    # eventsV2[:,0] += dataT2.shape[-1]
    # e1 = np.r_[eventsT1, eventsV1]
    # e2 = np.r_[eventsT2, eventsV2]
    # d1 = np.c_[dataT1, dataV1]
    # d2 = np.c_[dataT2, dataV2]
    
    eventsT2[:,0] += dataT1.shape[-1]
    eventsV2[:,0] += dataV1.shape[-1]
    e1 = np.r_[eventsT1, eventsT2]
    e2 = np.r_[eventsV1, eventsV2]
    d1 = np.c_[dataT1, dataT2]
    d2 = np.c_[dataV1, dataV2]
    
    del (globals()['eventsT1'], globals()['eventsT2'], globals()['eventsV1'], globals()['eventsV2'])
    del (globals()['dataT1'], globals()['dataT2'], globals()['dataV1'], globals()['dataV2'])
    
    e1[:, 1] = np.where(e1[:, 1] == 2, 1, 2) # troca class_ids 1=LH, 2=RH
    e2[:, 1] = np.where(e2[:, 1] == 2, 1, 2)

    info = {'fs':Fs, 'class_ids':[1, 2], 'trial_tcue':3.0, 'trial_tpause':7.0,
            'trial_mi_time':4.0, 'trials_per_class':100, 'eeg_channels':62,
            'ch_labels':list(['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10','FC3','FC4',
                              'C5','C1','C2','C6','CP3','CPz','CP4','P1','P2','POz','FT9','FTT9h','TTP7h','TP7','TPP9h','FT10','FTT10h','TPP8h','TP8','TPP10h','F9','F10','AF7','AF3','AF4','AF8','PO3','PO4']),
            'datetime':datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
     
    if ch_grid == 'm1':
        sensors = [7, 8, 9, 10, 12, 13, 14, 17, 18, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 40]
        d1, d2 = d1[sensors], d2[sensors]
        info['eeg_channels'] = len(sensors)
        info['ch_labels'] = ['FC5','FC1','FC2','FC6','C3','Cz','C4','CP5','CP1','CP2','CP6','FC3', 'FC4','C5','C1','C2','C6', 'CP3', 'CPz','CP4']
    
    elif ch_grid == 'openbci_16':
        sensors = [12,35,13,36,14,38,39,40,32,8,9,33,34,37,28,30]
        d1, d2 = d1[sensors], d2[sensors]
        info['eeg_channels'] = len(sensors)
        info['ch_labels'] = ['C3','C1','Cz','C2','C4','CP3','CPz','CP4','FC3','FC1','FC2','FC4','C5','C6','O1','O2']
        downsampling = True
        factor = 8
    
    elif ch_grid == 'openbci_8':
        sensors = [12,35,13,36,14,38,39,40]
        d1, d2 = d1[sensors], d2[sensors]
        info['eeg_channels'] = len(sensors)
        info['ch_labels'] = ['C3','C1','Cz','C2','C4','CP3','CPz','CP4']
        downsampling = True
        factor = 4
    
    
    if downsampling:
        Fs = Fs//factor
        info['fs'] = Fs
        # d1 = np.asarray([d1[:,i] for i in range(0,d1.shape[-1],factor)]).T 
        # d2 = np.asarray([d2[:,i] for i in range(0,d2.shape[-1],factor)]).T
        d1, d2 = decimate(d1, factor), decimate(d2, factor)
        # d1, d2 = resample(d1, d1.shape[-1]//factor, axis=-1), resample(d2, d2.shape[-1]//factor, axis=-1) 

        e1[:,0] = [round(e1[i,0]/factor) for i in range(e1.shape[0])]
        e2[:,0] = [round(e2[i,0]/factor) for i in range(e2.shape[0])]
        
    #%% save npy session files
    np.save(path_out + 'S' + str(suj) + 'T', [d1, e1, info]) #sess1
    np.save(path_out + 'S' + str(suj) + 'E', [d2, e2, info]) #sess2
    # # np.save(path_out + 'sess01_subj' + suj_in + '_EEG_MI', [d1, e1, info])
    # # np.save(path_out + 'sess02_subj' + suj_in + '_EEG_MI', [d2, e2, info])

    #%% prepare agregate file (single npy file with all sessions)
    ee2 = np.copy(e2)
    ee2[:,0] += d1.shape[-1] # e2 pos + last d1 pos (e2 is continued by e1)
    all_events = np.r_[e1, ee2]
    all_data = np.c_[d1, d2] 
    info['trials_per_class'] = int(info['trials_per_class'] * 2)

    #%% save npy agregate file
    np.save(path_out + 'S' + str(suj), [all_data, all_events, info])
    # np.save(path_out + 'subj' + suj_in + '_EEG_MI', [all_data, all_events, info])
    # pickle.dump([all_data,all_events,info], open(path_out + 'subj' + suj_in + '_EEG_MI' + '.pkl', 'wb'))
    del(globals()['d1'], globals()['d2'], globals()['e1'], globals()['e2'], globals()['ee2'])
    del(globals()['all_events'], globals()['all_data'], globals()['info'])