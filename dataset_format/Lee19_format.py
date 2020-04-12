# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 14:27:46 2020
@author: Vitor Vilas-Boas
"""
import pickle
import numpy as np
from datetime import datetime
from scipy.signal import decimate
from scipy.io import loadmat

"""  'EEG_MI_train' and 'EEG_MI_test': training and test data
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

path = '/mnt/dados/eeg_data/Lee19/'
fs = 1000
downsampling = True

chnames = list(['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4',
          'T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz',
          'O2','PO10','FC3','FC4','C5','C1','C2','C6','CP3','CPz','CP4','P1','P2','POz','FT9',
          'FTT9h','TTP7h','TP7','TPP9h','FT10','FTT10h','TPP8h','TP8','TPP10h','F9','F10','AF7',
          'AF3','AF4','AF8','PO3','PO4'])

info = {'fs':fs, 'class_ids':[1, 2], 'trial_tcue':3.0, 'trial_tpause':7.0, 'trial_mi_time':4.0, 
        'trials_per_class':100, 'eeg_channels':62, 'ch_labels':chnames,
        'datetime':datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}

for suj in range(1, 55):
    suj_in = str(suj) if suj>=10 else ('0' + str(suj))
    S1 = loadmat(path + 'session1/sess01_subj' + suj_in + '_EEG_MI.mat')
    S2 = loadmat(path + 'session2/sess02_subj' + suj_in + '_EEG_MI.mat')
    T1 = S1['EEG_MI_train']
    V1 = S1['EEG_MI_test']
    T2 = S2['EEG_MI_train']
    V2 = S2['EEG_MI_test']
    dataT1 = T1['x'][0,0].T
    dataV1 = V1['x'][0,0].T
    dataT2 = T2['x'][0,0].T
    dataV2 = V2['x'][0,0].T
    eventsT1 = np.r_[ T1['t'][0,0], T1['y_dec'][0,0] ].T
    eventsV1 = np.r_[ V1['t'][0,0], V1['y_dec'][0,0] ].T
    eventsT2 = np.r_[ T2['t'][0,0], T2['y_dec'][0,0] ].T
    eventsV2 = np.r_[ V2['t'][0,0], V2['y_dec'][0,0] ].T
    
    if downsampling:
        factor = 4
        fs = fs/factor
        # dataT1 = np.asarray([ dataT1[:,i] for i in range(0, dataT1.shape[-1], factor) ]).T
        # dataV1 = np.asarray([ dataV1[:,i] for i in range(0, dataV1.shape[-1], factor) ]).T
        # dataT2 = np.asarray([ dataT2[:,i] for i in range(0, dataT2.shape[-1], factor) ]).T
        # dataV2 = np.asarray([ dataV2[:,i] for i in range(0, dataV2.shape[-1], factor) ]).T
        dataT1 = decimate(dataT1, factor)
        dataV1 = decimate(dataV1, factor)
        dataT2 = decimate(dataT2, factor)
        dataV2 = decimate(dataV2, factor)
        eventsT1[:, 0] = [ round(eventsT1[i, 0]/factor) for i in range(eventsT1.shape[0]) ]
        eventsV1[:, 0] = [ round(eventsV1[i, 0]/factor) for i in range(eventsV1.shape[0]) ]
        eventsT2[:, 0] = [ round(eventsT2[i, 0]/factor) for i in range(eventsT2.shape[0]) ]
        eventsV2[:, 0] = [ round(eventsV2[i, 0]/factor) for i in range(eventsV2.shape[0]) ]
        
    eventsV1[:,0] += dataT1.shape[-1]
    eventsV2[:,0] += dataT2.shape[-1]
    e1 = np.r_[eventsT1, eventsV1]
    e2 = np.r_[eventsT2, eventsV2]
    d1 = np.c_[dataT1, dataV1]
    d2 = np.c_[dataT2, dataV2]
    e1[:, 1] = np.where(e1[:, 1]==2, 1, 2) # troca class_ids 1=LH, 2=RH
    e2[:, 1] = np.where(e2[:, 1]==2, 1, 2)
    
    # cortex = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
    # d1, d2 = d1[cortex], d2[cortex]
    # info['eeg_channels'] = len(cortex)
    # info['ch_labels'] = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
    
    np.save(path+'npy/S'+str(suj)+'_sess1', [d1, e1, info])
    np.save(path+'npy/S'+str(suj)+'_sess2', [d2, e2, info])
    
    ee2 = np.copy(e2)
    ee2[:,0] += d1.shape[-1] # e2 pos + last d1 pos (e2 is continued by e1)
    all_events = np.r_[e1, ee2]
    all_data = np.c_[d1, d2] 
    info['trials_per_class'] = 200
    np.save(path+'npy/S'+str(suj), [all_data, all_events, info])
    # # with open(path+'npy/full/subj'+str(suj)+'.pkl', 'wb') as h: pickle.dump([all_data,all_events,info], h)
        
    

