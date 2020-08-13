# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:44:10 2020
@author: Vitor Vilas-Boas
"""
""" 3 sujeitos (K3, K6, L1) | 4 classes | 60 canais | Fs 250Hz
    K3->(360 trials (90 por classe)) - 2 sessões
    K6,L1->(240 trials (60 por classe)) - 2 sessões 
    startTrial=0; beep/cross=2; startCue=3; startMI=4; endMI=7; endTrial(break)=10    

    Dataset description/Meta-info MNE (Linux) (by vboas):
    1=Beep (accustic stimulus, BCI experiment)
    2=Cross on screen (BCI experiment)
    3=Rejection of whole trial
    4=Start of Trial, Trigger at t=0s
    5=769 class1, Left hand - cue onset (BCI experiment)
    6=770 class2, Right hand - cue onset (BCI experiment)
    7=771 class3, Foot, towards Right - cue onset (BCI experiment)
    8=772 class4, Tongue - cue onset (BCI experiment)
    9=783 cue unknown/undefined (used for BCI competition) 
"""

import os
import mne
import warnings
import pandas as pd
import numpy as np
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

path = '/mnt/dados/eeg_data/III3a/' ## >>> ENTER THE PATH TO THE DATASET HERE

path_out = path + 'npy8/'
if not os.path.isdir(path_out): os.makedirs(path_out)

for suj in ['K3','K6','L1']:
    raw = mne.io.read_raw_gdf(path + suj + '.gdf').load_data()
    d = raw.get_data()[:60] # [channels x samples]
    # d = corrigeNaN(d)
    e_raw = mne.events_from_annotations(raw) # raw.find_edf_events()
    e = np.delete(e_raw[0], 1, axis=1) # elimina coluna de zeros
    truelabels = np.ravel(pd.read_csv(path + 'true_labels/' + suj + '.csv'))
    ch_names = raw.ch_names
       
    cond = False
    for i in [1, 2, 3]: cond += (e[:,1] == i)
    idx = np.where(cond)[0]
    e = np.delete(e, idx, axis=0)
    
    e[:,1] = np.where(e[:,1] == 4, 0, e[:,1]) # Labeling Start trial t=0
    
    idx = np.where(e[:,1] != 0)
    e[idx,1] = truelabels
    
    for i in range(0, len(e)):
        if e[i,1] == 0: e[i,1] = (e[i+1,1]+10) # labeling start trial [11 a 14] according cue [1,2,3,4]
    
    info = {'fs':250, 'class_ids': [1, 2, 3, 4], 'trial_tcue': 3.0, 'trial_tpause': 7.0, 'trial_mi_time': 4.0,
            'trials_per_class': 90 if suj == 'K3' else 60, 'eeg_channels':d.shape[0], 'ch_labels': ch_names,
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
    
    # ### to 8 channels m1 grid 
    # grid_8ch = [27,28,30,32,33,38,40,42]
    # d = d[grid_8ch]
    # info['ch_labels'] = ['EEG-C3','EEG-C1','EEG-Cz','EEG-C2','EEG-C4','EEG-CP3','EEG-CPz','EEG-CP4']
    # info['eeg_channels'] = len(grid_8ch)

    #%% save npy file
    np.save(path_out + suj, [d, e, info], allow_pickle=True)
    # pickle.dump([data, ev, info], open(path_out + suj + '.pkl', 'wb'))