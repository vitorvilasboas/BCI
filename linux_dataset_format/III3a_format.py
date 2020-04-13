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

for suj in ['K3','K6','L1']:
    raw = mne.io.read_raw_gdf(path + 'gdf/' + suj + '.gdf').load_data()
    data = raw.get_data()[:60] # [channels x samples]
    # data = corrigeNaN(data)
    events_raw = mne.events_from_annotations(raw) # raw.find_edf_events()
    ev = np.delete(events_raw[0], 1, axis=1) # elimina coluna de zeros
    truelabels = np.ravel(pd.read_csv(path + 'gdf/true_labels/trues_' + suj + '.csv'))
       
    cond = False
    for i in [1, 2, 3]: cond += (ev[:,1] == i)
    idx = np.where(cond)[0]
    ev = np.delete(ev, idx, axis=0)
    
    ev[:,1] = np.where(ev[:,1]==4, 0, ev[:,1]) # Labeling Start trial t=0
    
    idx = np.where(ev[:,1]!=0)
    ev[idx,1] = truelabels  
    
    for i in range(0, len(ev)):
        if ev[i,1]==0: ev[i,1] = (ev[i+1,1]+10) # labeling start trial [11 a 14] according cue [1,2,3,4]
    
    info = {'fs':250, 'class_ids': [1, 2, 3, 4], 'trial_tcue': 3.0, 'trial_tpause': 7.0, 'trial_mi_time': 4.0,
            'trials_per_class': 90 if suj == 'K3' else 60, 'eeg_channels':60, 'ch_labels': raw.ch_names,
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}

    path_out = path + '/npy/'
    if not os.path.isdir(path_out): os.makedirs(path_out)

    # with open(path + '/omi/' + suj + '.omi', 'wb') as handle: pickle.dump([data, ev, info], handle)
    np.save(path + 'npy/' + suj, [data, ev, info], allow_pickle=True)