# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:44:10 2020
@author: Vitor Vilas-Boas
"""
""" 72 trials per classe * 2 sessions
    T = startTrial=0; cue=2; startMI=3.25; endMI=6; endTrial=7.5-8.5
    
    Dataset description MNE (Linux) (by vboas): more info in http://bbci.de/competition/iv/desc_2a.pdf
    Meta-info (Training data _T):
     	1=1023 (rejected trial)
     	2=768 (start trial)
     	3=1072 (Unknown/ Eye Moviments)
     	4=769 (Class 1 - LH - cue onset)
     	5=770 (Class 2 - RH - cue onset)
     	6=771 (Class 3 - Foot - cue onset)
     	7=772 (Class 4 - Tongue - cue onset)
     	8=277 (Eye closed) [suj 4 = 32766 (Start a new run)]
     	9=276 (Eye open)   [suj 4 = None ]
     	10=32766 (Start a new run) [suj 4 = None ]
    Meta-info (Test data _E):
     	1=1023 (rejected trial)
     	2=768 (start trial)
     	3=1072 (Unknown/ Eye Moviments)
     	4=783 (Cue unknown/undefined)
     	5=277 (Eye closed)
     	6=276 (Eye open)
     	7=32766 (Start a new run)
    
    Dataset description MNE (MAC & Windows) (by vboas): more info in http://bbci.de/competition/iv/desc_2a.pdf
    Meta-info (Training data _T):
     	1=1023 (rejected trial)
     	2=1072 (Unknown/ Eye Moviments) 
     	3=276 (Eye open)                      [suj 4 = 32766 (Start a new run)]  
     	4=277 (Eye closed)                    [suj 4 = 768 (start trial)]
     	5=32766 (Start a new run)             [suj 4 = 769 (Class 1 - LH - cue onset)]
     	6=768 (start trial)                   [suj 4 = 770 (Class 2 - RH - cue onset)]
     	7=769 (Class 1 - LH - cue onset)      [suj 4 = 771 (Class 3 - Foot - cue onset)]
     	8=770 (Class 2 - RH - cue onset)      [suj 4 = 772 (Class 4 - Tongue - cue onset)]
     	9=771 (Class 3 - Foot - cue onset)    [suj 4 = None ] 
     	10=772 (Class 4 - Tongue - cue onset) [suj 4 = None ]
    Meta-info (Test data _E):
     	1=1023 (rejected trial)
     	2=1072 (Unknown/ Eye Moviments) 
     	3=276 (Eye open)
     	4=277 (Eye closed) 
     	5=32766 (Start a new run)
     	6=768 (start trial)
     	7=783 (Cue unknown/undefined)
"""

import os
import mne
import pickle
import warnings
import numpy as np
from scipy.io import loadmat, savemat
from datetime import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

path = '/mnt/dados/eeg_data/IV2a/' ## >>> ENTER THE PATH TO THE DATASET HERE

path_out = path + 'npy/'
if not os.path.isdir(path_out): os.makedirs(path_out)

for suj in range(1,10):    
    raw = mne.io.read_raw_gdf(path + 'A0' + str(suj) + 'T.gdf').load_data()
    dt = raw.get_data()[:22] # [channels x samples]
    et_raw = mne.events_from_annotations(raw)  # raw.find_edf_events()
    
    et = np.delete(et_raw[0], 1, axis=1) # remove MNE zero columns
    et = np.delete(et,np.where(et[:,1] == 1), axis=0) # remove rejected trial
    et = np.delete(et,np.where(et[:,1] == 3), axis=0) # remove eye movements/unknown
    et = np.delete(et,np.where(et[:,1] == 8), axis=0) # remove eyes closed
    et = np.delete(et,np.where(et[:,1] == 9), axis=0) # remove eyes open
    et = np.delete(et,np.where(et[:,1] == 10), axis=0) # remove start of a new run/segment
    et[:,1] = np.where(et[:,1] == 2, 0, et[:,1]) # start trial t=0
    et[:,1] = np.where(et[:,1] == 4, 1, et[:,1]) # LH
    et[:,1] = np.where(et[:,1] == 5, 2, et[:,1]) # RH
    et[:,1] = np.where(et[:,1] == 6, 3, et[:,1]) # Foot
    et[:,1] = np.where(et[:,1] == 7, 4, et[:,1]) # Tongue
    for i in range(0, len(et)):
        if et[i,1] == 0: et[i,1] = (et[i+1,1]+10) # labeling start trial [11 a 14] according cue [1,2,3,4]
             
    raw = mne.io.read_raw_gdf(path + 'A0' + str(suj) + 'E.gdf').load_data()
    trues = np.ravel(loadmat(path + 'true_labels/A0' + str(suj) + 'E.mat')['classlabel'])
    dv = raw.get_data()[:22] # [channels x samples]
    ev_raw = mne.events_from_annotations(raw)  # raw.find_edf_events()
    ev = np.delete(ev_raw[0], 1, axis=1) # remove MNE zero columns
    ev = np.delete(ev,np.where(ev[:,1] == 1), axis=0) # remove rejected trial
    ev = np.delete(ev,np.where(ev[:,1] == 3), axis=0) # remove eye movements/unknown
    ev = np.delete(ev,np.where(ev[:,1] == 5), axis=0) # remove eyes closed
    ev = np.delete(ev,np.where(ev[:,1] == 6), axis=0) # remove eyes open
    ev = np.delete(ev,np.where(ev[:,1] == 7), axis=0) # remove start of a new run/segment
    ev[:,1] = np.where(ev[:,1] == 2, 0, ev[:,1]) # start trial t=0
    ev[np.where(ev[:,1] == 4),1] = trues # change unknown value labels(4) to value in [1,2,3,4]
    for i in range(0, len(ev)):
        if ev[i,1] == 0: ev[i,1] = (ev[i+1,1]+10) # labeling start trial [11 a 14] according cue [1,2,3,4]
    
    info = {'fs':250, 'class_ids':[1,2,3,4], 'trial_tcue':2.0, 'trial_tpause':6.0, 'trial_mi_time':4.0,
            'trials_per_class':72, 'eeg_channels':dt.shape[0], 'ch_labels':raw.ch_names,
            'datetime':datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
    
    # ### to 8 channels m1 grid 
    # grid_8ch = [7,8,9,10,11,13,15,17]
    # dt, dv = dt[grid_8ch], dv[grid_8ch]
    # info['ch_labels'] = ['EEG-C3','EEG-C1','EEG-Cz','EEG-C2','EEG-C4','EEG-CP3','EEG-CPz','EEG-CP4']
    # info['eeg_channels'] = len(grid_8ch)

    #%% save npy session files
    np.save(path_out + 'A0' + str(suj) + 'T', [dt,et,info], allow_pickle=True)
    np.save(path_out + 'A0' + str(suj) + 'E', [dv,ev,info], allow_pickle=True)

    #%% prepare agregate file (single npy file with all sessions)
    ev1 = np.copy(ev)
    ev[:,0] += len(dt.T) # ev pos + last dt pos (et is continued by ev)
    events = np.r_[et, ev]
    data = np.c_[dt, dv]
    info['trials_per_class'] = 144

    #%% save npy agregate file
    np.save(path_out + 'A0' + str(suj), [data,events,info], allow_pickle=True)
    # pickle.dump([data,events,info], open(path_out + ' A0' + str(suj) + '.pkl', 'wb'))
    # savemat(path_out + 'A0' + str(suj) + '.mat', mdict={'data':data,'events':events,'info':info})
