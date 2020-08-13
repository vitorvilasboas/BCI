# -*- coding: utf-8 -*-
"""
Created on Sat Abr 13 09:44:10 2020
@author: Vitor Vilas-Boas
"""
import os
import mne
import warnings
import numpy as np
from datetime import datetime
from scripts.bci_utils import extractEpochs

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

path = '/mnt/dados/eeg_data/LINCE/' ## >>> ENTER THE PATH TO THE DATASET HERE

path_out = path + 'npy/'
if not os.path.isdir(path_out): os.makedirs(path_out)

################################ CL #######################################
""" 1 subject (CL) | 3 classes (lh, rh, foot) | 16 channels | Fs 125Hz
    lh-rh -> 100 trials (50 per class) 5*20 - 1 session
    lh-ft -> 48 trials (24 per class) 3*16 - 1 session
    Start trial=0; Beep=1; Wait=2; Start cue=2; Start MI=3; End MI=9; End trial(break)=14
"""
## Cleison Left Hand x Right Hand
d, e = np.load(path + 'CL_LR_data.npy').T, np.load(path + 'CL_LR_events.npy').astype(int)
for i in range(0, len(e)): 
    if e[i,1]==0: e[i,1] = (e[i+1,1]+10) # labeling start trial [11,12] according cue [1,2]
info = {'fs': 125, 'class_ids': [1, 2], 'trial_tcue': 2.0, 'trial_tpause': 9.0,
        'trial_mi_time': 7.0, 'trials_per_class': 50, 'eeg_channels': d.shape[0],
        'ch_labels': None, 'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
np.save(path_out + 'CL_LR', [d, e, info], allow_pickle=True) # pickle.dump([d, e, i], open(path_out + 'CL_LR' + '.pkl', 'wb'))

## Cleison Left Hand x Foot
d, e = np.load(path + 'CL_LF_data.npy').T, np.load(path + 'CL_LF_events.npy').astype(int)
e[:,1] = np.where(e[:,1] == 2, 3, e[:,1]) # LH=1, FooT=3
for i in range(0, len(e)): 
    if e[i,1]==0: e[i,1] = (e[i+1,1]+10) # labeling start trial [11,12] according cue [1,2]
info['class_ids'] = [1, 3]
info['trials_per_class'] = 24
np.save(path_out + 'CL_LF', [d, e, info], allow_pickle=True)

    
# ############################# TL & WL #####################################
""" 2 subjects (TL, WL) | 2 classes (lh, rh) | Fs 250Hz
    40 trials (20 per class) - TL: 2 sessions; WL:3 sessions
    8 channels (1=Cz 2=Cpz 3=C1 4=C3 5=CP3 6=C2 7=C4 8=CP4)
    Scalp map:      C3  C1  Cz  C2  CP4     4  3  1  6  7
                        CP3   CPz  CP4 		  5   2   8
    Start trial=0; Wait beep=2; Start cue=3; Start MI=4.25; End MI=8; End trial(break)=10-12

    Dataset description/Meta-info MNE (Linux) (by vboas):
    1=Cross on screen (BCI experiment)
    2=Feedback (continuous) - onset (BCI experiment)
    3=768 Start of Trial, Trigger at t=0s
    4=783 Unknown
    5=769 class1, Left hand - cue onset (BCI experiment)
    6=770 class2, Right hand - cue onset (BCI experiment)
"""
for subj in ['TL', 'WL']:
    DT, DV, ET, EV = [],[],[],[]
    for session in ['S1', 'S2']:
        raw = mne.io.read_raw_gdf(path + subj + '_' + session + '.gdf').load_data()
        d = raw.get_data()[:8]  # [channels x samples]
        # d = corrigeNaN(d) 
        e_raw = mne.events_from_annotations(raw)  # raw.find_edf_events()
        e = np.delete(e_raw[0], 1, axis=1)  # elimina coluna de zeros
        e = np.delete(e, np.where(e[:, 1] == 1), axis=0)  # elimina marcações inuteis (cross on screen)
        e = np.delete(e, np.where(e[:, 1] == 2), axis=0)  # elimina marcações inuteis (feedback continuous)
        e = np.delete(e, np.where(e[:, 1] == 4), axis=0)  # elimina marcações inuteis (unknown)
        
        e[:, 1] = np.where(e[:, 1] == 3, 0, e[:, 1])
        e[:, 1] = np.where(e[:, 1] == 5, 1, e[:, 1])  # altera label lh de 5 para 1
        e[:, 1] = np.where(e[:, 1] == 6, 2, e[:, 1])  # altera label rh de 6 para 2
        
        for i in range(0, len(e)):
            if e[i,1]==0: e[i,1] = (e[i+1,1]+10) # labeling start trial [11,12] according cue [1,2]
        
        i = {'fs': 250, 'class_ids': [1, 2], 'trial_tcue': 3.0, 'trial_tpause': 8.0,
              'trial_mi_time': 5.0, 'trials_per_class': 20, 'eeg_channels': d.shape[0],
              'ch_labels': ['EEG-Cz','EEG-Cpz','EEG3-C1','EEG-C3','EEG-CP3','EEG-C2','EEG-C4','EEG-CP4'],
              'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        
        #%% save npy file
        np.save(path_out + subj + '_' + session, [d, e, i], allow_pickle=True)
        # pickle.dump([d, e, i], open(path_out + subj + '_' + session + '.pkl', 'wb'))
        
        if session == 'S1': DT, ET = d, e
        else: DV, EV = d, e
    
    #%% prepare agregate file (single npy file with all sessions)
    all_data = np.c_[DT, DV]    
    EV_ = np.copy(EV)
    EV_[:,0] += len(DT.T) # eventsV pos + last dataT pos (eventsT is continued by eventsV)
    all_events = np.r_[ET, EV_]
    i['trials_per_class'] = 40
    
    #%% save npy agregate file
    np.save(path_out + subj, [all_data,all_events,i], allow_pickle=True)
    # pickle.dump([data,events,info], open(path_out + ' A0' + str(suj) + '.pkl', 'wb'))
    # savemat(path_out + 'A0' + str(suj) + '.mat', mdict={'data':data,'events':events,'info':info})
    
    
    
    
    
    
    
    
    
    
    
    

    

    
