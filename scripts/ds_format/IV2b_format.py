# -*- coding: utf-8 -*-
import mne
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.io import loadmat

""" 9 subjects | 2 classes (LH, RH) | 3 channels | Fs 250Hz
    6 channels (first 3 is EEG: C3, C4, Cz; last 3 is EOG)
    120 trials (60 per class) - 5 sessions
    2 sessions without feedback
    3 sessions with feedback (smiley)
    Total sessions = 5 (01T,02T,03T,04E,05E)
         5 * 120 trials = 600 total trials -> 5*60 = 300 per class -> 2*60 = 120 per session
	     2 training sessions (no feedback) - 01T,02T 
         1 training session (WITH feedback) - 03T
	     2 evaluate sessions (WITH feedback) - 04E,05E
         
    # startTrial=0; cue=3; startMI=4; endMI=7; endTrial=8.5-9.5
"""

""" Dataset Description (by vboas): more info in http://bbci.de/competition/iv/desc_2b.pdf
    01T e 02T (without feedback)
    		10 trials * 2 classes * 6 runs * 2 sessions = 240 trials (120 per class)
    		Cross t=0 (per 3s)
    		beep t=2s
    		cue t=3s (per 1.25s)
    		MI t=4s (per 3s)
    		Pause t=7s (per 1.5-2.5s)
    		EndTrial t=8.5-9.5
    	03T, 04E e 05E (with feedback)
    		10 trials * 2 classes * 4 runs * 3 sessions = 240 trials (120 per class)
    		Smiley(grey) t=0 (per 3.5s)
    		beep t=2s
    		cue t=3s (per 4.5s)
    		MI (Feedback períod) t=3.5s (per 4s)
    		Pause t=7.5s (per 1-2s)
    		EndTrial t=8.5-9.5
    	Meta-info 01T e 02T:
    		1=1023 (rejected trial)
    		2=768 (start trial)
    		3=769 (Class 1 - LH - cue onset)
    		4=770 (Class 2 - RH - cue onset)
    		5=277 (Eye closed)
    		6=276 (Eye open)
    		7=1081 (Eye blinks)
    		8=1078 (Eye rotation)
    		9=1077 (Horizontal eye movement)
    		10=32766 (Start a new run) *(to B0102T == 5)
    		11=1078 (Vertical eye movement)			
    	Meta-info 03T:
    		1=781 (BCI feedback - continuous)
    		2=1023 (rejected trial)
    		3=768 (start trial)
    		4=769 (Class 1 - LH - cue onset)
    		5=770 (Class 2 - RH - cue onset)
    		6=277 (Eye closed)
    		7=276 (Eye open)
    		8=1081 (Eye blinks)
    		9=1078 (Eye rotation)
    		10=1077 (Horizontal eye movement)
    		11=32766 (Start a new run)
    		12=1078 (Vertical eye movement)
    	Meta-info 04E e 05E:
    		1=781 (BCI feedback - continuous)
    		2=1023 (rejected trial)
    		3=768 (start trial)
    		4=783 (Cue unknown/undefined)
    		5=277 (Eye closed)
    		6=276 (Eye open)
    		7=1081 (Eye blinks)
    		8=1078 (Eye rotation)
    		9=1077 (Horizontal eye movement)
    		10=32766 (Start a new run)
    		11=1078 (Vertical eye movement)
"""

# mne.set_log_level('WARNING','DEBUG')
path = '/mnt/dados/eeg_data/IV2b/'
fs = 250

info = {'fs': 250, 'class_ids': [1, 2], 'trial_tcue': 3.0,
            'trial_tpause': 7.0, 'trial_mi_time': 4.0, 'trials_per_class': 60,
            'eeg_channels': 3, 'ch_labels': {'EEG1':'C3', 'EEG2':'Cz', 'EEG3':'C4'},
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}

for suj in range(1,10): 
    DT, EV = [], []
    for ds in ['01T','02T','03T','04E','05E']:
        raw = mne.io.read_raw_gdf(path + 'gdf/B0' + str(suj) + ds + '.gdf')
        raw.load_data()
        data = raw.get_data() # [channels x samples]
        data = data[:3]
        # data = corrigeNaN(data) # Correção de NaN nos dados brutos
        ev = raw.find_edf_events()
        ev = np.delete(ev[0],1,axis=1) # elimina coluna de zeros
        truelabels = np.ravel(loadmat(path + 'gdf/true_labels/B0' + str(suj) + ds + '.mat')['classlabel'])
    
        if ds in ['01T','02T']:
            for rm in range(5,12): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
            ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete rejected trials
            ev[:,1] = np.where(ev[:,1]==2, 0, ev[:,1]) # altera label start trial de 2 para 0
            ev[:,1] = np.where(ev[:,1]==3, 1, ev[:,1]) # altera label cue LH de 3 para 1
            ev[:,1] = np.where(ev[:,1]==4, 2, ev[:,1]) # altera label cue RH de 4 para 2
        elif ds=='03T': 
            for rm in range(6,13): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
            ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # delete rejected trials
            ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete feedback continuous
            ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1]) # altera label start trial de 3 para 0
            ev[:,1] = np.where(ev[:,1]==4, 1, ev[:,1]) # altera label cue LH de 4 para 1
            ev[:,1] = np.where(ev[:,1]==5, 2, ev[:,1]) # altera label cue RH de 5 para 2
        else:
            for rm in range(5,12): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
            ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # delete rejected trials
            ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete feedback continuous
            ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1]) # altera label start trial de 3 para 0
            ev[np.where(ev[:,1]==4),1] = truelabels #rotula momento da dica conforme truelabels
        
        np.save(path + 'npy/B0' + str(suj) + ds, [ data, ev, info ], allow_pickle=True)
        
        DT.append(data)
        EV.append(ev)
        
    # Save a unique npy file with all datasets
    soma = 0
    for i in range(1,len(EV)): 
        soma += len(DT[i-1].T)
        EV[i][:,0] += soma
        
    all_data = np.c_[DT[0],DT[1],DT[2],DT[3],DT[4]]
    all_events = np.r_[EV[0],EV[1],EV[2],EV[3],EV[4]]
    info['trials_per_class'] = 360
    
    
    np.save(path + 'npy/B0' + str(suj), [ all_data, all_events, info ], allow_pickle=True)
    # with open(path + '/omi/B0' + str(suj) + '.omi', 'wb') as handle: pickle.dump([ all_data, all_events, info ], handle)

