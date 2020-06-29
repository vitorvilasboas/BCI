# -*- coding: utf-8 -*-
import re
import os
import mne
import math
import pickle
import warnings
import itertools
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from time import time, sleep
from scipy.io import loadmat
from scipy.stats import norm
from datetime import datetime
from scipy.fftpack import fft
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import cohen_kappa_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter, decimate, welch
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold 

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

def labeling(path=None, ds=None, session=None, subj=None, channels=None, save=False, downsampling=True):
    
    if ds == 'IV2a':
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
        mne.set_log_level(50, 50)
        raw = mne.io.read_raw_gdf(path + 'A0' + str(subj) + session + '.gdf').load_data()
        d = raw.get_data()[:22] # [channels x samples]
        if not channels is None: d = d[channels]
        e_raw = mne.events_from_annotations(raw) #raw.find_edf_events()
        e = np.delete(e_raw[0], 1, axis=1) # remove MNE zero columns
        e = np.delete(e,np.where(e[:,1]==1), axis=0) # remove rejected trial
        e = np.delete(e,np.where(e[:,1]==3), axis=0) # remove eye movements/unknown
        if session == 'T':
            e = np.delete(e,np.where(e[:,1]==8), axis=0) # remove eyes closed
            e = np.delete(e,np.where(e[:,1]==9), axis=0) # remove eyes open 
            e = np.delete(e,np.where(e[:,1]==10), axis=0) # remove start of a new run/segment
            e[:,1] = np.where(e[:,1]==2, 0, e[:,1]) # start trial t=0
            e[:,1] = np.where(e[:,1]==4, 1, e[:,1]) # LH 
            e[:,1] = np.where(e[:,1]==5, 2, e[:,1]) # RH 
            e[:,1] = np.where(e[:,1]==6, 3, e[:,1]) # Foot
            e[:,1] = np.where(e[:,1]==7, 4, e[:,1]) # Tongue
        else:
            trues = np.ravel(loadmat(path + 'true_labels/A0' + str(subj) + 'E.mat' )['classlabel'])
            e = np.delete(e,np.where(e[:,1]==5), axis=0) # remove eyes closed
            e = np.delete(e,np.where(e[:,1]==6), axis=0) # remove eyes open
            e = np.delete(e,np.where(e[:,1]==7), axis=0) # remove start of a new run/segment
            e[:,1] = np.where(e[:,1]==2, 0, e[:,1]) # start trial t=0
            e[np.where(e[:,1]==4),1] = trues # change unknown value labels(4) to value in [1,2,3,4]   
        for i in range(0, len(e)):
            if e[i,1]==0: e[i,1] = (e[i+1,1]+10) # labeling start trial [11 a 14] according cue [1,2,3,4]
        i = {'fs':250, 'class_ids':[1,2,3,4], 'trial_tcue':2.0, 'trial_tpause':6.0, 
             'trial_mi_time':4.0, 'trials_per_class':72, 'eeg_channels':d.shape[0], 
             'ch_labels':raw.ch_names,
             'datetime':datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        out = 'A0' + str(subj) + session
    
    elif ds == 'IV2b':
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
            
            Dataset Description (by vboas): more info in http://bbci.de/competition/iv/desc_2b.pdf
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
        mne.set_log_level(50, 50)
        raw = mne.io.read_raw_gdf(path + 'B0' + str(subj) + session + '.gdf').load_data()
        d = raw.get_data()[:3] # [channels x samples]
        if not channels is None: d = d[channels]
        e = mne.events_from_annotations(raw) #raw.find_edf_events()
        e = np.delete(e[0],1,axis=1) # elimina coluna de zeros
        truelabels = np.ravel(loadmat(path + 'true_labels/B0' + str(subj) + session + '.mat')['classlabel'])
        if session in ['01T','02T']:
            for rm in range(5,12): e = np.delete(e,np.where(e[:,1]==rm),axis=0) # detele various eye movements marks
            e = np.delete(e,np.where(e[:,1]==1),axis=0) # delete rejected trials
            e[:,1] = np.where(e[:,1]==2, 0, e[:,1]) # altera label start trial de 2 para 0
            e[:,1] = np.where(e[:,1]==3, 1, e[:,1]) # altera label cue LH de 3 para 1
            e[:,1] = np.where(e[:,1]==4, 2, e[:,1]) # altera label cue RH de 4 para 2
        elif session == '03T': 
            for rm in range(6,13): e = np.delete(e,np.where(e[:,1]==rm),axis=0) # detele various eye movements marks
            e = np.delete(e,np.where(e[:,1]==2),axis=0) # delete rejected trials
            e = np.delete(e,np.where(e[:,1]==1),axis=0) # delete feedback continuous
            e[:,1] = np.where(e[:,1]==3, 0, e[:,1]) # altera label start trial de 3 para 0
            e[:,1] = np.where(e[:,1]==4, 1, e[:,1]) # altera label cue LH de 4 para 1
            e[:,1] = np.where(e[:,1]==5, 2, e[:,1]) # altera label cue RH de 5 para 2
        else:
            for rm in range(5,12): e = np.delete(e,np.where(e[:,1]==rm),axis=0) # detele various eye movements marks
            e = np.delete(e,np.where(e[:,1]==2),axis=0) # delete rejected trials
            e = np.delete(e,np.where(e[:,1]==1),axis=0) # delete feedback continuous
            e[:,1] = np.where(e[:,1]==3, 0, e[:,1]) # altera label start trial de 3 para 0
            e[np.where(e[:,1]==4),1] = truelabels #rotula momento da dica conforme truelabels    
        for i in range(0, len(e)):
            if e[i,1]==0: e[i,1] = (e[i+1,1]+10) # labeling start trial [11,12] according cue [1,2]
        i = {'fs':250, 'class_ids':[1, 2], 'trial_tcue':3.0, 'trial_tpause':7.0, 
             'trial_mi_time':4.0, 'trials_per_class':60, 'eeg_channels':d.shape[0],
             'ch_labels':{'EEG1':'C3', 'EEG2':'Cz', 'EEG3':'C4'},
             'datetime':datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        out = 'B0' + str(subj) + session
        
    elif ds == 'III3a': 
        """ 3 sujeitos (K3, K6, L1) | 4 classes | 60 canais | Fs 250Hz
            K3->(360 trials (90 por classe)) - 2 sessões
            K6,L1->(240 trials (60 por classe)) - 2 sessões 
            startTrial=0; beep/cross=2; startCue=3; startMI=4; endMI=7; endTrial(break)=10    
            
            Dataset description/Meta-info MNE (Linux):
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
        mne.set_log_level(50, 50)
        raw = mne.io.read_raw_gdf(path + subj + '.gdf').load_data()
        d = raw.get_data()[:60] # [channels x samples]
        if not channels is None: d = d[channels]
        e_raw = mne.events_from_annotations(raw) #raw.find_edf_events()
        e = np.delete(e_raw[0], 1,axis=1) # elimina coluna de zeros
        truelabels = np.ravel(pd.read_csv(path + 'true_labels/' + subj + '.csv'))
        cond = False
        for i in [1, 2, 3]: cond += (e[:,1] == i)
        idx = np.where(cond)[0]
        e = np.delete(e, idx, axis=0)
        e[:,1] = np.where(e[:,1] == 4, 0, e[:,1]) # Labeling Start trial t=0
        idx = np.where(e[:,1] != 0)
        e[idx,1] = truelabels  
        for i in range(0, len(e)):
            if e[i,1] == 0: e[i,1] = (e[i+1,1]+10) # labeling start trial [11 a 14] according cue [1,2,3,4]
        i = {'fs':250, 'class_ids':[1,2,3,4], 'trial_tcue':3.0, 'trial_tpause':7.0, 
             'trial_mi_time':4.0, 'trials_per_class':90 if subj == 'K3' else 60, 'eeg_channels':d.shape[0], 
             'ch_labels':raw.ch_names,
             'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        out = subj
    
    elif ds == 'III4a':
        """ 5 subjects | 2 classes (RH, FooT)
            Epoch distribution:
                aa : train=168 test=112  
                al : train=224 test=56
                av : train=84  test=196
                aw : train=56  test=224
                ay : train=28  test=252
            Start trial=0; Start cue=0; Start MI=0; End MI=3.5; End trial(break)=5.25~5.75
        """
        mat = loadmat(path + subj + '.mat')
        d = mat['cnt'].T # (0.1 * mat['cnt'].astype(float)).T # convert to uV
        if not channels is None: d = d[channels]
        pos = mat['mrk'][0][0][0][0]
        true_mat = loadmat(path + 'true_labels/' + subj + '.mat')
        true_y = np.ravel(true_mat['true_y']) # RH=1 Foot=2
        true_y = np.where(true_y == 2, 3, true_y) # RH=1 Foot=3
        e = np.c_[pos, true_y]
        i = {'fs':100, 'class_ids':[1, 3], 'trial_tcue':0, 'trial_tpause':4.0, 
             'trial_mi_time':4.0, 'trials_per_class':140, 'eeg_channels':d.shape[0], 
             'ch_labels': mat['nfo']['clab'],
             'datetime':datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        out = subj
    
    elif ds == 'Lee19':
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
        subj_in = str(subj) if subj>=10 else ('0' + str(subj))
        S = loadmat(path + 'session'+ str(session) +'/sess0' + str(session) + '_subj' + subj_in + '_EEG_MI.mat')
        T = S['EEG_MI_train']
        V = S['EEG_MI_test']
        dataT = T['x'][0,0].T
        dataV = V['x'][0,0].T
        eventsT = np.r_[ T['t'][0,0], T['y_dec'][0,0] ].T
        eventsV = np.r_[ V['t'][0,0], V['y_dec'][0,0] ].T
        Fs = 1000
        if downsampling:
            factor = 4
            Fs = Fs/factor
            # dataT = np.asarray([ dataT[:,i] for i in range(0, dataT.shape[-1], factor) ]).T
            # dataV = np.asarray([ dataV[:,i] for i in range(0, dataV.shape[-1], factor) ]).T
            dataT = decimate(dataT, factor)
            dataV = decimate(dataV, factor)
            eventsT[:, 0] = [ round(eventsT[i, 0]/factor) for i in range(eventsT.shape[0]) ]
            eventsV[:, 0] = [ round(eventsV[i, 0]/factor) for i in range(eventsV.shape[0]) ]    
        eventsV[:,0] += dataT.shape[-1]
        e = np.r_[eventsT, eventsV]
        d = np.c_[dataT, dataV]
        if not channels is None: d = d[channels]
        e[:, 1] = np.where(e[:, 1]==2, 1, 2) # troca class_ids 1=LH, 2=RH 
        i = {'fs':Fs, 'class_ids':[1, 2], 'trial_tcue':3.0, 'trial_tpause':7.0, 
             'trial_mi_time':4.0, 'trials_per_class':100, 'eeg_channels':d.shape[0],
             'ch_labels':list(['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4',
                               'T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz',
                               'O2','PO10','FC3','FC4','C5','C1','C2','C6','CP3','CPz','CP4','P1','P2','POz','FT9',
                               'FTT9h','TTP7h','TP7','TPP9h','FT10','FTT10h','TPP8h','TP8','TPP10h','F9','F10','AF7',
                               'AF3','AF4','AF8','PO3','PO4']),
             'datetime':datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        out = 'sess0' + str(session) + '_subj' + subj_in + '_EEG_MI'

    elif ds == 'TWL':
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
        mne.set_log_level(50, 50)
        raw = mne.io.read_raw_gdf(path + subj + '_' + session + '.gdf').load_data()
        d = raw.get_data()[:8] # [channels x samples]
        if not channels is None: d = d[channels]
        e_raw = mne.events_from_annotations(raw) #raw.find_edf_events()
        e = np.delete(e_raw[0],1,axis=1) # elimina coluna de zeros
        e = np.delete(e,np.where(e[:,1]==1),axis=0) # elimina marcações inuteis (cross on screen)
        e = np.delete(e,np.where(e[:,1]==2),axis=0) # elimina marcações inuteis (feedback continuous)
        e = np.delete(e,np.where(e[:,1]==4),axis=0) # elimina marcações inuteis (unknown)
        e[:,1] = np.where(e[:,1]==3, 0, e[:,1])
        e[:,1] = np.where(e[:,1]==5, 1, e[:,1]) # altera label lh de 5 para 1
        e[:,1] = np.where(e[:,1]==6, 2, e[:,1]) # altera label rh de 6 para 2
        for i in range(0, len(e)):
            if e[i,1]==0: e[i,1] = (e[i+1,1]+10) # labeling start trial [11,12] according cue [1,2]
        i = {'fs':250, 'class_ids':[1,2], 'trial_tcue':3.0, 'trial_tpause':8.0, 
             'trial_mi_time':5.0, 'trials_per_class':40, 'eeg_channels':d.shape[0],
             'ch_labels':{'EEG1':'Cz', 'EEG2':'Cpz', 'EEG3':'C1', 'EEG4':'C3', 'EEG5':'CP3', 'EEG6':'C2', 'EEG7':'C4', 'EEG8':'CP4'},
             'datetime':datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        out = subj + '_' + session
               
    elif ds == 'CL':
        """ 1 subject (CL) | 3 classes (lh, rh, foot) | 16 channels | Fs 125Hz
            lh-rh -> 100 trials (50 per class) 5*20 - 1 session
            lh-ft -> 48 trials (24 per class) 3*16 - 1 session
            Start trial=0; Beep=1; Wait=2; Start cue=2; Start MI=3; End MI=9; End trial(break)=14
        """
        d = np.load(path + subj + '_' + session + '_data.npy').T
        if not channels is None: d = d[channels]
        e = np.load(path + subj + '_' + session + '_events.npy').astype(int)
        if session=='LF': e[:,1] = np.where(e[:,1] == 2, 3, e[:,1]) # LH=1, FooT=3
        for i in range(0, len(e)):
            if e[i,1]==0: e[i,1] = (e[i+1,1]+10) # labeling start trial [11,12] according cue [1,2]   
        i = {'fs': 125, 'class_ids': [1, 2] if session=='LR' else [1, 3], 'trial_tcue': 2.0, 'trial_tpause': 9.0,
             'trial_mi_time': 7.0, 'trials_per_class': 50 if session=='LR' else 24, 'eeg_channels': d.shape[0],
             'ch_labels': None, 'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        out = subj + '_' + session
        
    if save: np.save(path + out, [d, e, i], allow_pickle=True) # pickle.dump([d, e, i], open(path + subj + '.pkl', 'wb'))
    return d, e, i


def extractEpochs(data, events, smin, smax, class_ids):
    events_list = events[:, 1] # get class labels column
    cond = False
    for i in range(len(class_ids)): cond += (events_list == class_ids[i]) #get only class_ids pos in events_list
    idx = np.where(cond)[0]
    s0 = events[idx, 0] # get initial timestamps of each class epochs
    sBegin = s0 + smin
    sEnd = s0 + smax
    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    labels = events_list[idx]
    bad_epoch_list = []
    for i in range(n_epochs):
        epoch = data[:, sBegin[i]:sEnd[i]]
        if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch # Check if epoch is complete
        else:
            print('Incomplete epoch detected...', n_samples, '!=', epoch.shape[1])
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels
    
    
def nanCleaner(epoch):
    """Removes NaN from data by interpolation
    data_in : input data - np matrix channels x samples
    data_out : clean dataset with no NaN samples"""
    for i in range(epoch.shape[0]):
        bad_idx = np.isnan(epoch[i, :])
        epoch[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], epoch[i, ~bad_idx])
    return epoch
    
    
def corrigeNaN(data):
    for ch in range(data.shape[0] - 1):
        this_chan = data[ch]
        data[ch] = np.where(this_chan == np.min(this_chan), np.nan, this_chan)
        mask = np.isnan(data[ch])
        meanChannel = np.nanmean(data[ch])
        data[ch, mask] = meanChannel
    return data


class Filter:
    def __init__(self, fl, fh, srate, filt_info, forder=5, band_type='bandpass'):
        self.ftype = filt_info['design']
        if self.ftype != 'DFT':
            self.nyq = 0.5 * srate
            low = fl / self.nyq
            high = fh / self.nyq        
            if low == 0: low = 0.001
            if high >= 1: high = 0.99
            if self.ftype == 'IIR':
                self.forder = filt_info['iir_order']
                # self.b, self.a = iirfilter(self.forder, [low, high], btype='band')
                self.b, self.a = butter(self.forder, [low, high], btype=band_type)
            elif self.ftype == 'FIR':
                self.forder = filt_info['fir_order']
                self.b = firwin(self.forder, [low, high], window='hamming', pass_zero=False)
                self.a = [1]

    def apply_filter(self, X, is_epoch=False):
        if self.ftype != 'DFT': XF = lfilter(self.b, self.a, X) # lfilter, filtfilt
        else:
            XF = fft(X)
            if is_epoch:
                real, imag = np.real(XF).T, np.imag(XF).T
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))))
            else:
                real = np.transpose(np.real(XF), (2, 0, 1))
                imag = np.transpose(np.imag(XF), (2, 0, 1))
                XF = np.transpose(list(itertools.chain.from_iterable(zip(imag, real))), (1, 2, 0)) 
        return XF


class CSP():
    def __init__(self, n_components):
        self.n_components = n_components
        self.filters_ = None
    def fit(self, X, y):
        e, c, s = X.shape
        classes = np.unique(y)   
        Xa = X[classes[0] == y,:,:]
        Xb = X[classes[1] == y,:,:]
        S0 = np.zeros((c, c)) 
        S1 = np.zeros((c, c))
        for epoca in range(int(e/2)):
            # S0 = np.add(S0, np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T), out=S0, casting="unsafe")
            # S1 = np.add(S1, np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T), out=S1, casting="unsafe")
            # S0 += np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T) #covA Xa[epoca]
            # S1 += np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T) #covB Xb[epoca]
            S0 += np.dot(Xa[epoca, :, :], Xa[epoca, :, :].T) / Xa[epoca].shape[-1]  # sum((Xa * Xa.T)/q)
            S1 += np.dot(Xb[epoca, :, :], Xb[epoca, :, :].T) / Xb[epoca].shape[-1]  # sum((Xb * Xb.T)/q)
        S0 /= len(Xa)
        S1 /= len(Xb)
        [D, W] = eigh(S0, S0 + S1) # + 1e-10 * np.eye(22))
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
        ind[1::2] = np.arange(0, c // 2)
        # W += 1e-1 * np.eye(22)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self # instruction add because cross-validation pipeline
    def transform(self, X):        
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        # XVAR = np.log(np.mean(XT ** 2, axis=2)) # Xcsp
        XVAR = np.log(np.var(XT, axis=2))
        return XVAR


class BCI():
    
    def __init__(self, data=None, events=None, class_ids=[1,2], fs=250, overlap=True, crossval=False, nfolds=10, test_perc=0.5, 
                 f_low=None, f_high=None, tmin=None, tmax=None, ncomp=None, nbands=None, ap=None, filt_info=None, clf=None, split='common'):
        self.data = data
        self.events = events
        self.class_ids = class_ids
        self.overlap = overlap
        self.fs = fs
        self.crossval = crossval
        self.nfolds = nfolds
        self.test_perc = test_perc
        self.f_low = f_low
        self.f_high = f_high
        self.tmin = tmin
        self.tmax = tmax
        self.ncomp = ncomp
        self.nbands = nbands
        self.ap = ap
        self.filt_info = filt_info
        self.clf = clf
        self.acc = None
        self.kappa = None
        self.csp_list = None
        self.split = split
        

    def evaluate(self):
        
        if self.clf['model'] == 'LDA': self.clf_final = LDA()
            # lda_shrinkage = None
            # if not (clf_dict['lda_solver'] == 'svd'): 
            #     lda_shrinkage = self.clf['shrinkage'] if self.clf['shrinkage'] in [None,'auto'] else self.clf['shrinkage']['shrinkage_float']
            # self.clf_final = LDA(solver=self.clf['lda_solver'], shrinkage=lda_shrinkage)
        elif self.clf['model'] == 'Bayes': self.clf_final = GaussianNB()
        elif self.clf['model'] == 'SVM': 
            # degree = self.clf['kernel']['degree'] if self.clf['kernel']['kf'] == 'poly' else 3
            # gamma = self.clf['gamma'] if self.clf['gamma'] in ['scale', 'auto'] else 10 ** (self.clf['gamma']['gamma_float'])
            # C=10**(self.clf['C']) # para C discretizado
            self.clf_final = SVC(kernel=self.clf['kernel']['kf'], C=10**(self.clf['C']), gamma='scale', degree=3, probability=True)
        elif self.clf['model'] == 'KNN':   
            self.clf_final = KNeighborsClassifier(n_neighbors=int(self.clf['neig']), metric=self.clf['metric'], p=3) # p=self.clf['p']                                       
        elif self.clf['model'] == 'DTree':
            # if self.clf['min_split'] == 1.0: self.clf['min_split'] += 1
            # max_depth = self.clf['max_depth'] if self.clf['max_depth'] is None else int(self.clf['max_depth']['max_depth_int'])
            # min_samples_split = self.clf['min_split'] # math.ceil(self.clf['min_split']), # profundidade maxima da arvore - representa a poda;
            self.clf_final = DecisionTreeClassifier(criterion=self.clf['crit'], random_state=0, max_depth=None, min_samples_split=2)       
        elif self.clf['model'] == 'MLP':   
            self.clf_final = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4, 
                                           learning_rate_init=10**self.clf['eta'], activation=self.clf['activ']['af'],  
                                           learning_rate='constant', # solver=self.clf['mlp_solver'], alpha=10**self.clf['alpha'],
                                           hidden_layer_sizes=(int(self.clf['n_neurons']), int(self.clf['n_hidden']))) 
        
        while (self.tmax-self.tmin)<1: self.tmax+=0.5
        smin = math.floor(self.tmin * self.fs)
        smax = math.floor(self.tmax * self.fs)
        # print(smax-smin)
        self.res_freq = self.fs/(smax-smin) # rf=Fs/Q
        self.dft_size = 2/self.res_freq # 2=sen/cos complexo fft
        
        self.epochs, self.labels = extractEpochs(self.data, self.events, smin, smax, self.class_ids)
        self.epochs = nanCleaner(self.epochs)
        # print(self.epochs.shape)
        # self.epochs = np.asarray([ nanCleaner(ep) for ep in self.epochs ])
        
        if self.crossval:
            self.cross_scores, self.cross_kappa = [], []
            kf = StratifiedShuffleSplit(self.nfolds, test_size=self.test_perc, random_state=42)
            # kf = StratifiedKFold(self.nfolds, False)
            # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
            # self.cross_scores = cross_val_score(self.chain, XF, self.labels, cv=kf)
            for idx_treino, idx_teste in kf.split(self.epochs, self.labels):
                XT, XV, yT, yV = self.epochs[idx_treino], self.epochs[idx_teste], self.labels[idx_treino], self.labels[idx_teste]
                # print(np.asarray(XT).shape, np.asarray(XV).shape)
                acc_fold, kappa_fold = self.classic_approach(XT, XV, yT, yV) if (self.ap['option'] == 'classic') else self.sbcsp_approach(XT, XV, yT, yV)     
                self.cross_scores.append(acc_fold) # self.cross_scores.append(self.chain.score(XV, yV))
                self.cross_kappa.append(kappa_fold)
            self.acc, self.kappa = np.mean(self.cross_scores), np.mean(self.cross_kappa)
        
        else:
            test_size = int(len(self.epochs) * self.test_perc)
            train_size = int(len(self.epochs) - test_size)
            train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
            
            epochsT, labelsT = self.epochs[:train_size], self.labels[:train_size] 
            epochsV, labelsV = self.epochs[train_size:], self.labels[train_size:]
            
            ET = [ epochsT[np.where(labelsT == i)] for i in self.class_ids ] # Extrair épocas de cada classe
            EV = [ epochsV[np.where(labelsV == i)] for i in self.class_ids ]
            
            XT = np.concatenate([ET[0],ET[1]]) # Train data classes A + B
            XV = np.concatenate([EV[0],EV[1]]) # Test data classes A + B 
            
            # XA = np.r_[ET[0], EV[0]] # class A only
            # XB = np.r_[ET[1], EV[1]] # class B only           
            # if self.split == 'common':
            #     XT = np.concatenate([ET[0],ET[1]]) # Train data classes A + B
            #     XV = np.concatenate([EV[0],EV[1]]) # Test data classes A + B          
            # if self.split == 'as_train':
            #     XT = np.r_[XA[:58], XB[:58]]
            #     XV = np.r_[XA[58:86], XB[58:86]]
            # if self.split == 'as_test': 
            #     XT = np.r_[XA[:58], XB[:58]]
            #     XV = np.r_[XA[86:], XB[86:]]
                  
            # print(np.asarray(XT).shape, np.asarray(XV).shape)
            yT = np.concatenate([self.class_ids[0] * np.ones(int(len(XT)/2)), self.class_ids[1] * np.ones(int(len(XT)/2))])
            yV = np.concatenate([self.class_ids[0] * np.ones(int(len(XV)/2)), self.class_ids[1] * np.ones(int(len(XV)/2))])
            self.acc, self.kappa = self.classic_approach(XT, XV, yT, yV) if (self.ap['option'] == 'classic') else self.sbcsp_approach(XT, XV, yT, yV)
    
    def classic_approach(self, XT, XV, yT, yV):
        self.filt = Filter(self.f_low, self.f_high, self.fs, self.filt_info)
        XTF = self.filt.apply_filter(XT)
        XVF = self.filt.apply_filter(XV)
        if self.filt_info['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
            bmin = round(self.f_low * self.dft_size)
            bmax = round(self.f_high * self.dft_size)
            XTF = XTF[:, :, bmin:bmax]
            XVF = XVF[:, :, bmin:bmax]
        # print('teste')
        self.csp = CSP(n_components=int(self.ncomp))
        # self.csp = mne.decoding.CSP(n_components=int(self.ncomp))
        
        # # Option 1:
        self.csp.fit(XTF, yT)
        # print(self.csp.filters_.shape)
        self.csp_filters = self.csp.filters_
        XT_CSP = self.csp.transform(XTF)
        XV_CSP = self.csp.transform(XVF) 
        self.clf_final.fit(XT_CSP, yT)
        self.scores = self.clf_final.predict(XV_CSP)
        
        # # Option 2:
        # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
        # self.chain.fit(XT, yT)
        # self.csp_filters = self.chain['CSP'].filters_
        # self.scores = self.chain.predict(XV)
        
        acc = np.mean(self.scores == yV)     
        kappa = cohen_kappa_score(self.scores, yV)
        return acc, kappa
        
    
    def sbcsp_approach(self, XT, XV, yT, yV):
        nbands = int(self.ap['nbands'])
        if nbands > (self.f_high - self.f_low): nbands = (self.f_high - self.f_low)
        # print(nbands)
        
        n_bins = self.f_high - self.f_low
        # overlap =  if self.overlap else 1
        if self.overlap: 
            step = n_bins/(nbands+1)
            size = step/0.5 # overlap=0.5
        else:
            step = n_bins/nbands
            size = step
        
        sub_bands, bins = [], []        
        for i in range(nbands):
            fl_sb = i * step + self.f_low
            fh_sb = i * step + size + self.f_low
            # if fh_sb <= self.f_high: sub_bands.append([fl_sb, fh_sb]) # extrapola limite superior 1: descarta última sub-banda 
            # if fh_sb > self.f_high: fh_sb = self.f_high # extrapola limite superior 2: ajusta f_high ao limite
            sub_bands.append([fl_sb, fh_sb])
        # print(sub_bands)
        nbands = len(sub_bands)
        
        XTF, XVF = [], []
        if self.filt_info['design'] == 'DFT':
            self.filt = Filter(self.f_low, self.f_high, self.fs, self.filt_info)
            XT_FFT = self.filt.apply_filter(XT)
            XV_FFT = self.filt.apply_filter(XV)
            for i in range(nbands):
                bmin = round(sub_bands[i][0] * self.dft_size)
                bmax = round(sub_bands[i][1] * self.dft_size)
                XTF.append(XT_FFT[:, :, bmin:bmax])
                XVF.append(XV_FFT[:, :, bmin:bmax])
                bins.append([bmin,bmax])
            # print(bins)
        elif self.filt_info['design'] in ['IIR' or 'FIR']:
            for i in range(nbands):
                filt = Filter(sub_bands[i][0], sub_bands[i][1], self.fs, self.filt_info)
                XTF.append(filt.apply_filter(XT))
                XVF.append(filt.apply_filter(XV))
        
        # # Option 1:
        self.chain = [ Pipeline([('CSP', CSP(n_components=int(self.ncomp))), ('LDA', LDA())]) for i in range(nbands) ]
        # self.chain = [ Pipeline([('CSP', CSP(n_components=self.csp_list[i])), ('LDA', LDA())]) for i in range(nbands) ] # uncomment to tuning ncsp
        for i in range(nbands): self.chain[i]['CSP'].fit(XTF[i], yT)  
        XT_CSP = [ self.chain[i]['CSP'].transform(XTF[i]) for i in range(nbands) ]
        XV_CSP = [ self.chain[i]['CSP'].transform(XVF[i]) for i in range(nbands) ]
        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        for i in range(nbands): 
            self.chain[i]['LDA'].fit(XT_CSP[i], yT)
            SCORE_T[:, i] = np.ravel(self.chain[i]['LDA'].transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(self.chain[i]['LDA'].transform(XV_CSP[i]))
        csp_filters_sblist = [ self.chain[i]['CSP'].filters_ for i in range(nbands) ]
        lda_sblist = [ self.chain[i]['LDA'] for i in range(nbands) ] 
        
        # # Option 2:
        # SCORE_T = np.zeros((len(XT), nbands))
        # SCORE_V = np.zeros((len(XV), nbands))
        # self.csp_filters_sblist = []
        # self.lda_sblist = []
        # for i in range(nbands):
        #     self.chain = Pipeline([('CSP', CSP(n_components=self.ncomp)), ('LDA', LDA()) ])
        #     self.chain['CSP'].fit(XTF, yT)
        #     XT_CSP = self.chain['CSP'].transform(XTF)
        #     XV_CSP = self.chain['CSP'].transform(XVF)
        #     self.chain['LDA'].fit(XT_CSP, yT)
        #     SCORE_T[:, i] = np.ravel(self.chain['LDA'].transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
        #     SCORE_V[:, i] = np.ravel(self.chain['LDA'].transform(XV_CSP))
        #     self.csp_filters_sblist.append(self.chain['CSP'].filters_)
        #     self.lda_sblist.append(self.chain['LDA'])
        
        SCORE_T0 = SCORE_T[yT == self.class_ids[0], :]
        SCORE_T1 = SCORE_T[yT == self.class_ids[1], :]
        self.p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
        self.p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
        META_SCORE_T = np.log(self.p0.pdf(SCORE_T) / self.p1.pdf(SCORE_T))
        META_SCORE_V = np.log(self.p0.pdf(SCORE_V) / self.p1.pdf(SCORE_V))
        self.clf_final.fit(META_SCORE_T, yT)
        self.scores = self.clf_final.predict(META_SCORE_V)
        
        self.classifier = {'csp_filt':csp_filters_sblist, 'lda':lda_sblist, 'p0':self.p0, 'p1':self.p1, 'clf_final':self.clf_final}
        
        acc = np.mean(self.scores == yV)
        kappa = cohen_kappa_score(self.scores, yV)
        return acc, kappa
