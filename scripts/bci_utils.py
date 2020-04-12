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
import matplotlib.pyplot as plt
from time import time, sleep
from sklearn.svm import SVC
from scipy.io import loadmat
from scipy.stats import norm
from datetime import datetime
from scipy.fftpack import fft
from scipy.linalg import eigh
from sklearn.pipeline import Pipeline
from scipy.signal import lfilter, butter, filtfilt, firwin, iirfilter, decimate
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import cohen_kappa_score 

np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)

def labeling(path=None, ds=None, session=None, subj=None, n_channels=None, prefix=None):
    
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
        
        # mne.set_log_level(50, 50)
        raw = mne.io.read_raw_gdf(path + '/A0' + str(subj) + session + '.gdf').load_data()
        d = raw.get_data()[:22] # [channels x samples]
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
            trues = np.ravel(loadmat(path + '/true_labels/A0' + str(subj) + 'E.mat' )['classlabel'])
            e = np.delete(e,np.where(e[:,1]==5), axis=0) # remove eyes closed
            e = np.delete(e,np.where(e[:,1]==6), axis=0) # remove eyes open
            e = np.delete(e,np.where(e[:,1]==7), axis=0) # remove start of a new run/segment
            e[:,1] = np.where(e[:,1]==2, 0, e[:,1]) # start trial t=0
            e[np.where(e[:,1]==4),1] = trues # change unknown value labels(4) to value in [1,2,3,4]
            
        for i in range(0, len(e)):
            if e[i,1]==0: e[i,1] = (e[i+1,1]+10) # labeling start trial [11 a 14] according cue [1,2,3,4]
        return d, e


def create_omi(suj, ds_name, path, channels):
    path = path + ds_name
    
    if ds_name == 'III3a':
        
        """ 3 sujeitos (K3, K6, L1) | 4 classes | 60 canais | Fs 250Hz
            K3->(360 trials (90 por classe)) - 2 sessões
            K6,L1->(240 trials (60 por classe)) - 2 sessões 
            startTrial=0; beep/cross=2; startCue=3; startMI=4; endMI=7; endTrial(break)=10    
        """
        
        """ Dataset description/Meta-info MNE (Linux) (by vboas):
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
        mne.set_log_level('WARNING','DEBUG')
        # raw = mne.io.read_raw_gdf(path + '/gdf/K3.gdf')
        raw = mne.io.read_raw_gdf(path + '/gdf/' + suj + '.gdf')
        raw.load_data()
        data = raw.get_data() # [channels x samples]
        data = data[:channels]
        # data = corrigeNaN(data) # Correção de NaN nos dados brutos
        events_raw = raw.find_edf_events()
        ev = np.delete(events_raw[0],1,axis=1) # elimina coluna de zeros
        truelabels = np.ravel(pd.read_csv(path + '/gdf/true_labels/trues_' + suj + '.csv'))
           
        cond = False
        for i in [1, 2, 3]: cond += (ev[:,1] == i)
        idx = np.where(cond)[0]
        ev = np.delete(ev, idx, axis=0)
        
        ev[:,1] = np.where(ev[:,1]==4, 0, ev[:,1]) # Labeling Start trial t=0
        
        idx = np.where(ev[:,1]!=0)
        ev[idx,1] = truelabels  
    
        # cond = False
        # for i in [5,6,7,8,9]: cond += (ev[:,1] == i)
        # idx = ev[np.where(cond)]
        # ev[np.where(cond),1] = truelabels
        
        info = {'fs': 250, 'class_ids': [1, 2, 3, 4], 'trial_tcue': 3.0,
                'trial_tpause': 7.0, 'trial_mi_time': 4.0, 'trials_per_class': 60,
                'eeg_channels': 90 if suj == 'K3' else 60, 'ch_labels': raw.ch_names,
                'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        
        omi_data = [ data, ev, info ]
        with open(path + '/omi/' + suj + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)
    
    
    elif ds_name == 'III4a': 
        
        """ 5 subjects | 2 classes (RH, FooT)
            Epoch distribution:
                aa : train=168 test=112  
                al : train=224 test=56
                av : train=84  test=196
                aw : train=56  test=224
                ay : train=28  test=252
            Start trial= 0; Start cue=0; Start MI= 0; End MI=3.5; End trial(break)= 5.25~5.75
        """
        
        mat = loadmat(path + '/mat/' + suj + '.mat')
        data = mat['cnt'].T # 0.1 * mat['cnt'].T # convert to uV
        pos = mat['mrk'][0][0][0][0]
        true_mat = loadmat(path + '/mat/true_labels/trues_' + suj + '.mat')
        true_y = np.ravel(true_mat['true_y']) # RH=1 Foot=2
        true_y = np.where(true_y == 2, 3, true_y) # RH=1 Foot=3
        # true_test_idx = np.ravel(true_mat['test_idx'])
        events = np.c_[pos, true_y]
        # data = corrigeNaN(data)
        # data = np.asarray([ np.nan_to_num(dt) for dt in data ])
        # data = np.asarray([ np.ravel(pd.DataFrame(dt).fillna(pd.DataFrame(dt).mean())) for dt in data ])
        info = {'fs': 100, 'class_ids': [1, 3], 'trial_tcue': 0,
                'trial_tpause': 4.0, 'trial_mi_time': 4.0, 'trials_per_class': 140,
                'eeg_channels': 118, 'ch_labels': mat['nfo']['clab'],
                'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        
        omi_data = [ data, events, info ]
        with open(path + '/omi/' + suj + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)
      
        
    elif ds_name == 'IV2a':
        
        """ 72 trials per classe * 2 sessions
            T = startTrial=0; cue=2; startMI=3.25; endMI=6; endTrial=7.5-8.5
        """
        """ Dataset description MNE (Linux) (by vboas): more info in http://bbci.de/competition/iv/desc_2a.pdf
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
        """
        """ Dataset description MNE (MAC & Windows) (by vboas): more info in http://bbci.de/competition/iv/desc_2a.pdf
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
        
        mne.set_log_level('WARNING','DEBUG')
        dataT, dataV, eventsT, eventsV = [],[],[],[]
        channels_labels = None
        for ds in ['T', 'E']:
            # raw = mne.io.read_raw_gdf(path + '/gdf/A01T.gdf')
            raw = mne.io.read_raw_gdf(path + '/gdf/A0' + str(suj) + ds + '.gdf')
            raw.load_data()
            data = raw.get_data() # [channels x samples]
            data = data[:channels]
            # data = self.corrigeNaN(data) # Correção de NaN nos dados brutos
            raw_events = raw.find_edf_events()
            ev = np.delete(raw_events[0], 1, axis=1) # elimina coluna de zeros do MNE
            ev = np.delete(ev,np.where(ev[:,1]==1), axis=0) # Rejected trial
            ev = np.delete(ev,np.where(ev[:,1]==3), axis=0) # Eye movements / Unknown
            ev[:,1] = np.where(ev[:,1]==2, 0, ev[:,1]) # Start trial t=0
            if ds=='T':
                ev[:,1] = np.where(ev[:,1]==4, 1, ev[:,1]) # LH (classe 1) 
                ev[:,1] = np.where(ev[:,1]==5, 2, ev[:,1]) # RH (classe 2) 
                ev[:,1] = np.where(ev[:,1]==6, 3, ev[:,1]) # Foot (classe 3)
                ev[:,1] = np.where(ev[:,1]==7, 4, ev[:,1]) # Tongue (classe 4) 
                ev = np.delete(ev,np.where(ev[:,1]==8), axis=0) # Idling EEG (eyes closed) 
                ev = np.delete(ev,np.where(ev[:,1]==9), axis=0) # Idling EEG (eyes open) 
                ev = np.delete(ev,np.where(ev[:,1]==10), axis=0) # Start of a new run/segment (after a break)
                dataT, eventsT = data, ev
            else:
                ev = np.delete(ev,np.where(ev[:,1]==5), axis=0) # Idling EEG (eyes closed) 
                ev = np.delete(ev,np.where(ev[:,1]==6), axis=0) # Idling EEG (eyes open)
                ev = np.delete(ev,np.where(ev[:,1]==7), axis=0) # Start of a new run/segment (after a break)
                trues = np.ravel(loadmat(path + '/gdf/true_labels/A0' + str(suj) + 'E.mat' )['classlabel']) # Loading true labels to use in evaluate files (E)
                ev[np.where(ev[:,1]==4),1] = trues # muda padrão dos rotulos desconhecidos de 4 para [1,2,3,4]
                dataV, eventsV = data, ev
            channels_labels = raw.ch_names
        all_data = np.c_[dataT, dataV]    
        new_events = np.copy(eventsV)
        new_events[:,0] += len(dataT.T) # eventsV pos + last dataT pos (eventsT is continued by eventsV)
        all_events = np.r_[eventsT, new_events]
        info = {'fs': 250, 'class_ids': [1, 2, 3, 4], 'trial_tcue': 2.0,
                'trial_tpause': 6.0, 'trial_mi_time': 4.0, 'trials_per_class': 144,
                'eeg_channels': 22, 'ch_labels': channels_labels,
                'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        omi_data = [ all_data, all_events, info ]
        with open(path + '/omi/A0' + str(suj) + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)
    
    
    elif ds_name == 'IV2b':
        
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
        mne.set_log_level('WARNING','DEBUG')
        DT, EV = [], []
        for ds in ['01T','02T','03T','04E','05E']:
            # raw = mne.io.read_raw_gdf(path + '/gdf/B0101T.gdf')
            raw = mne.io.read_raw_gdf(path + '/gdf/B0' + str(suj) + ds + '.gdf')
            raw.load_data()
            data = raw.get_data() # [channels x samples]
            data = data[:channels]
            # data = corrigeNaN(data) # Correção de NaN nos dados brutos
            ev = raw.find_edf_events()
            ev = np.delete(ev[0],1,axis=1) # elimina coluna de zeros
            truelabels = np.ravel(loadmat(path + '/gdf/true_labels/B0' + str(suj) + ds + '.mat')['classlabel'])
    
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
            
            DT.append(data)
            EV.append(ev)
            
        # Save a unique npy file with all datasets
        soma = 0
        for i in range(1,len(EV)): 
            soma += len(DT[i-1].T)
            EV[i][:,0] += soma
            
        all_data = np.c_[DT[0],DT[1],DT[2],DT[3],DT[4]]
        all_events = np.r_[EV[0],EV[1],EV[2],EV[3],EV[4]]
        
        info = {'fs': 250, 'class_ids': [1, 2], 'trial_tcue': 3.0,
                'trial_tpause': 7.0, 'trial_mi_time': 4.0, 'trials_per_class': 360,
                'eeg_channels': 3, 'ch_labels': {'EEG1':'C3', 'EEG2':'Cz', 'EEG3':'C4'},
                'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        
        omi_data = [ all_data, all_events, info ]
        with open(path + '/omi/B0' + str(suj) + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)
   

    elif ds_name == 'LEE54':
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
        fs = 1000
        reduced = False  # True if generate data only session 2
        downsampling = True
        
        suj_in = str(suj) if suj>=10 else ('0' + str(suj))
    
        mat1 = loadmat(path + '/session1/sess01_subj' + suj_in + '_EEG_MI.mat')
        mat2 = loadmat(path + '/session2/sess02_subj' + suj_in + '_EEG_MI.mat')
        
        TRAIN1 = mat1['EEG_MI_train']
        TEST1 = mat1['EEG_MI_test']
        TRAIN2 = mat2['EEG_MI_train']
        TEST2 = mat2['EEG_MI_test']
        
        dataT1 = TRAIN1['x'][0,0].T
        dataV1 = TEST1['x'][0,0].T
        dataT2 = TRAIN2['x'][0,0].T
        dataV2 = TEST2['x'][0,0].T
        
        eventsT1 = np.r_[ TRAIN1['t'][0,0], TRAIN1['y_dec'][0,0] ].T
        eventsV1 = np.r_[ TEST1['t'][0,0], TEST1['y_dec'][0,0] ].T
        eventsT2 = np.r_[ TRAIN2['t'][0,0], TRAIN2['y_dec'][0,0] ].T
        eventsV2 = np.r_[ TEST2['t'][0,0], TEST2['y_dec'][0,0] ].T
        
        ## DOWNSAMPLING
        if downsampling:
            factor = 4
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
            fs = fs/factor
        
        eventsT1[:, 1] = np.where(eventsT1[:, 1]==2, 1, 2) # troca class_ids 1=LH, 2=RH
        eventsV1[:, 1] = np.where(eventsV1[:, 1]==2, 1, 2)
        eventsT2[:, 1] = np.where(eventsT2[:, 1]==2, 1, 2) # troca class_ids 1=LH, 2=RH
        eventsV2[:, 1] = np.where(eventsV2[:, 1]==2, 1, 2)
        
        # epochsT1 = TRAIN1['smt'][0,0]
        # epochsV1 = TEST1['smt'][0,0]
        # epochsT2 = TRAIN2['smt'][0,0]
        # epochsV2 = TEST2['smt'][0,0]
        
        if reduced:
            
            cortex_ch = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
            ch_labels = ['FC5', 'FC3', 'FC1', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6']
            
            all_data = np.c_[dataT1, dataV1]
            all_data = all_data[cortex_ch]
            
            eventsV1[:,0] += dataT1.shape[-1]
            all_events = np.r_[eventsT1, eventsV1]
            
            info = {'fs': fs, 'class_ids': [1, 2], 'trial_tcue': 3.0,
                    'trial_tpause': 7.0, 'trial_mi_time': 4.0, 'trials_per_class': 100,
                    'eeg_channels': 20, 'ch_labels': ch_labels,
                    'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
            
            omi_data = [ all_data, all_events, info ]
            with open(path + '/omi_s1_cortex/subj' + str(suj) + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)
        
        else:
            
            ch_labels = list(['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4',
                  'T8','TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz',
                  'O2','PO10','FC3','FC4','C5','C1','C2','C6','CP3','CPz','CP4','P1','P2','POz','FT9',
                  'FTT9h','TTP7h','TP7','TPP9h','FT10','FTT10h','TPP8h','TP8','TPP10h','F9','F10','AF7',
                  'AF3','AF4','AF8','PO3','PO4'])
            
            all_data = np.c_[dataT1, dataT2, dataV1, dataV2]  
            
            # opção 1
            eventsT2[:,0] += dataT1.shape[-1] # eventsT2 pos + last dataT1 pos (eventsT2 is continued by eventsT1) ...
            eventsV1[:,0] += (dataT1.shape[-1] + dataT2.shape[-1])
            eventsV2[:,0] += (dataT1.shape[-1] + dataT2.shape[-1] + dataV1.shape[-1])
            all_events = np.r_[eventsT1, eventsT2, eventsV1, eventsV2]
            
            # opção 2
            # DT = [dataT1, dataT2, dataV1, dataV2]
            # EV = [eventsT1, eventsT2, eventsV1, eventsV2]
            # soma = 0
            # for i in range(1,len(EV)): 
            #     soma += len(DT[i-1].T)
            #     EV[i][:,0] += soma
            # all_events = np.r_[EV[0],EV[1],EV[2],EV[3]]
            
            # DOWNSAMPLING Option 2
            # factor = 10
            # # all_data = np.asarray([ all_data[:,i] for i in range(0, all_data.shape[-1], factor) ]).T
            # all_data = decimate(all_data, factor)
            # all_events[:, 0] = [ round(all_events[i, 0]/factor) for i in range(all_events.shape[0]) ]
            # fs = fs/factor
            
            info = {'fs': fs, 'class_ids': [1, 2], 'trial_tcue': 3.0,
                    'trial_tpause': 7.0, 'trial_mi_time': 4.0, 'trials_per_class': 200,
                    'eeg_channels': 62, 'ch_labels': ch_labels,
                    'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
            
            omi_data = [ all_data, all_events, info ]
            with open(path + '/omi/subj' + str(suj) + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)
            
    
    elif ds_name == 'TWL':
        """ 2 subjects (TL, WL) | 2 classes (lh, rh) | Fs 250Hz
            40 trials (20 per class) - TL: 2 sessions; WL:3 sessions
            8 channels (1=Cz 2=Cpz 3=C1 4=C3 5=CP3 6=C2 7=C4 8=CP4)
            Scalp map:      C3  C1  Cz  C2  CP4     4  3  1  6  7
                               CP3   CPz  CP4 		  5   2   8
            Start trial=0; Wait beep=2; Start cue=3; Start MI=4.25; End MI=8; End trial(break)=10-12
        """
        """ Dataset description/Meta-info MNE (Linux) (by vboas):
            1=Cross on screen (BCI experiment) 
            2=Feedback (continuous) - onset (BCI experiment)
            3=768 Start of Trial, Trigger at t=0s
            4=783 Unknown
            5=769 class1, Left hand - cue onset (BCI experiment)
            6=770 class2, Right hand - cue onset (BCI experiment)
        """
        mne.set_log_level('WARNING','DEBUG')
        dataT, dataV, eventsT, eventsV = [],[],[],[]
        for ds in ['S1','S2']:
            raw = mne.io.read_raw_gdf(path + '/gdf/' + suj + '_' + ds + '.gdf')
            raw.load_data()
            data = raw.get_data() # [channels x samples]
            data = data[:channels]
            # data = corrigeNaN(data) # Correção de NaN nos dados brutos
            events_raw = raw.find_edf_events()
            ev = np.delete(events_raw[0],1,axis=1) # elimina coluna de zeros
            
            ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # elimina marcações inuteis (cross on screen)
            ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # elimina marcações inuteis (feedback continuous)
            ev = np.delete(ev,np.where(ev[:,1]==4),axis=0) # elimina marcações inuteis (unknown)
            ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1])
            ev[:,1] = np.where(ev[:,1]==5, 1, ev[:,1]) # altera label lh de 5 para 1
            ev[:,1] = np.where(ev[:,1]==6, 2, ev[:,1]) # altera label rh de 6 para 2
            
            if ds == 'S1': dataT, eventsT = data, ev
            else: dataV, eventsV = data, ev
        
        all_data = np.c_[dataT, dataV]    
        new_events = np.copy(eventsV)
        new_events[:,0] += len(dataT.T) # eventsV pos + last dataT pos (eventsT is continued by eventsV)
        all_events = np.r_[eventsT, new_events]
        
        info = {'fs': 250, 'class_ids': [1, 2], 'trial_tcue': 3.0,
                'trial_tpause': 8.0, 'trial_mi_time': 5.0, 'trials_per_class': 20,
                'eeg_channels': 8, 'ch_labels': {'EEG1':'Cz', 'EEG2':'Cpz', 'EEG3':'C1', 'EEG4':'C3', 'EEG5':'CP3', 'EEG6':'C2', 'EEG7':'C4', 'EEG8':'CP4'},
                'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        omi_data = [ all_data, all_events, info ]
        with open(path + '/omi/' + suj + '.omi', 'wb') as handle: pickle.dump(omi_data, handle)
        
    
    elif ds_name == 'CL':
        """ 1 subject (CL) | 3 classes (lh, rh, foot) | 16 channels | Fs 125Hz
            lh-rh -> 100 trials (50 per class) 5*20 - 1 session
            lh-ft -> 48 trials (24 per class) 3*16 - 1 session
            Start trial=0; Beep=1; Wait=2; Start cue=2; Start MI=3; End MI=9; End trial(break)=14
        """
        data = np.load(path + '/original/orig_' + suj + '_LF_data.npy').T
        events = np.load(path + '/original/orig_' + suj + '_LF_events.npy').astype(int)
        events[:,1] = np.where(events[:,1] == 2, 3, events[:,1]) # LH=1, FooT=3
        
        info = {'fs': 125, 'class_ids': [1, 3], 'trial_tcue': 2.0,
                'trial_tpause': 10.0, 'trial_mi_time': 8.0, 'trials_per_class': 24,
                'eeg_channels': 16, 'ch_labels': None,
                'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        omi_data = [ data, events, info ]
        with open(path + '/omi/' + suj + '_LF.omi', 'wb') as handle: pickle.dump(omi_data, handle)
        
        data = np.load(path + '/original/orig_' + suj + '_LR_data.npy').T
        events = np.load(path + '/original/orig_' + suj + '_LR_events.npy').astype(int)
        info = {'fs': 125, 'class_ids': [1, 2], 'trial_tcue': 2.0,
                'trial_tpause': 9.0, 'trial_mi_time': 7.0, 'trials_per_class': 50,
                'eeg_channels': 16, 'ch_labels': None,
                'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}
        omi_data = [ data, events, info ]
        with open(path + '/omi/' + suj + '_LR.omi', 'wb') as handle: pickle.dump(omi_data, handle)
    

def create_npy(suj, ds_name, path, channels):
    path = path + ds_name
    
    if ds_name == 'III3a':
        mne.set_log_level('WARNING','DEBUG')
        raw = mne.io.read_raw_gdf(path + '/gdf/' + suj + '.gdf')
        raw.load_data()
        data = raw.get_data() # [channels x samples]
        # data = data[:channels]
        # data = corrigeNaN(data) # Correção de NaN nos dados brutos
        events_raw = raw.find_edf_events()
        events = np.delete(events_raw[0],1,axis=1) # elimina coluna de zeros
        truelabels = np.ravel(pd.read_csv(path + '/gdf/true_labels/trues_' + suj + '.csv'))
        events = iii3a_labeling(events, suj, truelabels) # Labeling correctly the events like competition description
        np.save(path + '/npy/' + suj + '_data', data)
        np.save(path + '/npy/' + suj + '_events', events)
    
    
    elif ds_name == 'III4a': 
        mat = loadmat(path + '/mat/' + suj + '.mat')
        data = mat['cnt'].T # 0.1 * mat['cnt'].T # convert to uV
        pos = mat['mrk'][0][0][0][0]
        true_mat = loadmat(path + '/mat/true_labels/trues_' + suj + '.mat')
        true_y = np.ravel(true_mat['true_y']) # RH=1 Foot=2
        true_y = np.where(true_y == 2, 3, true_y) # RH=1 Foot=3
        # true_test_idx = np.ravel(true_mat['test_idx'])
        events = np.c_[pos, true_y]
        # data = corrigeNaN(data)
        # data = np.asarray([ np.nan_to_num(dt) for dt in data ])
        # data = np.asarray([ np.ravel(pd.DataFrame(dt).fillna(pd.DataFrame(dt).mean())) for dt in data ])        
        np.save(path + '/npy/' + suj + '_data', data)
        np.save(path + '/npy/' + suj + '_events', events)  
      
        
    elif ds_name == 'IV2a':
        mne.set_log_level('WARNING','DEBUG')
        dataT, dataV, eventsT, eventsV = [],[],[],[]
        for ds in ['T', 'E']:
            raw = mne.io.read_raw_gdf(path + '/gdf/A0' + str(suj) + ds + '.gdf')
            raw.load_data()
            data = raw.get_data() # [channels x samples]
            # data = data[:channels]
            # data = self.corrigeNaN(data) # Correção de NaN nos dados brutos
            raw_events = raw.find_edf_events()
            ev = np.delete(raw_events[0], 1, axis=1) # elimina coluna de zeros do MNE
            ev = np.delete(ev,np.where(ev[:,1]==1), axis=0) # Rejected trial
            ev = np.delete(ev,np.where(ev[:,1]==3), axis=0) # Eye movements / Unknown
            ev[:,1] = np.where(ev[:,1]==2, 0, ev[:,1]) # Start trial t=0
            if ds=='T':
                ev = np.delete(ev,np.where(ev[:,1]==8), axis=0) # Idling EEG (eyes closed) 
                ev = np.delete(ev,np.where(ev[:,1]==9), axis=0) # Idling EEG (eyes open) 
                ev = np.delete(ev,np.where(ev[:,1]==10), axis=0) # Start of a new run/segment (after a break)
                ev[:,1] = np.where(ev[:,1]==4, 1, ev[:,1]) # LH (classe 1) 
                ev[:,1] = np.where(ev[:,1]==5, 2, ev[:,1]) # RH (classe 2) 
                ev[:,1] = np.where(ev[:,1]==6, 3, ev[:,1]) # Foot (classe 3)
                ev[:,1] = np.where(ev[:,1]==7, 4, ev[:,1]) # Tongue (classe 4) 
                dataT, eventsT = data, ev
            else:
                ev = np.delete(ev,np.where(ev[:,1]==5), axis=0) # Idling EEG (eyes closed) 
                ev = np.delete(ev,np.where(ev[:,1]==6), axis=0) # Idling EEG (eyes open)
                ev = np.delete(ev,np.where(ev[:,1]==7), axis=0) # Start of a new run/segment (after a break)
                trues = np.ravel(loadmat(path + '/gdf/true_labels/A0' + str(suj) + 'E.mat' )['classlabel']) # Loading true labels to use in evaluate files (E)
                ev[np.where(ev[:,1]==4),1] = trues # muda padrão dos rotulos desconhecidos de 4 para [1,2,3,4]
                dataV, eventsV = data, ev
            
            
            # events[:,1] = iv2a_labeling(events[:,1], ds, suj, truelabels, 'lnx') # Labeling correctly the events like competition description ('win','mac','lnx')

    
        # Save separated npy files to each dataset type (train and train)
        # np.save(path_to_npy + str(suj) + '_dataT', dataT)
        # np.save(path_to_npy + str(suj) + '_dataV', dataV)
        # np.save(path_to_npy + str(suj) + '_eventsT', eventsT)
        # np.save(path_to_npy + str(suj) + '_eventsV', eventsV)
        
        all_data = np.c_[dataT, dataV]    
        new_events = np.copy(eventsV)
        new_events[:,0] += len(dataT.T) # eventsV pos + last dataT pos (eventsT is continued by eventsV)
        all_events = np.r_[eventsT, new_events]
        
        # Save a unique npy file with both dataset type (train + test)
        np.save(path + '/npy/A0' + str(suj) + '_data', all_data)    
        np.save(path + '/npy/A0' + str(suj) + '_events', all_events)
         
    
    elif ds_name == 'IV2b':
        mne.set_log_level('WARNING','DEBUG')
        DT, EV = [], []
        for ds in ['01T','02T','03T','04E','05E']:
            raw = mne.io.read_raw_gdf(path + '/gdf/B0' + str(suj) + ds + '.gdf')
            raw.load_data()
            data = raw.get_data() # [channels x samples]
            # data = data[:channels]
            # data = corrigeNaN(data) # Correção de NaN nos dados brutos
            events = raw.find_edf_events()
            events = np.delete(events[0],1,axis=1) # elimina coluna de zeros
            truelabels = np.ravel(loadmat(path + '/gdf/true_labels/B0' + str(suj) + ds + '.mat')['classlabel'])
            events = iv2b_labeling(events, ds, suj, truelabels, 'lnx') # Labeling correctly the events like competition description
            
            # Save separated npy files to each dataset
            # np.save(path + '/npy/B0' + str(suj) + ds + '_data', data)
            # np.save(path + '/npy/B0' + str(suj) + ds + '_events', events)
            
            DT.append(data)
            EV.append(events)
            
        # Save a unique npy file with all datasets
        soma = 0
        for i in range(1,len(EV)): 
            soma += len(DT[i-1].T)
            EV[i][:,0] += soma
            
        all_data = np.c_[DT[0],DT[1],DT[2],DT[3],DT[4]]
        all_events = np.r_[EV[0],EV[1],EV[2],EV[3],EV[4]]
        
        np.save(path + '/npy/B0' + str(suj) + '_data', all_data)    
        np.save(path + '/npy/B0' + str(suj) + '_events', all_events)
        
    
    elif ds_name == 'TWL':
        mne.set_log_level('WARNING','DEBUG')
        dataT, dataV, eventsT, eventsV = [],[],[],[]
        for ds in ['S1','S2']:
            raw = mne.io.read_raw_gdf(path + '/gdf/' + suj + '_' + ds + '.gdf')
            raw.load_data()
            data = raw.get_data() # [channels x samples]
            # data = data[:channels]
            # data = corrigeNaN(data) # Correção de NaN nos dados brutos
            events_raw = raw.find_edf_events()
            events = np.delete(events_raw[0],1,axis=1) # elimina coluna de zeros
            events = twl_labeling(events, ds) # Labeling correctly the events like competition description
            
            # Save separated npy files to each dataset type (train and train)
            # np.save(path + '/npy/' + suj + '_' + ds + '_data', data)
            # np.save(path + '/npy/' + suj + '_' + ds + '_events', events)
            
            if ds == 'S1': dataT, eventsT = data, events
            else: dataV, eventsV = data, events
        
        all_data = np.c_[dataT, dataV]    
        new_events = np.copy(eventsV)
        new_events[:,0] += len(dataT.T) # eventsV pos + last dataT pos (eventsT is continued by eventsV)
        all_events = np.r_[eventsT, new_events]
        
        np.save(path + '/npy/' + suj + '_data', all_data)    
        np.save(path + '/npy/' + suj + '_events', all_events)
        
    
    elif ds_name == 'CL':
        for cl in ['_LF', '_LR']:
            data = np.load(path + '/original/orig_' + suj + cl + '_data.npy').T
            events = np.load(path + '/original/orig_' + suj + cl + '_events.npy').astype(int)
            # data = corrigeNaN(data)
            for i in range(len(events)-1): 
                if events[i,1]==0:
                    events[i,1] = events[i+1,1]
                    events[i+1,1] = events[i+1,1] + 768
            # if suj=='CL_LF': events[:,1] = np.where(events[:,1] == 2, 3, events[:,1]) # LH=1, FooT=3
            np.save(path + '/' + suj + '_data.npy', data)
            np.save(path + '/' + suj + '_events.npy', events)
    

def iii3a_labeling(ev, suj, trueLabels):
    """ Dataset description/Meta-info MNE (Linux) (by vboas):
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
    cond = False
    for i in [1, 2, 3]: cond += (ev[:,1] == i)
    idx = np.where(cond)[0]
    ev = np.delete(ev, idx, axis=0)
    
    idx = np.where(ev[:,1]==4)
    ev[idx,1] = trueLabels  # Labeling Start trial t=0

    cond = False
    for i in [5,6,7,8,9]: cond += (ev[:,1] == i)
    idx = ev[np.where(cond)]
    ev[np.where(cond),1] = trueLabels + 768
    
    return ev


def iv2b_labeling(ev, ds, suj, trues, so='lnx'): 
    # normaliza rotulos de eventos conforme descrição oficial do dataset
    """ Dataset Description (by vboas): more info in http://bbci.de/competition/iv/desc_2b.pdf
        01T e 02T (without feedback)
    		10 trial * 2 classes * 6 runs * 2 sessions = 240 trials (120 per class)
    		Cross t=0 (per 3s)
    		beep t=2s
    		cue t=3s (per 1.25s)
    		MI t=4s (per 3s)
    		Pause t=7s (per 1.5-2.5s)
    		EndTrial t=8.5-9.5
    	03T, 04E e 05E (with feedback)
    		10 trial * 2 classes * 4 runs * 3 sessions = 240 trials (120 per class)
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
    # Remove marcações inúteis e normaliza rotulos de eventos conforme descrição oficial do dataset
    if ds in ['01T','02T']:
        for rm in range(5,12): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
        ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete rejected trials
        ev[:,1] = np.where(ev[:,1]==2, 0, ev[:,1]) # altera label start trial de 2 para 0
        ev[:,1] = np.where(ev[:,1]==3, 1, ev[:,1]) # altera label cue LH de 3 para 1
        ev[:,1] = np.where(ev[:,1]==4, 2, ev[:,1]) # altera label cue RH de 4 para 2
        for i in range(len(ev[:,1])):
            if ev[i,1]==0: ev[i,1] = ev[i+1,1] + 768
    elif ds=='03T': 
        for rm in range(6,13): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
        ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # delete rejected trials
        ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete feedback continuous
        ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1]) # altera label start trial de 3 para 0
        ev[:,1] = np.where(ev[:,1]==4, 1, ev[:,1]) # altera label cue LH de 4 para 1
        ev[:,1] = np.where(ev[:,1]==5, 2, ev[:,1]) # altera label cue RH de 5 para 2
        for i in range(len(ev[:,1])):
            if ev[i,1]==0: ev[i,1] = ev[i+1,1] + 768
    else:
        for rm in range(5,12): ev = np.delete(ev,np.where(ev[:,1]==rm),axis=0) # detele various eye movements marks
        ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # delete rejected trials
        ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # delete feedback continuous
        ev[:,1] = np.where(ev[:,1]==3, 0, ev[:,1]) # altera label start trial de 3 para 0
        ev[np.where(ev[:,1]==4),1] = trues #rotula momento da dica conforme truelabels
        for i in range(len(ev[:,1])):
            if ev[i,1]==0: ev[i,1] = ev[i+1,1] + 768
    
    return ev


def twl_labeling(ev, ds): # normaliza rotulos de eventos conforme descrição oficial do dataset
    """ Dataset description/Meta-info MNE (Linux) (by vboas):
        1=Cross on screen (BCI experiment) 
        2=Feedback (continuous) - onset (BCI experiment)
        3=768 Start of Trial, Trigger at t=0s
        4=783 Unknown
        5=769 class1, Left hand - cue onset (BCI experiment)
        6=770 class2, Right hand - cue onset (BCI experiment)
    """
    ev = np.delete(ev,np.where(ev[:,1]==1),axis=0) # elimina marcações inuteis (cross on screen)
    ev = np.delete(ev,np.where(ev[:,1]==2),axis=0) # elimina marcações inuteis (feedback continuous)
    ev = np.delete(ev,np.where(ev[:,1]==4),axis=0) # elimina marcações inuteis (unknown)
    ev[:,1] = np.where(ev[:,1]==5, 769, ev[:,1]) # altera label lh de 5 para 1
    ev[:,1] = np.where(ev[:,1]==6, 770, ev[:,1]) # altera label rh de 6 para 2
    
    for i in range(len(ev)): 
        if ev[i,1]==3: ev[i,1] = ev[i+1,1] - 768 # rotula dica conforme tarefa (idx+1=769 ou 770, idx=1 ou 2)

    return ev
 

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
            print('Incomplete epoch detected...')
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
    def __init__(self, fl, fh, buffer_len, srate, filt_info, forder=None, band_type='bandpass'):
        self.ftype = filt_info['design']
        if fl == 0: fl = 0.001
        self.nyq = 0.5 * srate
        low = fl / self.nyq
        high = fh / self.nyq
        self.res_freq = (srate / buffer_len)
        if high >= 1: high = 0.99

        if self.ftype == 'IIR':
            self.forder = filt_info['iir_order']
            # self.b, self.a = iirfilter(self.forder, [low, high], btype='band')
            self.b, self.a = butter(self.forder, [low, high], btype=band_type)
        elif self.ftype == 'FIR':
            self.forder = filt_info['fir_order']
            self.b = firwin(self.forder, [low, high], window='hamming', pass_zero=False)
            self.a = [1]
        elif self.ftype == 'DFT':
            self.bmin = int(fl / self.res_freq)  # int(fl * (srate/self.nyq)) # int(low * srate)
            self.bmax = int(fh / self.res_freq)  # int(fh * (srate/self.nyq)) # int(high * srate)


    def apply_filter(self, data_in, is_epoch=False):
        if self.ftype != 'DFT':
            # data_out = filtfilt(self.b, self.a, data_in)
            data_out = lfilter(self.b, self.a, data_in)
        else:
            if is_epoch:
                data_out = fft(data_in)
                REAL = np.real(data_out)[:, self.bmin:self.bmax].T
                IMAG = np.imag(data_out)[:, self.bmin:self.bmax].T
                data_out = np.transpose(list(itertools.chain.from_iterable(zip(IMAG, REAL))))
            else:
                data_out = fft(data_in)
                REAL = np.transpose(np.real(data_out), (2, 0, 1)) # old: np.transpose(np.real(data_out)[:, :, self.bmin:self.bmax], (2, 0, 1))
                IMAG = np.transpose(np.imag(data_out), (2, 0, 1)) # old: np.transpose(np.imag(data_out)[:, :, self.bmin:self.bmax], (2, 0, 1))
                data_out = list(itertools.chain.from_iterable(zip(IMAG, REAL)))
                data_out = np.transpose(data_out, (1, 2, 0))
                
        return data_out


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
            S0 += np.dot(Xa[epoca,:,:], Xa[epoca,:,:].T) #covA Xa[epoca]
            S1 += np.dot(Xb[epoca,:,:], Xb[epoca,:,:].T) #covB Xb[epoca]
        [D, W] = eigh(S0, S0 + S1)
        ind = np.empty(c, dtype=int)
        ind[0::2] = np.arange(c - 1, c // 2 - 1, -1) 
        ind[1::2] = np.arange(0, c // 2)
        W = W[:, ind]
        self.filters_ = W.T[:self.n_components]
        return self # instruction add because cross-validation pipeline
    def transform(self, X):        
        XT = np.asarray([np.dot(self.filters_, epoch) for epoch in X])
        XVAR = np.log(np.mean(XT ** 2, axis=2)) # Xcsp
        return XVAR


class BCI():
    
    def __init__(self, data, events, class_ids, overlap, fs, crossval, nfolds, test_perc,
                 f_low=None, f_high=None, tmin=None, tmax=None, ncomp=None, ap=None, filt_info=None, clf=None):
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
        self.ap = ap
        self.filt_info = filt_info
        self.clf = clf
        self.acc = None
        self.kappa = None
        
        
    def objective(self, args):
        # print(args)
        self.f_low, self.f_high, self.tmin, self.tmax, self.ncomp, self.ap, self.filt_info, self.clf = args
        self.f_low, self.f_high, self.ncomp = int(self.f_low), int(self.f_high), int(self.ncomp)
        while (self.tmax-self.tmin)<1: self.tmax+=0.5 # garante janela minima de 1seg
        # self.acc, self.kappa = self.evaluate()
        self.evaluate()
        return self.acc * (-1)  
    
    
    def evaluate(self):
        
        if self.clf['model'] == 'LDA': 
            # if clf_dict['lda_solver'] == 'svd': lda_shrinkage = None
            # else:
            #     lda_shrinkage = self.clf['shrinkage'] if self.clf['shrinkage'] in [None,'auto'] else self.clf['shrinkage']['shrinkage_float']
            self.clf_final = LDA(solver=self.clf['lda_solver'], shrinkage=None)
        
        if self.clf['model'] == 'Bayes': 
            self.clf_final = GaussianNB()
        
        if self.clf['model'] == 'SVM': 
            # degree = self.clf['kernel']['degree'] if self.clf['kernel']['kf'] == 'poly' else 3
            # gamma = self.clf['gamma'] if self.clf['gamma'] in ['scale', 'auto'] else 10 ** (self.clf['gamma']['gamma_float'])
            self.clf_final = SVC(kernel=self.clf['kernel']['kf'], C=10 ** (self.clf['C']), 
                                 gamma='scale', degree=3, probability=True) # degree=3,
        
        if self.clf['model'] == 'KNN':   
            self.clf_final = KNeighborsClassifier(n_neighbors=int(self.clf['neig']), 
                                                  metric=self.clf['metric'], p=3) # p=self.clf['p'] #metric='minkowski', p=3)  # minkowski,p=2 -> distancia euclidiana padrão
                                                  
        if self.clf['model'] == 'DTree': 
            # print(self.clf['min_split'])
            # if self.clf['min_split'] == 1.0: self.clf['min_split'] += 1
            # max_depth = self.clf['max_depth'] if self.clf['max_depth'] is None else int(self.clf['max_depth']['max_depth_int'])
            
            self.clf_final = DecisionTreeClassifier(criterion=self.clf['crit'], random_state=0,
                                                    max_depth=None, # max_depth=max_depth,
                                                    min_samples_split=2 # min_samples_split=self.clf['min_split'], #math.ceil(self.clf['min_split']),
                                                    ) # None (profundidade maxima da arvore - representa a pode); ENTROPIA = medir a pureza e a impureza dos dados
                
        if self.clf['model'] == 'MLP':   
            self.clf_final = MLPClassifier(verbose=False, max_iter=10000, tol=1e-4,
                                           learning_rate_init=10**self.clf['eta'],
                                           # alpha=10**self.clf['alpha'],
                                           activation=self.clf['activ']['af'],
                                           hidden_layer_sizes=(int(self.clf['n_neurons']), int(self.clf['n_hidden'])),
                                           learning_rate='constant', # self.clf['eta_type'], 
                                           solver=self.clf['mlp_solver'],
                                           )
        
        # cortex_ch = [7, 32, 8, 9, 33, 10, 34, 12, 35, 13, 36, 14, 37, 17, 38, 18, 39, 19, 40, 20]
        # self.data = self.data[cortex_ch]
        
        smin = math.floor(self.tmin * self.fs)
        smax = math.floor(self.tmax * self.fs)
        self.buffer_len = smax - smin
        
        self.dft_rf = self.fs/self.buffer_len # resolução em frequência fft
        self.dft_size_band = round(2/self.dft_rf) # 2 representa sen e cos que foram separados do componente complexo da fft intercalados
        
        self.epochs, self.labels = extractEpochs(self.data, self.events, smin, smax, self.class_ids)
        self.epochs = nanCleaner(self.epochs)
        # self.epochs = np.asarray([ nanCleaner(ep) for ep in self.epochs ])
        
        # self.epochs, self.labels = self.epochs[:int(len(self.epochs)/2)], self.labels[:int(len(self.labels)/2)] # Lee19 sessão 1
        # self.epochs, self.labels = self.epochs[int(len(self.epochs)/2):], self.labels[int(len(self.labels)/2):] # Lee19 somente sessão 2
        
        self.filt = Filter(self.f_low, self.f_high, self.buffer_len, self.fs, self.filt_info)
        
        self.csp = CSP(n_components=self.ncomp)
            
        if self.crossval:
            
            self.cross_scores = []
            self.cross_kappa = []
            
            kf = StratifiedShuffleSplit(self.nfolds, test_size=self.test_perc, random_state=42)
            #kf = StratifiedKFold(self.nfolds, False)
            
            if self.ap['option'] == 'classic':
                
                XF = self.filt.apply_filter(self.epochs)
        
                if self.filt_info['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
                    bmin = self.f_low * self.dft_size_band
                    bmax = self.f_high * self.dft_size_band
                    XF = XF[:, :, bmin:bmax]
                    # print(bmin, bmax)
                
                # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
                # self.cross_scores = cross_val_score(self.chain, XF, self.labels, cv=kf)
                
                for idx_treino, idx_teste in kf.split(XF, self.labels):
                    XT, XV, yT, yV = XF[idx_treino], XF[idx_teste], self.labels[idx_treino], self.labels[idx_teste]
                    
                    
                    # Option 1
                    self.csp.fit(XT, yT)
                    XT_CSP = self.csp.transform(XT)
                    XV_CSP = self.csp.transform(XV) 
                    self.clf_final.fit(XT_CSP, yT)
                    self.scores = self.clf_final.predict(XV_CSP)
                    # self.csp_filters = self.csp.filters_
                    
                    # Option 2
                    # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
                    # self.chain.fit(XT, yT)
                    # self.scores = self.chain.predict(XV)
                    # self.csp_filters = self.chain['CSP'].filters_
                    
                    acc_fold = np.mean(self.scores == yV) # or self.cross_scores.append(self.chain.score(XV, yV)) 
                    kappa_fold = cohen_kappa_score(self.scores, yV)     
                    self.cross_scores.append(acc_fold)
                    self.cross_kappa.append(kappa_fold)
                
            
            elif self.ap['option'] == 'sbcsp':
                
                for idx_treino, idx_teste in kf.split(self.epochs, self.labels):
                    acc_fold, kappa_fold = self.sbcsp_approach(self.epochs[idx_treino], self.epochs[idx_teste], 
                                                               self.labels[idx_treino], self.labels[idx_teste])
                    self.cross_scores.append(acc_fold)
                    self.cross_kappa.append(kappa_fold)
            
            self.acc = np.mean(self.cross_scores) 
            self.kappa = np.mean(self.cross_kappa) 
            
            
        else: # NO crossval
            
            test_size = int(len(self.epochs) * self.test_perc)
            train_size = int(len(self.epochs) - test_size)
            train_size = train_size if (train_size % 2 == 0) else train_size - 1 # garantir balanço entre as classes (amostragem estratificada)
            epochsT, labelsT = self.epochs[:train_size], self.labels[:train_size] 
            epochsV, labelsV = self.epochs[train_size:], self.labels[train_size:]
            
            XT = [ epochsT[np.where(labelsT == i)] for i in self.class_ids ] # Extrair épocas de cada classe
            XV = [ epochsV[np.where(labelsV == i)] for i in self.class_ids ]
            
            XT = np.concatenate([XT[0],XT[1]]) # Train data classes A + B
            XV = np.concatenate([XV[0],XV[1]]) # Test data classes A + B        
            yT = np.concatenate([self.class_ids[0] * np.ones(int(len(XT)/2)), self.class_ids[1] * np.ones(int(len(XT)/2))])
            yV = np.concatenate([self.class_ids[0] * np.ones(int(len(XV)/2)), self.class_ids[1] * np.ones(int(len(XV)/2))])
            # print(XT.shape, XV.shape)
           
            if self.ap['option'] == 'classic':
                
                XTF = self.filt.apply_filter(XT)
                XVF = self.filt.apply_filter(XV)
                
                if self.filt_info['design'] == 'DFT': # extrai somente os bins referentes à banda de interesse
                    bmin = self.f_low * self.dft_size_band
                    bmax = self.f_high * self.dft_size_band
                    XTF = XTF[:, :, bmin:bmax]
                    XVF = XVF[:, :, bmin:bmax]
                    # print(bmin, bmax)
                
                # Option 1
                self.csp.fit(XTF, yT)
                XT_CSP = self.csp.transform(XTF)
                XV_CSP = self.csp.transform(XVF) 
                self.clf_final.fit(XT_CSP, yT)
                self.scores = self.clf_final.predict(XV_CSP)
                # self.csp_filters = self.csp.filters_
                
                # Option 2
                # self.chain = Pipeline([('CSP', self.csp), ('SVC', self.clf_final)])
                # self.chain.fit(XTF, yT)           
                # self.scores = self.chain.predict(XVF)
                # self.csp_filters = self.chain['CSP'].filters_
                
                self.acc = np.mean(self.scores == yV) # or chain.score(XVF, yV)     
                self.kappa = cohen_kappa_score(self.scores, yV)
                
            elif self.ap['option'] == 'sbcsp': 
                self.acc, self.kappa = self.sbcsp_approach(XT, XV, yT, yV)
        
    
    def sbcsp_approach(self, XT, XV, yT, yV):
        
        nbands = int(self.ap['nbands'])
        if nbands > (self.f_high-self.f_low): nbands = (self.f_high-self.f_low)
    
        
        n_bins = self.f_high - self.f_low
        overlap = 0.5 if self.overlap else 1
        step = n_bins / nbands
        size = step / overlap
        
        sub_bands = []
        for i in range(nbands):
            fl_sb = round(i * step + self.f_low)
            fh_sb = round(i * step + size + self.f_low)
            # if fl_sb == 0: fl_sb = 0.001
            if fh_sb <= self.f_high: sub_bands.append([fl_sb, fh_sb]) 
            # se ultrapassar o limite superior da banda total, desconsidera a última sub-banda
            # ... para casos em que a razão entre a banda total e n_bands não é exata 
        
        # print(sub_bands)
        nbands = len(sub_bands)
        
        XTF, XVF = [], []
        if self.filt_info['design'] == 'DFT':
            
            XT_FFT = self.filt.apply_filter(XT)
            XV_FFT = self.filt.apply_filter(XV)
            for i in range(nbands):
                bmin = sub_bands[i][0] * self.dft_size_band
                bmax = sub_bands[i][1] * self.dft_size_band
                XTF.append(XT_FFT[:, :, bmin:bmax])
                XVF.append(XV_FFT[:, :, bmin:bmax]) 
                # print(bmin, bmax)
        
        elif self.filt_info['design'] in ['IIR' or 'FIR']:
            
            for i in range(nbands):
                filt_sb = Filter(sub_bands[i][0], sub_bands[i][1], len(XT[0,0,:]), self.fs, self.filt_info)
                XTF.append(filt_sb.apply_filter(XT))
                XVF.append(filt_sb.apply_filter(XV))
        
        self.chain = [ Pipeline([('CSP', CSP(n_components=self.ncomp)), ('LDA', LDA())]) for i in range(nbands) ]
        
        for i in range(nbands): self.chain[i]['CSP'].fit(XTF[i], yT)
            
        XT_CSP = [ self.chain[i]['CSP'].transform(XTF[i]) for i in range(nbands) ]
        XV_CSP = [ self.chain[i]['CSP'].transform(XVF[i]) for i in range(nbands) ]
        
        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        for i in range(len(sub_bands)): 
            self.chain[i]['LDA'].fit(XT_CSP[i], yT)
            SCORE_T[:, i] = np.ravel(self.chain[i]['LDA'].transform(XT_CSP[i]))  # classificações de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(self.chain[i]['LDA'].transform(XV_CSP[i]))
        
        # csp_filters_sblist = [self.chain[i]['CSP'].filters_ for i in range(nbands)]
        # lda_sblist = [self.chain[i]['LDA'] for i in range(nbands)] 
            

        SCORE_T0 = SCORE_T[yT == self.class_ids[0], :]
        SCORE_T1 = SCORE_T[yT == self.class_ids[1], :]
        
        self.p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
        self.p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
        META_SCORE_T = np.log(self.p0.pdf(SCORE_T) / self.p1.pdf(SCORE_T))
        META_SCORE_V = np.log(self.p0.pdf(SCORE_V) / self.p1.pdf(SCORE_V))
        

        self.clf_final.fit(META_SCORE_T, yT)
        self.scores = self.clf_final.predict(META_SCORE_V)
        
        acc = np.mean(self.scores == yV)
        kappa = cohen_kappa_score(self.scores, yV)
        
        return acc, kappa


    def sbcsp_approach_old(self, XT, XV, yT, yV):
        
        self.chain = Pipeline([('CSP', CSP(n_components=self.ncomp)), ('LDA', LDA()) ])
        
        nbands = int(self.ap['nbands'])
        nbands = (self.f_high-self.f_low) if nbands >= (self.f_high-self.f_low) else nbands
        
        if self.filt_info['design'] == 'DFT':
            XT_FFT = self.filt.apply_filter(XT)
            XV_FFT = self.filt.apply_filter(XV)
            n_bins = len(XT_FFT[0, 0, :])  # ou (fh-fl) * 4 # Número total de bins de frequencia
        else: n_bins = self.f_high - self.f_low
        overlap = 0.5 if self.overlap else 1
        step = n_bins / nbands # int()
        size = step / overlap # int() 
        SCORE_T = np.zeros((len(XT), nbands))
        SCORE_V = np.zeros((len(XV), nbands))
        self.csp_filters_sblist = []
        self.lda_sblist = []
        for i in range(nbands):
            if self.filt_info['design'] == 'DFT':
                bin_ini = round(i * step)
                bin_fim = round(i * step + size)
                if bin_fim >= n_bins: bin_fim = n_bins # - 1
                XTF = XT_FFT[:, :, bin_ini:bin_fim]
                XVF = XV_FFT[:, :, bin_ini:bin_fim]
                # print( round(bin_ini * (self.filt.res_freq/2) + self.f_low), round(bin_fim * (self.filt.res_freq/2) + self.f_low) ) # print bins convertidos para Hertz
            
            else:
                fl_sb = round(i * step + self.f_low)
                fh_sb = round(i * step + size + self.f_low)
                if fh_sb > self.f_high: fh_sb = self.f_high
                if fl_sb > fh_sb: fl_sb = fh_sb
                filt_sb = Filter(fl_sb, fh_sb, len(XT[0,0,:]), self.fs, self.filt_info)
                XTF = filt_sb.apply_filter(XT)
                XVF = filt_sb.apply_filter(XV)
                # print(fl_sb, fh_sb)
                
            self.chain['CSP'].fit(XTF, yT)
            XT_CSP = self.chain['CSP'].transform(XTF)
            XV_CSP = self.chain['CSP'].transform(XVF)
            self.chain['LDA'].fit(XT_CSP, yT)
            SCORE_T[:, i] = np.ravel(self.chain['LDA'].transform(XT_CSP))  # classificações de cada época nas N sub bandas - auto validação
            SCORE_V[:, i] = np.ravel(self.chain['LDA'].transform(XV_CSP))
            self.csp_filters_sblist.append(self.chain['CSP'].filters_)
            self.lda_sblist.append(self.chain['LDA'])
            
        SCORE_T0 = SCORE_T[yT == self.class_ids[0], :]
        SCORE_T1 = SCORE_T[yT == self.class_ids[1], :]
        self.p0 = norm(np.mean(SCORE_T0, axis=0), np.std(SCORE_T0, axis=0))
        self.p1 = norm(np.mean(SCORE_T1, axis=0), np.std(SCORE_T1, axis=0))
        META_SCORE_T = np.log(self.p0.pdf(SCORE_T) / self.p1.pdf(SCORE_T))
        META_SCORE_V = np.log(self.p0.pdf(SCORE_V) / self.p1.pdf(SCORE_V))
            
        self.clf_final.fit(META_SCORE_T, yT)
        self.scores = self.clf_final.predict(META_SCORE_V)
        acc = np.mean(self.scores == yV)
        kappa = cohen_kappa_score(self.scores, yV)

        return acc, kappa
    

    