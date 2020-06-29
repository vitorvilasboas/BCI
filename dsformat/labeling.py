import mne
import pickle
import numpy as np
import pandas as pd
from scipy.io import loadmat
from datetime import datetime
from scipy.signal import decimate

warnings.filterwarnings("ignore", category=DeprecationWarning)
mne.set_log_level(50, 50)

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

