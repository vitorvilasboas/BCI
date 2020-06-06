# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:44:10 2020
@author: Vitor Vilas-Boas

5 subjects | 2 classes (RH, FooT)
Epoch distribution:
    aa : train=168 test=112  
    al : train=224 test=56
    av : train=84  test=196
    aw : train=56  test=224
    ay : train=28  test=252
Start trial= 0; Start cue=0; Start MI= 0; End MI=3.5; End trial(break)= 5.25~5.75
"""

import os
import pickle
import numpy as np
from datetime import datetime
from scipy.io import loadmat

path = '/mnt/dados/eeg_data/III4a/' ## >>> ENTER THE PATH TO THE DATASET HERE

path_out = path + 'npy/'
if not os.path.isdir(path_out): os.makedirs(path_out)

for suj in ['aa','al','av','aw','ay']:
    mat = loadmat(path + suj + '.mat')
    d = mat['cnt'].T # (0.1 * mat['cnt'].astype(float)).T # convert to uV
    pos = mat['mrk'][0][0][0][0]
    true_mat = loadmat(path + 'true_labels/' + suj + '.mat')
    true_y = np.ravel(true_mat['true_y']) # RH=1 Foot=2
    true_y = np.where(true_y == 2, 3, true_y) # Foot=3
    true_y = np.where(true_y == 1, 2, true_y) # RH=2
    e = np.c_[pos, true_y]
    # d = corrigeNaN(d)
    # d = np.asarray([ np.nan_to_num(j) for j in d ])
    # d = np.asarray([ np.ravel(pd.DataFrame(j).fillna(pd.DataFrame(j).mean())) for j in d ])

    i = {'fs':100, 'class_ids':[2,3], 'trial_tcue':0, 'trial_tpause':4.0, 'trial_mi_time':4.0,
         'trials_per_class':140, 'eeg_channels':d.shape[0], 'ch_labels':mat['nfo']['clab'],
         'datetime':datetime.now().strftime('%d-%m-%Y_%Hh%Mm')}

    #%% save npy file
    np.save(path_out + suj, [d, e, i], allow_pickle=True)
    # pickle.dump([data, events, info], open(path_out + suj + '.pkl', 'wb'))