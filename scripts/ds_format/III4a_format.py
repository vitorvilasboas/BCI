# -*- coding: utf-8 -*-

from datetime import datetime
from scipy.io import loadmat
import numpy as np
import pickle

""" 5 subjects | 2 classes (RH, FooT)
    Epoch distribution:
        aa : train=168 test=112  
        al : train=224 test=56
        av : train=84  test=196
        aw : train=56  test=224
        ay : train=28  test=252
    Start trial= 0; Start cue=0; Start MI= 0; End MI=3.5; End trial(break)= 5.25~5.75
"""

path = '/mnt/dados/eeg_data/III4a/'

for suj in ['aa','al','av','aw','ay']:
    mat = loadmat(path + 'mat/' + suj + '.mat')
    data = mat['cnt'].T # 0.1 * mat['cnt'].T # convert to uV
    pos = mat['mrk'][0][0][0][0]
    true_mat = loadmat(path + 'mat/true_labels/trues_' + suj + '.mat')
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
    

    np.save(path + 'npy/' + suj, [data, events, info], allow_pickle=True)
    # with open(path + 'omi/' + suj + '.omi', 'wb') as handle: pickle.dump([data, events, info], handle)