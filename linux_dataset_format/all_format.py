# -*- coding: utf-8 -*-
# @author: Vitor Vilas-Boas
import pickle
import numpy as np
from bci_utils import create_omi, create_npy
from scipy.signal import decimate

# ds = {'name':'III3a', 'subjects':['K3','K6','L1'], 'n_channels':60, 'prefix':''}
# ds = {'name':'III4a', 'subjects':['aa','al','av','aw','ay'], 'n_channels':118, 'prefix':''}
# ds = {'name':'IV2a', 'subjects':range(1,10), 'n_channels':22, 'prefix':'A0'}
# ds = {'name':'IV2b', 'subjects':range(1,10), 'n_channels':3, 'prefix':'B0'}
ds = {'name':'LEE54', 'subjects':range(1,55), 'n_channels':62, 'prefix':'subj'}
# ds = {'name':'TWL', 'subjects':['TL', 'WL'], 'n_channels':8, 'prefix':''}
# ds = {'name':'CL', 'subjects':['CL'], 'n_channels':16, 'prefix':''}

path = '/mnt/dados/eeg_data/' ## >>> SET HERE THE DATA SET PATH

file_type = 'npy' # omi or npy

if file_type == 'npy': ### Create NPY file
    for subj in ds['subjects']: create_npy(subj, ds['name'], path, ds['n_channels']) # create

elif file_type == 'omi': ### Create OMI file
    for subj in ds['subjects']: create_omi(subj, ds['name'], path, ds['n_channels']) # create
    
    
# data = np.load(path + ds['name'] + '/' + file_type + '/' + ds['prefix'] + str(ds['subjects'][0]) + '_data.npy') # loading test _data.npy
# events = np.load(path + ds['name'] + '/' + file_type + '/' + ds['prefix'] + str(ds['subjects'][0]) + '_events.npy') # loading test _events.npy
# d, e, i = pickle.load(open(path + ds['name'] + '/' + file_type + '_1000/' + ds['prefix'] + str(ds['subjects'][0]) + '.omi', 'rb')) # loading test .omi

# x = decimate(d, 10)

# ## DOWNSAMPLING
# dd = np.asarray([ d[:,i] for i in range(0, d.shape[-1], 10) ]).T
# ee = np.copy(e)
# ee[:, 0] = [ ee[i, 0]/10 for i in range(ee.shape[0]) ]
    