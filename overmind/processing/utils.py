import os
import numpy as np
import json
import mne
import pickle
from scipy.io import loadmat, savemat

# constants
FONT_SIZE = 20
BUTTON_SIZE = (300, 50)
BUTTON_BOX_SIZE = (1, 0.4)
PATH_TO_SESSION = os.path.dirname(__file__) + '/../userdata/'

def save_pickle_data(data_in, path, mode='wb'):
    with open(path, mode) as handle: pickle.dump(data_in, handle)

def load_pickle_data(path, mode='rb'):
    return pickle.load(open(path, mode))

def load_npy_data(path):
    """Loads text file content as numpy matrix
    path: path to text file     cols: order of columns to be read
    Returns matrix : numpy matrix, shape as written in txt """
    matrix = np.load(open(path, "rb"))
    # return np.fliplr(matrix.T).T
    return matrix

def save_npy_data(data_in, path, mode='a'): #mode='wb'
    #with open(path, mode) as data_file: np.save(data_file, data_in)
    np.save(path, data_in)

def readEvents(events_path):
    e = np.load(events_path)
    # insert dummy column to fit mne event list format
    t_events = np.insert(e, 1, values=0, axis=1)
    t_events = t_events.astype(int)  # convert to integer
    return t_events

def nanCleaner(data):
    # Removes NaN from data by interpolation. in:np matrix channels X samples; out:clean data with no NaN samples
    for i in range(data.shape[0]):
        bad_idx = np.isnan(data[i, ...])
        data[i, bad_idx] = np.interp(bad_idx.nonzero()[0], (~bad_idx).nonzero()[0], data[i, ~bad_idx])
    return data

def extractEpochs(data, e, smin, smax, ev_id): # (loadedData, ev, smin, smax, [1, 2])
    """Extracts the epochs from data based on event information
    data : raw data in mne format
    event_id : labels of each class
    tmin: time in seconds at which the epoch starts (event as reference)
    tmax: time in seconds at which the epoch ends (event as reference)
    Returns:
    epochs: epochs in mne format
    labels: labels of each extracted epoch
    """
    events_list = e[:, 1] #e[:, 2]
    cond = False
    for i in range(len(ev_id)): cond += (events_list == ev_id[i])
    idx = np.where(cond)[0]
    s = e[idx, 0]
    sBegin = s + smin
    sEnd = s + smax
    n_epochs = len(sBegin)
    n_channels = data.shape[0]
    n_samples = smax - smin
    epochs = np.zeros([n_epochs, n_channels, n_samples])
    labels = events_list[idx]
    bad_epoch_list = []
    for i in range(n_epochs):
        epoch = data[:, sBegin[i]:sEnd[i]]
        # Check if epoch is complete
        if epoch.shape[1] == n_samples: epochs[i, :, :] = epoch
        else:
            print('Incomplete epoch detected...')
            bad_epoch_list.append(i)
    labels = np.delete(labels, bad_epoch_list)
    epochs = np.delete(epochs, bad_epoch_list, axis=0)
    return epochs, labels

def extractEpochsMNE(data, events_list, events_id, tmin, tmax): ## Não utilizado
    """Extracts the epochs from data based on event information
    data: raw data in mne format
    events_list: list of events in mne format,
    shape(time stamp (in samples), offset (can be a range arr), label)
    event_id: labels of each class
    tmin: time in seconds at which the epoch starts (event as reference)
    tmax: time in seconds at which the epoch ends (event as reference)
    Returns
    epochs: epochs in mne format
    labels: labels of each extracted epoch """
    # mne.find_events
    # Read epochs (train will be done only between 1 and 2s)
    # Testing will be done with a running classifier
    picks = mne.pick_types(data.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
    epochs = mne.Epochs(data, events_list, events_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True, add_eeg_ref=False, verbose=False)
    labels = epochs.events[:, -1]
    return epochs, labels

def saveObjAsJson(obj, filename): ## Não utilizado
    with open(filename, "w") as file: file.write(json.dumps(obj.__dict__, file, indent=4))

def loadChannelLabels(path): ## Não utilizado
    # if os.path.exists("data/rafael/precal_config"):
    with open(path, "r") as data_file: data = json.load(data_file)
    return data["ch_labels"].split(' ')