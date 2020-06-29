# -*- coding: utf-8 -*-
import numpy as np
import json
import os
import threading
import collections  # circular buffer
from math import floor
from random import randint
from processing.utils import save_pickle_data, load_pickle_data, save_npy_data, load_npy_data, extractEpochs, readEvents
import processing.openbci_v3 as realBoard
import processing.openbci_simu as simuBoard
import pyOpenBCI

BAUD = 115200

class SampleManager(threading.Thread): # define a classe manager
    def __init__(self, srate, p, buf_len=None, tmin=0.5, tmax=2.5, class_ids=[1,2], daisy=False, mode='openbci', path=None, labels_path=None, dummy=False):
        super(SampleManager, self).__init__()
        self.srate = srate
        self._stopper = threading.Event() # alterado de _stop para _stopper
        self.sample_counter = 0
        self.stop_flag = False
        self.acq_mode = mode
        self.first_sample = True
        self.playback_path = path
        self.playback_labels_path = labels_path
        self.last_playback_label = [None, None]
        self.current_playback_label = [None, None]
        self.next_playback_label = [None, None]
        self.dummy = dummy
        self.daisy = daisy
        self.class_ids = class_ids
        self.current_cmd = 0 # indica a classe da amostra que se deseja usar, se 0 usa o valor de winning
        self.winning = 1 # se 1 comeca pegando amostras da classe B, se 0 pega amostras da classe A
        # self.winning = np.random.choice(2, 1)

        smin = int(floor(tmin * self.srate))  # não seria justo deixar continuo (epoca inteira, de 0 a 7.5s) ???
        smax = int(floor(tmax * self.srate))
        self.buffer_length = int(smax - smin) if buf_len is None else buf_len
        # print(self.buffer_length)

        self.circBuff = collections.deque(maxlen=self.buffer_length)  # create a qeue for input data # buffer que armazena n=buf_len amostras
        self.tBuff = collections.deque(maxlen=self.buffer_length) # create a qeue for time series # buffer que armazena n=buf_len posições de amostras (serie temporal)
        self.event_list = np.array([]).reshape(0, 2)

        if self.acq_mode == 'openbci':
            # self.board = realBoard.OpenBCIBoard(port=p, baud=BAUD, daisy=self.daisy)
            self.board = pyOpenBCI.OpenBCICyton(port=p, baud=BAUD, daisy=self.daisy) # '/dev/ttyUSB0'
        elif self.acq_mode == 'simu':
            if self.dummy:
                loadedData = np.ones([2, 16])
                self.board = simuBoard.OpenBCIBoard(port=p, baud=BAUD, daisy=self.daisy, data=loadedData, sample_rate=self.srate)
            else:
                self.loadedData, self.playback_labels, self.info_data = np.load(self.playback_path, allow_pickle=True) #load_pickle_data(self.playback_path)
                # self.loadedData = load_npy_data(self.playback_path)[:22] #org = .T[:22]
                # if not self.playback_labels_path == '':
                ev = self.playback_labels
                if not self.playback_labels == []: # self.playback_labels é a lista de eventos [cronologia amostral, rótulo] das épocas carregadas
                    # self.playback_labels = iter(load_npy_data(self.playback_labels_path))
                    self.playback_labels = iter(self.playback_labels) # transforma a lista de rótulos das épocas em um obj iterável
                    self.current_playback_label = next(self.playback_labels) # avanca uma posicao pela lista iteravel (do 1o rótulo (1a época) para o 2o)
                    self.next_playback_label = next(self.playback_labels) # avanca mais uma posicao pela lista iteravel (do 2o rótulo para o 3o)
                # ev = readEvents(self.playback_labels_path) # only add dummy column to mne extract epoch
                self.epochs, self.labels = extractEpochs(self.loadedData, ev, smin, smax, self.class_ids)
                self.playbackData = np.zeros([1, self.epochs.shape[1]]) # playbackData é iniciado com uma linha de 0 com [channels] colunas
                self.board = simuBoard.OpenBCIBoard(port=p, baud=BAUD, daisy=self.daisy, data=self.playbackData, sample_rate=self.srate)
                self.append_epoch() # add a new epoch in playbackData and board
                self.append_epoch() # add a new epoch in playbackData and board
                # neste ponto, terei duas épocas inteiras [ ch x (2*buf_len)+1 ] em playbackData

    def append_epoch(self):
        # if self.current_cmd == 1: print(f'Command {self.current_cmd} << LH')
        # else: print(f'Command {self.current_cmd} >> RH')
        # self.winning = np.random.choice(2, 1)
        if self.current_cmd == 0:  # usado quando o rótulo da classe NÃO é passado
            idx_a = np.where(self.labels == self.class_ids[0])[0]
            idx_b = np.where(self.labels == self.class_ids[1])[0]
            # print(idx_a.shape, idx_b.shape)
            if self.winning == 1:
                k = randint(0, len(idx_b) - 1)  # escolhe uma amostra aleatoria da classe B
                idx = idx_b[k]
                # self.current_cmd = self.class_ids[1]
            else:
                k = randint(0, len(idx_a) - 1)  # escolhe uma amostra aleatoria da classe A
                idx = idx_a[k]
                # self.current_cmd = self.class_ids[0]
        else:
            idx_ab = np.where(self.labels == self.current_cmd)[0]  # usado quando o rótulo da classe é passado
            k = randint(0, len(idx_ab) - 1)
            idx = idx_ab[k]
        self.playbackData = np.vstack([self.playbackData, self.epochs[idx].T]) # acopla em playbackData uma época inteira [ch x buf_len]
        self.board.playback_data = self.playbackData

    def run(self):
        if self.acq_mode == 'openbci':
            self.board.start_stream(self.GetData()) # start getting data from amplifier ###### pyOpenBCI
            #self.board.start_streaming(self.GetData)
        elif self.acq_mode == 'simu':
            self.board.start_stream(self.GetData)

    def GetData(self, sample): #### step 5
        #print(sample.channels_data)
        """ Get the data from amplifier and push it into the circular buffer.
        Also implements a counter to plot against the read value
        ps: This function is called by the OpenBci start_streaming() function """
        indata = sample.channels_data # channel_data
        if not self.expected_package(sample.id):
            pass
            # print('wrong sequence')
            # nan_arr = np.empty(len(indata))
            # nan_arr[:] = np.nan
            # nan_arr = nan_arr.tolist()
        # print(len(indata)) # indata é uma entrada (1 amostra de EEG para cada canal = [ 1 x ch ]
        self.updateCircBuf(indata)
        self.StoreData(indata)
        if self.board.sample_counter > (self.playbackData.shape[0] - 30): # quando a fonte do streaming simulado estiver acabando (faltando 30)... add mais uma época
            # print(self.board.sample_counter, self.playbackData.shape[0])
            self.append_epoch()
        self.sample_counter += 1
        if self.stop_flag: self.Stop()

    def updateCircBuf(self, indata):
        self.circBuff.append(indata)
        self.tBuff.append(self.sample_counter) # numero de amostras streamzadas
        # print('>>', np.asarray(data).shape, np.asarray(self.circBuff).shape, np.asarray(self.tBuff).shape)

    def StoreData(self, new_data):
        data = np.array(new_data)  # transform list into numpy array
        if not hasattr(self, 'all_data'): self.all_data = np.array([]).reshape(0, len(data)) # cria all_data na primeira amostra recebida
        self.all_data = np.vstack((self.all_data, data))  # append to data stack
        # print('all_data: ', self.all_data.shape)
        #print('data: ', data.shape)

    def SaveData(self, path):
        #print(self.all_data, path)
        save_npy_data(self.all_data, path, mode='w')

    def expected_package(self, sid):
        if self.first_sample:
            self.last_sid = sid
            self.first_sample = False
        else:
            if self.daisy:
                if sid == (self.last_sid + 2) % 256:
                    self.last_sid = sid
                    return True
                else:
                    self.last_sid = sid
                    return False
            else:
                if sid == (self.last_sid + 1) % 256:
                    return True
                    self.last_sid = sid
                else:
                    return False
                    self.last_sid = sid
        # print('sid: ', sid)
        # print('last sid: ', self.last_sid)

    def GetBuffData(self, mode=None): # retorna os dados atualmente no buffer de amostras por canal (circBuff) e no buffer com a serie temporal (tBuff)
        t = np.array(self.tBuff)
        d = np.array(self.circBuff)
        return t, d

    def Stop(self):
        print('Streaming stopped. Closing connection to hardware')
        self.board.stop()
        self.board.disconnect()
        self._stopper.set() # alterado de _stop para _stopper

    def Stopped(self):
        return self._stopper.isSet() # alterado de _stop para _stopper

    def MarkEvents(self, ev_type):
        new = np.array([self.sample_counter, ev_type])
        self.event_list = np.vstack((self.event_list, new))

    def SaveEvents(self, path):
        save_npy_data(self.event_list, path, mode='w')

    def SaveAll(self, acq_info, path):
        for i in range(0, len(self.event_list)):
            if self.event_list[i, 1] == 0: self.event_list[i, 1] = (self.event_list[i + 1, 1] + 10)  # labeling start trial [11 a 14] according cue [1,2,3,4]

        data_to_save = [self.all_data.T, self.event_list.astype(int), acq_info]
        # save_pickle_data(data_to_save, path + '.omi', mode='wb')
        np.save(path, data_to_save, allow_pickle=True)

    def clear_buffer(self):
        self.circBuff.clear()
        self.tBuff.clear()
        # self.playbackData = np.delete(self.playbackData, range(self.board.sample_counter + 30, self.playbackData.shape[0]), axis=0)
        # self.append_epoch()

    def jump_playback_data(self):
        self.board.sample_counter = self.board.playback_data.shape[0] - 50