import re
import os
import random
import time
import threading
import math
import pyautogui
import pygame
from pygame.locals import*
import collections
import numpy as np
# import Bar
import kivy.garden.bar  # from kivy.garden.bar import Bar
from kivy.app import App
from kivy.clock import Clock
from datetime import datetime
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.checkbox import CheckBox
from kivy.uix.label import Label
from kivy.uix.spinner import Spinner
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.widget import Widget
from kivy.core.window import Window
from kivy.properties import ObjectProperty, StringProperty, NumericProperty, ListProperty, BooleanProperty
from view.session_info import UserSession
from processing.processor import Approach
from processing.optimizer import Optimize
from processing.sample_manager import SampleManager
from processing.utils import save_npy_data, load_npy_data, PATH_TO_SESSION, FONT_SIZE, extractEpochs, nanCleaner, save_pickle_data, load_pickle_data
import tkinter as tk
from tkinter import filedialog
from plyer import filechooser

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class CalLoad(Screen):
    max_channels = NumericProperty(8)
    selection = ListProperty([])

    def __init__(self, session, **kwargs):
        super(CalLoad, self).__init__(**kwargs)
        self.session = session

    def back_to_bci(self, *args):
        self.manager.current = 'BCIMenu'
        self.manager.transition.direction = 'right'

    def change_to_cal_setup(self, *args):
        self.manager.current = 'CalSettings'
        self.manager.transition.direction = 'left'

    def popup_required(self, *args):
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(500, 180))
        label = Label(text='You need to set all eeg data info:\nEEG file, sample rate, class_ids and channels!')
        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
        box.add_widget(label)
        box.add_widget(botao)
        pop.open()

    def popup_load_error(self, *args):
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(300, 180))
        label = Label(text='EEG file not found!')
        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
        box.add_widget(label)
        box.add_widget(botao)
        pop.open()

    def popup_channels_error(self, no_channels, max_channels, *args):
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(500, 180))
        label = Label(text='Indexes ' + str(list(map(lambda x: x+1, no_channels))) + ' are outside the range of valid channels.\nIndex must be between 1 and ' + str(max_channels) + '.')
        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
        box.add_widget(label)
        box.add_widget(botao)
        pop.open()

    def save_and_progress(self, *args):
        if self.ids.eeg_path == '' or self.ids.srate.value == 0 or self.ids.class_ids.text == '< Select >' or self.ids.channels.value == '':
            self.popup_required()
        else:
            try:
                self.data, self.events, self.info = np.load(self.ids.eeg_path.value, allow_pickle=True)  # load_pickle_data() # load_mat_data()
                self.session.dp.eeg_path = self.ids.eeg_path.value
                self.session.dp.eeg_info = self.info
                self.session.dp.class_ids = list(map(int, self.ids.class_ids.text.split(' ')))
                self.session.dp.cross_val = self.ids.cross_val.active
                self.session.dp.n_folds = self.ids.n_folds.value if self.session.dp.cross_val else None
                self.session.dp.test_perc = self.ids.test_perc.value

                if ':' in self.ids.channels.value: # if :E or S:E (channels sequence)
                    limits = list(map(int, [ele for ele in self.ids.channels.value.split(':') if ele.isnumeric()]))
                    if len(limits) == 1: ch_idx = list(np.arange(0, limits[0])) # if :E
                    else: ch_idx = list(np.arange(limits[0]-1, limits[1])) # if S:E
                elif '-' in self.ids.channels.value: # se -1 (all channels)
                    # ch_idx = list(map(int, self.ids.channels.value.split(' ')))
                    ch_idx = list(np.arange(0, self.info['eeg_channels']))
                else: # if A B C D (channels list)
                    idx_list = [ele for ele in self.ids.channels.value.split(' ') if ele.isnumeric()]
                    if idx_list: ch_idx = list(map(lambda x: int(x) - 1, idx_list))
                    else: ch_idx = list(np.arange(0, self.info['eeg_channels']))

                # print(ch_idx)
                no_channels = [ele for ele in ch_idx if ele >= self.data.shape[0] or ele < 0]
                if no_channels != []: self.popup_channels_error(no_channels, self.data.shape[0])
                else:
                    self.session.dp.channels = ch_idx
                    self.session.dp.flag_load = True
                    self.session.saveSession()
                    self.change_to_cal_setup()
            except:
                self.popup_load_error()

    def set_disabled_field(self):
        self.ids.srate.value == 0
        self.ids.class_ids.values = ''
        self.ids.class_ids.text = '< Select >'
        self.ids.channels.value = '-1'
        self.ids.btn_save_progress.disabled = True
        self.ids.item_class_ids.disabled = True
        self.ids.class_ids.disabled = True
        self.ids.channels.disabled = True
        self.ids.item_cross_val.disabled = True
        self.ids.cross_val.disabled = True
        self.ids.test_perc.disabled = True
        self.ids.n_folds.disabled = False if self.ids.cross_val.active else True

    def set_enabled_field(self):
        self.ids.btn_save_progress.disabled = False
        self.ids.item_class_ids.disabled = False
        self.ids.class_ids.disabled = False
        self.ids.channels.disabled = False
        self.ids.item_cross_val.disabled = False
        self.ids.cross_val.disabled = False
        self.ids.test_perc.disabled = False
        self.ids.n_folds.disabled = False if self.ids.cross_val.active else True

    def update_settings(self):
        if self.ids.eeg_path.value != '':
            try:
                self.data, self.events, self.info = np.load(self.ids.eeg_path.value, allow_pickle=True) # load_pickle_data() # load_mat_data()
                self.ids.srate.value = self.info['fs']
                self.max_channels = self.data.shape[0] # self.info['eeg_channels']
                self.set_enabled_field()
                if self.info['class_ids'] == [1, 2, 3, 4]:
                    self.ids.class_ids.values = ['< Select >', '1 2', '1 3', '1 4', '2 3', '2 4', '3 4']
                elif self.info['class_ids'] == [1, 3]:
                    self.ids.class_ids.values = ['< Select >', '1 3']
                    self.ids.class_ids.text = '1 3'
                else:
                    self.ids.class_ids.values = ['< Select >', '1 2']
                    self.ids.class_ids.text = '1 2'
            except: self.set_disabled_field()
        else: self.set_disabled_field()

    def load_saved_settings(self, *args):
        self.ids.eeg_path.value = '' if self.session.dp.eeg_path is None else self.session.dp.eeg_path
        self.update_settings()
        self.ids.class_ids.text = '< Select >' if self.session.dp.class_ids is None else str(self.session.dp.class_ids[0]) + ' ' + str(self.session.dp.class_ids[1]) # str(self.session.dp.class_ids).replace(',', '').replace('[', '').replace(']', '')
        self.ids.cross_val.active = self.session.dp.cross_val
        self.ids.n_folds.value = 10 if self.session.dp.n_folds is None else self.session.dp.n_folds
        self.ids.test_perc.value = 0.5 if self.session.dp.test_perc is None else self.session.dp.test_perc

        if self.session.dp.channels is None: self.ids.channels.value = '-1'
        else:
            channels_content = list(map(lambda x: int(x) + 1, self.session.dp.channels))
            # self.ids.channels.value = str(channels_content).replace(',', '').replace('[', '').replace(']', '')
            if len(channels_content) == self.max_channels: # self.session.dp.eeg_info['eeg_channels']
                self.ids.channels.value = '-1'
            else:
                ch_idx = ''
                for ele in list(map(str, channels_content)):
                    if ele.isnumeric():
                        ch_idx += (ele + ' ')
                self.ids.channels.value = ch_idx

    def check_crossval_enabled(self, *args):
        self.ids.n_folds.disabled = False if self.ids.cross_val.active else True

    # def choose(self):
    #     root = tk.Tk()
    #     root.withdraw()
    #     return (filedialog.askopenfilename())

    def choose(self):
        '''Call plyer filechooser API to run a filechooser Activity.'''
        filechooser.open_file(on_selection=self.handle_selection)

    def handle_selection(self, selection):
        '''Callback function for handling the selection response from Activity.'''
        self.selection = selection

    def on_selection(self, *a, **k):
        '''Update TextInput.text after FileChoose.selection is changed via FileChoose.handle_selection'''
        self.ids.input.text = str(self.selection).replace('[', '').replace(']', '').replace("'", '')


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class CalSettings(Screen):
    epoch_lim_max = NumericProperty(8)
    epoch_lim_min = NumericProperty(8)
    srate = NumericProperty(8)

    def __init__(self, session, **kwargs):
        super(CalSettings, self).__init__(**kwargs)
        self.session = session

    def back_to_calLoad(self, *args):
        self.manager.current = 'CalLoad'
        self.manager.transition.direction = 'right'

    def calibrate(self, *args):
        self.save_config()
        if self.ids.check_auto_cal.active:
            auto_setup = Optimize()
            auto_setup.run_optimizer(self.session, self.ids.load_last_setup.active)
            self.save_opt_config(auto_setup.best)
            print(auto_setup.best)
            # print(auto_setup.best, auto_setup.trials)
            # print(auto_setup.trials.best_trial['result']['loss'])
            # print(auto_setup.trials.best_trial['misc']['vals'])
            # score = (-1) * auto_setup.trials.best_trial['result']['loss']

        # print(self.session.dp.sb_method, self.session.acq.sample_rate, self.session.dp.eeg_info['fs'],
        #       self.session.dp.f_low, self.session.dp.f_high, self.session.dp.csp_nei, self.session.dp.class_ids,
        #       self.session.dp.epoch_start, self.session.dp.epoch_end, self.session.dp.filt_approach,
        #       self.session.dp.sb_clf, self.session.dp.final_clf, self.session.dp.f_order, self.session.dp.n_sbands,
        #       self.session.dp.overlap, self.session.dp.cross_val, self.session.dp.n_folds, self.session.dp.test_perc)

        ap = Approach(self.session)
        ap.set_channels(self.session.dp.channels)
        ap.set_eeg_path(self.session.dp.eeg_path)
        # ap.set_cal_path(self.session.dp.eeg_path) # ap.set_cal_path_old(self.session.dp.eeg_path, self.session.dp.events_path)
        ap.define_approach(self.session.dp.sb_method, self.session.dp.eeg_info['fs'],
                           self.session.dp.f_low, self.session.dp.f_high, self.session.dp.csp_nei, self.session.dp.class_ids,
                           self.session.dp.epoch_start, self.session.dp.epoch_end, self.session.dp.filt_approach,
                           self.session.dp.sb_clf, self.session.dp.final_clf, self.session.dp.f_order, self.session.dp.n_sbands,
                           self.session.dp.overlap, self.session.dp.cross_val, self.session.dp.n_folds, self.session.dp.test_perc)
                         # True, False, 2, 0.5)
        score = ap.validate_model()
        ap.saveSetup(PATH_TO_SESSION + self.session.info.nickname)
        self.update_settings()
        pup = PopupMl(self.session, round(score * 100, 2))
        popup = Popup(title='Calibration Results', content=pup, size_hint=(None, None), size=(400, 200))
        popup.open()

    def save_opt_config(self, setup):
        while (setup['tmax'] - setup['tmin']) < 1: setup['tmax'] += 0.5  # garante janela minima de 1seg
        self.session.dp.auto_cal = self.ids.check_auto_cal.active
        self.session.dp.n_iter = self.ids.n_iter.value

        self.session.dp.f_low = int(setup['fl'])
        self.session.dp.f_high = int(setup['fh'])
        self.session.dp.epoch_start = setup['tmin']
        self.session.dp.epoch_end = setup['tmax']
        self.session.dp.buf_len = self.session.acq.sample_rate * (setup['tmax'] - setup['tmin'])
        self.session.dp.csp_nei = int(setup['ncomp'])

        if setup['approach'] == 1:
            self.session.dp.sb_method = True
            if int(setup['nbands']) > (int(setup['fh']) - int(setup['fl'])):
                setup['nbands'] = (int(setup['fh']) - int(setup['fl'])) - 1
            self.session.dp.n_sbands = int(setup['nbands'])
            self.session.dp.overlap = True
            self.session.dp.sb_clf = {'model':'LDA', 'lda_solver':'svd'}
        else:
            self.session.dp.sb_method = False
            self.session.dp.n_sbands = None
            self.session.dp.overlap = None
            self.session.dp.sb_clf = None

        if setup['filt'] == 0:
            self.session.dp.filt_approach = 'DFT'
            self.session.dp.f_order = None
        if setup['filt'] == 1:
            self.session.dp.filt_approach = 'IIR'
            self.session.dp.f_order = setup['iir_order']
        elif setup['filt'] == 2:
            self.session.dp.filt_approach = 'FIR'
            self.session.dp.f_order = setup['fir_order']

        if setup['clf'] == 0: self.session.dp.final_clf = {'model': 'Bayes'}
        elif setup['clf'] == 1:
            lda_solver = 'svd' if setup['lda_solver'] == 0 else 'lsqr' if setup['lda_solver'] == 1 else 'eigen'
            # if lda_solver == 'svd': lda_shrinkage = None
            # else:
            #     lda_shrinkage = None if setup['shrinkage'] == 0 else 'auto' if setup['shrinkage'] == 1 else {
            #         'shrinkage_float': setup['shrinkage_float']}
            # self.session.dp.final_clf = {'model':'LDA', 'lda_solver':lda_solver, 'shrinkage':lda_shrinkage} # with shrinkage
            self.session.dp.final_clf = {'model': 'LDA', 'lda_solver': lda_solver}
        elif setup['clf'] == 2:
            if setup['metric'] == 0: mf = 'euclidean'
            if setup['metric'] == 1: mf = 'manhattan'
            if setup['metric'] == 2: mf = 'minkowski'
            if setup['metric'] == 3: mf = 'chebyshev'
            # self.session.dp.final_clf = {'model':'KNN', 'neig': int(setup['neig']), 'metric':mf, 'p':setup['p']}
            self.session.dp.final_clf = {'model': 'KNN', 'neig': int(setup['neig']), 'metric': mf}
        elif setup['clf'] == 3:
            if setup['kernel'] == 0: kf = 'linear'
            if setup['kernel'] == 1: kf = 'poly'
            if setup['kernel'] == 2: kf = 'sigmoid'
            if setup['kernel'] == 3: kf = 'rbf'
            kernel = {'kf': kf}
            # kernel = {'kf': kf, 'degree': setup['degree']} if setup['kernel'] == 1 else {'kf': kf}

            # gamma = 'scale' if setup['gamma'] == 0 else 'auto' if setup['gamma'] == 1 else {'gamma_float':setup['gamma_float']}
            self.session.dp.final_clf = {'model': 'SVM', 'C': setup['C'], 'kernel': kernel, 'gamma': 'scale'}
        elif setup['clf'] == 4:
            if setup['activ'] == 0: af = 'identity'
            elif setup['activ'] == 1: af = 'logistic'
            elif setup['activ'] == 2: af = 'tanh'
            elif setup['activ'] == 3: af = 'relu'

            # if setup['eta_type'] == 0: eta_type = 'constant'
            # elif setup['eta_type'] == 1: eta_type = 'invscaling'
            # elif setup['eta_type'] == 2: eta_type = 'adaptive'

            eta_type = 'adaptive'

            if setup['mlp_solver'] == 0: mlp_solver = 'adam'
            elif setup['mlp_solver'] == 1: mlp_solver = 'lbfgs'
            elif setup['mlp_solver'] == 2: mlp_solver = 'sgd'

            self.session.dp.final_clf = {'model': 'MLP', 'eta': setup['eta'],
                                         # 'alpha': setup['alpha'],
                                         'activ': {'af': af},
                                         'n_neurons': int(setup['n_neurons']), 'n_hidden': int(setup['n_hidden']),
                                         'eta_type': eta_type, 'mlp_solver': mlp_solver}
        elif setup['clf'] == 5:
            criterion = 'gini' if setup['crit'] == 0 else 'entropy'
            # max_depth = None if setup['max_depth'] == 0 else {'max_depth_int':setup['max_depth_int']} # with max_depth
            # self.session.dp.final_clf = {'model':'DTree', 'crit':criterion, 'max_depth':max_depth, 'min_split':setup['min_split']} # with max_depth and min_split
            self.session.dp.final_clf = {'model': 'DTree', 'crit': criterion, 'max_depth': None, 'min_split': 2}

        self.session.dp.flag_setup = True
        self.session.saveSession()

    def save_config(self):
        self.session.dp.auto_cal = self.ids.check_auto_cal.active
        self.session.dp.n_iter = self.ids.n_iter.value if self.ids.check_auto_cal.active else None

        self.session.dp.f_low = self.ids.f_low.value
        self.session.dp.f_high = self.ids.f_high.value
        self.session.dp.epoch_start = self.ids.epoch_start.value
        self.session.dp.epoch_end = self.ids.epoch_end.value
        self.session.dp.buf_len = self.ids.buf_len.value
        self.session.dp.csp_nei = self.ids.csp_nei.value
        # self.session.dp.max_amp = self.ids.max_amp.value
        # self.session.dp.max_mse = self.ids.max_mse.value

        self.session.dp.sb_method = self.ids.sb_method.active
        self.session.dp.n_sbands = self.ids.n_sbands.value if self.ids.sb_method.active else None
        self.session.dp.overlap = self.ids.overlap.active if self.ids.sb_method.active else None
        self.session.dp.sb_clf = {'model':'LDA', 'lda_solver':'svd'} if self.ids.sb_method.active else None

        self.session.dp.filt_approach = self.ids.filt_approach.text
        self.session.dp.f_order = self.ids.f_order.value if self.ids.filt_approach.text in ['FIR', 'IIR'] else None

        final_clf_info = {'model': self.ids.clf1.text}
        if self.ids.clf1.text == 'LDA':
            final_clf_info.update({'lda_solver': str(self.ids.lda_solver.text).lower(), 'shrinkage': None})
        if self.ids.clf1.text == 'SVM':
            # kernel = {'kf': str(self.ids.svm_kernel.text).lower(), 'degree': 3} if str(self.ids.svm_kernel.text).lower() == 'poly' else {'kf': str(self.ids.svm_kernel.text).lower()}
            kernel = {'kf': str(self.ids.svm_kernel.text).lower()}
            final_clf_info.update({'C': self.ids.svm_c.value, 'kernel': kernel, 'gamma': 'scale'})
        if self.ids.clf1.text == 'KNN':
            final_clf_info.update({'neig': self.ids.neighbors_knn.value, 'metric': str(self.ids.metric_knn.text).lower()})
            # final_clf_info.update({'neig': self.ids.neighbors_knn.value, 'metric': str(self.ids.metric_knn.text).lower(), 'p': 2}) # with p
        if self.ids.clf1.text == 'MLP':
            final_clf_info.update(
                {'eta': self.ids.mlp_eta.value, # 'alpha': ids.mlp_alpha.value,
                 'activ': {'af': self.ids.mlp_activation.text},
                 'n_neurons': self.ids.mlp_hidden_neurons.value, 'n_hidden': self.ids.mlp_hidden_size.value,
                 'eta_type': 'adaptive',
                 'mlp_solver': str(self.ids.mlp_solver.text).lower()})  # 'eta_type':{'eta_type':'constant'}
        if self.ids.clf1.text == 'DTree':
            final_clf_info.update({'crit': self.ids.dtree_criterion.text, 'max_depth': None, 'min_split': 2})
        self.session.dp.final_clf = final_clf_info

        self.session.dp.flag_setup = True
        self.session.saveSession()

    def update_settings(self):
        # self.ids.max_amp.value = self.session.dp.max_amp
        # self.ids.max_mse.value = self.session.dp.max_mse
        # self.ids.clf2.text = self.session.dp.sb_clf
        self.ids.check_auto_cal.active = self.session.dp.auto_cal
        if self.session.dp.auto_cal:
            self.ids.n_iter.value = self.session.dp.n_iter
            self.ids.load_last_setup.active = True

        self.ids.f_low.value = self.session.dp.f_low
        self.ids.f_high.value = self.session.dp.f_high
        self.ids.epoch_start.value = self.session.dp.epoch_start
        self.ids.epoch_end.value = self.session.dp.epoch_end
        self.ids.buf_len.value = self.session.dp.buf_len
        self.ids.csp_nei.value = self.session.dp.csp_nei

        self.ids.sb_method.active = self.session.dp.sb_method
        if self.session.dp.sb_method:
            self.ids.n_sbands.value = self.session.dp.n_sbands
            self.ids.overlap.active = self.session.dp.overlap

        self.ids.filt_approach.text = self.session.dp.filt_approach
        if self.session.dp.filt_approach in ['FIR', 'IIR']: self.ids.f_order.value = self.session.dp.f_order

        self.ids.clf1.text = self.session.dp.final_clf['model']
        if self.ids.clf1.text == 'LDA':
            self.ids.lda_solver.text = str(self.session.dp.final_clf['lda_solver'])
        if self.ids.clf1.text == 'SVM':
            self.ids.svm_c.value = self.session.dp.final_clf['C']
            self.ids.svm_kernel.text = str(self.session.dp.final_clf['kernel']['kf']).capitalize()
            # self.ids.svm_gamma.value = self.session.dp.final_clf['gamma']
        if self.ids.clf1.text == 'KNN':
            self.ids.neighbors_knn.value = self.session.dp.final_clf['neig']
            self.ids.metric_knn.text = str(self.session.dp.final_clf['metric'])
        if self.ids.clf1.text == 'MLP':
            self.ids.mlp_eta.value = self.session.dp.final_clf['eta']
            # self.ids.mlp_alpha.value = self.session.dp.final_clf['alpha']
            self.ids.mlp_hidden_size.value = self.session.dp.final_clf['n_hidden']
            self.ids.mlp_hidden_neurons.value = self.session.dp.final_clf['n_neurons']
            self.ids.mlp_activation.text = self.session.dp.final_clf['activ']['af']
            self.ids.mlp_solver.text = str(self.session.dp.final_clf['mlp_solver'])
        if self.ids.clf1.text == 'DTree':
            self.ids.dtree_criterion.text = self.session.dp.final_clf['crit']

        self.set_enabled_field()

    def set_enabled_field(self, *args):
        if self.ids.check_auto_cal.active:
            # self.ids.buf_len.disabled = True
            self.ids.item_load_last_setup.disabled = False
            self.ids.load_last_setup.disabled = False
            self.ids.n_iter.disabled = False
            self.ids.epoch_start.disabled = True
            self.ids.epoch_end.disabled = True
            self.ids.sb_method.disabled = True
            self.ids.item_sb_method.disabled = True
            self.ids.n_sbands.disabled = True
            self.ids.overlap.active = True
            self.ids.overlap.disabled = True
            self.ids.item_overlap.disabled = True
            self.ids.filt_approach.disabled = True
            self.ids.item_filt_approach.disabled = True
            self.ids.f_low.disabled = True
            self.ids.f_high.disabled = True
            self.ids.f_order.disabled = True
            self.ids.csp_nei.disabled = True
            self.ids.clf1.disabled = True
            self.ids.item_clf1.disabled = True
            self.ids.svm_c.disabled = True
            self.ids.svm_kernel.disabled = True
            self.ids.item_svm_kernel.disabled = True
            self.ids.metric_knn.disabled = True
            self.ids.neighbors_knn.disabled = True
            self.ids.dtree_criterion.disabled = True
            self.ids.item_dtree_criterion.disabled = True
            self.ids.mlp_eta.disabled = True
            self.ids.mlp_hidden_size.disabled = True
            self.ids.mlp_hidden_neurons.disabled = True
            self.ids.mlp_activation.disabled = True
            self.ids.item_mlp_activation.disabled = True
            self.ids.lda_solver.disabled = True
            self.ids.item_lda_solver.disabled = True
            # self.ids.clf2.disabled = True
            # self.ids.item_clf2.disabled = True
            # self.ids.mlp_alpha.disabled = True
        else:
            self.epoch_lim_max = self.session.dp.eeg_info['trial_tpause']
            self.epoch_lim_min = 0 - self.session.dp.eeg_info['trial_tcue']
            self.srate = self.session.dp.eeg_info['fs']
            # self.ids.clf2.disabled = False
            # self.ids.item_clf2.disabled = False
            # self.ids.mlp_alpha.disabled = False
            # self.ids.buf_len.disabled = False
            self.ids.item_load_last_setup.disabled = True
            self.ids.load_last_setup.disabled = True
            self.ids.n_iter.disabled = True
            self.ids.epoch_start.disabled = False
            self.ids.epoch_end.disabled = False
            self.ids.sb_method.disabled = False
            self.ids.item_sb_method.disabled = False
            self.ids.n_sbands.disabled = False
            self.ids.overlap.disabled = False
            self.ids.item_overlap.disabled = False
            self.ids.filt_approach.disabled = False
            self.ids.item_filt_approach.disabled = False
            self.ids.f_low.disabled = False
            self.ids.f_high.disabled = False
            self.ids.f_order.disabled = False
            self.ids.csp_nei.disabled = False
            self.ids.clf1.disabled = False
            self.ids.item_clf1.disabled = False
            self.ids.svm_c.disabled = False
            self.ids.svm_kernel.disabled = False
            self.ids.item_svm_kernel.disabled = False
            self.ids.neighbors_knn.disabled = False
            self.ids.metric_knn.disabled = False
            self.ids.item_metric_knn.disabled = False
            self.ids.dtree_criterion.disabled = False
            self.ids.item_dtree_criterion.disabled = False
            self.ids.mlp_eta.disabled = False
            self.ids.mlp_hidden_size.disabled = False
            self.ids.mlp_hidden_neurons.disabled = False
            self.ids.mlp_activation.disabled = False
            self.ids.item_mlp_activation.disabled = False
            self.ids.mlp_solver.disabled = False
            self.ids.item_mlp_solver.disabled = False
            self.ids.lda_solver.disabled = False
            self.ids.item_lda_solver.disabled = False

            if self.ids.filt_approach.text not in ['FIR', 'IIR']:
                self.ids.f_order.disabled = True

            if not self.ids.sb_method.active:
                self.ids.n_sbands.disabled = True
                self.ids.overlap.disabled = True

            if self.ids.clf1.text == 'MLP':
                self.ids.svm_c.disabled = True
                self.ids.svm_kernel.disabled = True
                self.ids.item_svm_kernel.disabled = True
                self.ids.metric_knn.disabled = True
                self.ids.item_metric_knn.disabled = True
                self.ids.neighbors_knn.disabled = True
                self.ids.dtree_criterion.disabled = True
                self.ids.item_dtree_criterion.disabled = True
                self.ids.lda_solver.disabled = True
                self.ids.item_lda_solver.disabled = True
            elif self.ids.clf1.text == 'SVM':
                # self.ids.mlp_alpha.disabled = True
                self.ids.neighbors_knn.disabled = True
                self.ids.metric_knn.disabled = True
                self.ids.item_metric_knn.disabled = True
                self.ids.dtree_criterion.disabled = True
                self.ids.item_dtree_criterion.disabled = True
                self.ids.mlp_eta.disabled = True
                self.ids.mlp_hidden_size.disabled = True
                self.ids.mlp_hidden_neurons.disabled = True
                self.ids.mlp_activation.disabled = True
                self.ids.item_mlp_activation.disabled = True
                self.ids.mlp_solver.disabled = True
                self.ids.item_mlp_solver.disabled = True
                self.ids.lda_solver.disabled = True
                self.ids.item_lda_solver.disabled = True
            elif self.ids.clf1.text == 'DTree':
                # self.ids.mlp_alpha.disabled = True
                self.ids.svm_c.disabled = True
                self.ids.mlp_solver.disabled = True
                self.ids.item_mlp_solver.disabled = True
                self.ids.svm_kernel.disabled = True
                self.ids.item_svm_kernel.disabled = True
                self.ids.neighbors_knn.disabled = True
                self.ids.metric_knn.disabled = True
                self.ids.item_metric_knn.disabled = True
                self.ids.mlp_eta.disabled = True
                self.ids.mlp_hidden_size.disabled = True
                self.ids.mlp_hidden_neurons.disabled = True
                self.ids.mlp_activation.disabled = True
                self.ids.item_mlp_activation.disabled = True

                self.ids.lda_solver.disabled = True
                self.ids.item_lda_solver.disabled = True
            elif self.ids.clf1.text == 'KNN':
                # self.ids.mlp_alpha.disabled = True
                self.ids.svm_c.disabled = True
                self.ids.svm_kernel.disabled = True
                self.ids.item_svm_kernel.disabled = True
                self.ids.dtree_criterion.disabled = True
                self.ids.item_dtree_criterion.disabled = True
                self.ids.mlp_solver.disabled = True
                self.ids.item_mlp_solver.disabled = True
                self.ids.mlp_eta.disabled = True
                self.ids.mlp_hidden_size.disabled = True
                self.ids.mlp_hidden_neurons.disabled = True
                self.ids.mlp_activation.disabled = True
                self.ids.item_mlp_activation.disabled = True
                self.ids.lda_solver.disabled = True
                self.ids.item_lda_solver.disabled = True
            elif self.ids.clf1.text == 'LDA':
                # self.ids.mlp_alpha.disabled = True
                self.ids.svm_c.disabled = True
                self.ids.svm_kernel.disabled = True
                self.ids.item_svm_kernel.disabled = True
                self.ids.neighbors_knn.disabled = True
                self.ids.metric_knn.disabled = True
                self.ids.item_metric_knn.disabled = True
                self.ids.dtree_criterion.disabled = True
                self.ids.item_dtree_criterion.disabled = True
                self.ids.mlp_solver.disabled = True
                self.ids.item_mlp_solver.disabled = True
                self.ids.mlp_eta.disabled = True
                self.ids.mlp_hidden_size.disabled = True
                self.ids.mlp_hidden_neurons.disabled = True
                self.ids.mlp_activation.disabled = True
                self.ids.item_mlp_activation.disabled = True
            else:
                # self.ids.mlp_alpha.disabled = True
                self.ids.svm_c.disabled = True
                self.ids.svm_kernel.disabled = True
                self.ids.item_svm_kernel.disabled = True
                self.ids.neighbors_knn.disabled = True
                self.ids.metric_knn.disabled = True
                self.ids.item_metric_knn.disabled = True
                self.ids.dtree_criterion.disabled = True
                self.ids.mlp_solver.disabled = True
                self.ids.item_mlp_solver.disabled = True
                self.ids.mlp_eta.disabled = True
                self.ids.mlp_hidden_size.disabled = True
                self.ids.mlp_hidden_neurons.disabled = True
                self.ids.mlp_activation.disabled = True
                self.ids.item_dtree_criterion.disabled = True
                self.ids.item_mlp_activation.disabled = True
                self.ids.lda_solver.disabled = True
                self.ids.item_lda_solver.disabled = True


class PopupMl(BoxLayout):
    def __init__(self, session, score, **kwargs):
        super(PopupMl, self).__init__(**kwargs)
        self.session = session
        self.score = score

        self.orientation = 'vertical'
        # autoBox = BoxLayout()
        # l1 = Label(text='Self Val Acc: ', font_size=FONT_SIZE)
        # l2 = Label(text=str(autoscore) + '%', font_size=FONT_SIZE)
        # autoBox.add_widget(l1)
        # autoBox.add_widget(l2)

        # valBox = BoxLayout()
        # l3 = Label(text=' Val Acc:', font_size=FONT_SIZE)
        # l4 = Label(text=str(valscore) + '%', font_size=FONT_SIZE)
        # valBox.add_widget(l3)
        # valBox.add_widget(l4)

        crossvalBox = BoxLayout()
        l3 = Label(text='Accuracy: ', font_size=FONT_SIZE)
        l4 = Label(text=str(self.score) + '%', font_size=FONT_SIZE)
        crossvalBox.add_widget(l3)
        crossvalBox.add_widget(l4)

        # self.add_widget(autoBox)
        # self.add_widget(valBox)
        self.add_widget(crossvalBox)

