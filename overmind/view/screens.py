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
from processing.utils import save_npy_data, load_npy_data, PATH_TO_SESSION, extractEpochs, nanCleaner, save_pickle_data, load_pickle_data

# constants
FONT_SIZE = 20
BUTTON_SIZE = (300, 50)
BUTTON_BOX_SIZE = (1, 0.4)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class StartScreen(Screen):  # layout - A classe implementa Screen o gerenciador de tela kivy (screenmanager)
    login = ObjectProperty(None)  # var python para armazenar objeto kivy
    label_msg = StringProperty('')  # var python para armazenar valor string oriundo do kivy
    session_text = StringProperty('')

    def __init__(self, session, **kwargs):  # session_header recebe a sessão_do_usuário ativo
        super(StartScreen, self).__init__(
            **kwargs)  # atribui a lista de argumentos (Keywordsargs) à superclasse implementada Screen
        self.session = session  # contém a sessão do usuário e seus atributos e métodos

    def on_pre_enter(self, *args):
        Window.bind(on_request_close=self.exit)

    def exit(self, *args, **kwargs):
        box = BoxLayout(orientation='vertical', padding=10, spacing=10, size_hint=(1, None))
        botoes = BoxLayout(padding=10, spacing=10, size_hint=(1, None), height=50)
        pop = Popup(title='Do you really want to close?', content=box, size_hint=(None, None), size=(300, 180))
        btnYes = Button(text='Yes', on_release=App.get_running_app().stop)
        btnNo = Button(text='No', on_release=pop.dismiss)
        botoes.add_widget(btnYes)
        botoes.add_widget(btnNo)
        box.add_widget(botoes)
        pop.open()
        return True

    def change_to_register(self, *args):
        self.manager.current = 'Register'
        self.manager.transition.direction = 'left'

    def update_screen(self):
        self.session.info.flag = False
        self.session.saveSession()
        print(self.session.info.nickname, 'saiu ({})'.format(self.session.info.flag))
        # self.session = UserSession()
        self.label_msg = ''
        self.session_text = ''

    def check_login(self, *args):
        sname = self.login.text  # self.ids.usuario.text

        if not os.path.isdir(PATH_TO_SESSION): os.makedirs(PATH_TO_SESSION)

        if sname == '':  # if no login is provided, use latest modified folder in data/session
            all_subdirs = []
            for d in os.listdir(PATH_TO_SESSION + '.'):
                bd = os.path.join(PATH_TO_SESSION, d)
                if os.path.isdir(bd): all_subdirs.append(bd)

            if all_subdirs != []:
                sname = max(all_subdirs, key=os.path.getmtime).split('/')[-1]  # pega diretorio modificado mais recentemente
                box = BoxLayout(orientation='vertical', padding=10, spacing=10)
                pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(300, 180))
                label = Label(text='Last saved user was selected: ' + sname)
                botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
                box.add_widget(label)
                box.add_widget(botao)
                pop.open()
                self.session.info.nickname = sname  # atribui o nome da sessão salva mais recente ao atributo name da nova sessão
                self.session_text = sname
                self.session.loadSession() # carrega os dados de sessão existentes do usuário sname
                self.label_msg = "Last saved user was selected: " + sname
                self.session.info.flag = True
                # self.session.saveSession()
                print(self.session.info.nickname, 'entrou ({})'.format(self.session.info.flag))
                self.change_to_bci()
            else:
                box = BoxLayout(orientation='vertical', padding=10, spacing=10)
                pop = Popup(title='Error', content=box, size_hint=(None, None), size=(300, 180))
                label = Label(text='No users found!')
                botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
                box.add_widget(label)
                box.add_widget(botao)
                pop.open()
        else:
            if os.path.isdir(PATH_TO_SESSION + sname):  # se já existir usuário com o nome sname
                box = BoxLayout(orientation='vertical', padding=10, spacing=10)
                pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(300, 180))
                label = Label(text='User ' + sname + ' found, data was be loaded.')
                botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
                box.add_widget(label)
                box.add_widget(botao)
                pop.open()

                self.label_msg = "Session " + sname + " found. Data was be loaded."
                self.session_text = sname
                self.session.info.nickname = sname  # atribui o nome da sessão salva mais recente ao atributo name da nova sessão
                self.session.loadSession()  # carrega os dados de sessão existentes do usuário sname
                self.session.info.flag = True
                print(self.session.info.nickname, 'entrou ({})'.format(self.session.info.flag))
                self.change_to_bci()
            else:  # se ainda não existir usuário com o nome sname
                box = BoxLayout(orientation='vertical', padding=10, spacing=10)
                pop = Popup(title='Error', content=box, size_hint=(None, None), size=(300, 180))
                label = Label(text='User ' + sname + ' not found!')
                botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
                box.add_widget(label)
                box.add_widget(botao)
                pop.open()
                self.session_text = ''
                self.ids.usuario.text = self.session_text

    def change_to_bci(self, *args):
        self.manager.current = 'BCIMenu'
        self.manager.transition.direction = 'left'


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class Register(Screen):
    subj_list = None
    ds_list = ('III 3a', 'III 4a', 'IV 2a', 'IV 2b', 'LEE 54')  # sorted(os.listdir('/mnt/dados/eeg_data/eeg_epochs/'))

    def __init__(self, session, **kwargs):
        super(Register, self).__init__(**kwargs)
        self.session = session
        self.nick = None

    def back_to_start(self, *args):
        self.manager.current = 'Start'
        self.manager.transition.direction = 'right'

    def enable_dataset(self, *args):
        if self.ids.belongs_dataset.active:
            self.ids.item_dataset.disabled = False
            self.ids.item_subject.disabled = False
            self.ids.field_dataset.disabled = False
            self.ids.field_subject.disabled = False
        else:
            self.ids.item_dataset.disabled = True
            self.ids.item_subject.disabled = True
            self.ids.field_dataset.disabled = True
            self.ids.field_subject.disabled = True

    def clear_fields(self, *args):
        self.ids.field_nickname.value = ''
        self.ids.field_fullname.value = ''
        self.ids.field_age.value = 25
        self.ids.field_gender.text = '< Select >'
        self.ids.belongs_dataset.active = False
        self.ids.field_dataset.text = '< Select >'
        self.ids.field_subject.text = '< Select >'
        self.ids.item_dataset.disabled = True
        self.ids.item_subject.disabled = True
        self.ids.field_dataset.disabled = True
        self.ids.field_subject.disabled = True
        self.ids.field_srate.disabled = True
        self.ids.field_srate.value = 250

    def update_user(self, *args):
        self.session.saveSession()
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Warning', content=box, size_hint=(None, None), size=(300, 180))
        label = Label(text='Info about  ' + self.nick + '  already updated.')
        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
        box.add_widget(label)
        box.add_widget(botao)
        pop.open()
        self.back_to_start()

    def save_user(self, *args):
        os.makedirs(PATH_TO_SESSION + self.nick)
        self.session.saveSession()
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Warning', content=box, size_hint=(None, None), size=(300, 180))
        label = Label(text='User  ' + self.nick + '  successfully registered.')
        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
        box.add_widget(label)
        box.add_widget(botao)
        pop.open()
        self.back_to_start()

    def popup_update(self, *args):
        # popup usuário com este nickname já cadastrado, deseja sobrescrever as informações
        box = BoxLayout(orientation='vertical', padding=10, spacing=10, size_hint=(1, None))
        botoes = BoxLayout(padding=10, spacing=10, size_hint=(1, None), height=50)
        pop = Popup(title='User  ' + self.nick + '  already exists. Do you want overwritten info?', content=box,
                    size_hint=(None, None), size=(300, 180))
        btnYes = Button(text='Yes', on_press=self.update_user, on_release=pop.dismiss)
        btnNo = Button(text='No', on_release=pop.dismiss)
        botoes.add_widget(btnYes)
        botoes.add_widget(btnNo)
        box.add_widget(botoes)
        pop.open()

    def update_subject_values(self, *args):
        if self.ids.field_dataset.text == 'III 3a':
            self.subj_list = ['K3', 'K6', 'L1']
            self.ids.field_srate.value = 250
        elif self.ids.field_dataset.text == 'III 4a':
            self.subj_list = ['aa', 'al', 'av', 'aw', 'ay']
            self.ids.field_srate.value = 100
        elif self.ids.field_dataset.text in ['IV 2a', 'IV 2b']:
            self.subj_list = list(map(lambda x: str(x), np.arange(1, 10)))
            self.ids.field_srate.value = 250
        elif self.ids.field_dataset.text in ['LEE 54']:
            self.subj_list = list(map(lambda x: str(x), np.arange(1, 55)))
            self.ids.field_srate.value = 250
        else:
            self.subj_list = ['--']
            self.ids.field_srate.value = 250
        self.ids.field_subject.values = self.subj_list


    def check_register(self, *args):
        self.nick = self.ids.field_nickname.value
        if self.nick == '':
            box = BoxLayout(orientation='vertical', padding=10, spacing=10)
            pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(300, 180))
            label = Label(text='You need to set a nickname!')
            botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
            box.add_widget(label)
            box.add_widget(botao)
            pop.open()
        else:
            self.session.info.nickname = self.nick
            self.session.info.age = self.ids.field_age.value
            self.session.info.gender = self.ids.field_gender.text
            self.session.acq.sample_rate = self.ids.field_srate.value
            if self.ids.field_fullname.value != '':
                self.session.info.fullname = self.ids.field_fullname.value
                if self.ids.belongs_dataset.active:
                    if self.ids.field_dataset.text != '< Select >' and self.ids.field_subject.text != '< Select >' :
                        self.session.info.is_dataset = True
                        self.session.info.ds_name = self.ids.field_dataset.text
                        self.session.info.ds_subject = self.ids.field_subject.text

                        if not os.path.isdir(PATH_TO_SESSION): os.makedirs(PATH_TO_SESSION)
                        if os.path.isdir(PATH_TO_SESSION + self.nick): self.popup_update()
                        else: self.save_user()  # caso ainda não existir usuário com o nick

                    else:
                        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
                        pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(300, 180))
                        label = Label(text='You need to associate a dataset and subject!')
                        botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
                        box.add_widget(label)
                        box.add_widget(botao)
                        pop.open()
                else:
                    self.session.info.is_dataset = False
                    if not os.path.isdir(PATH_TO_SESSION): os.makedirs(PATH_TO_SESSION)
                    if os.path.isdir(PATH_TO_SESSION + self.nick): self.popup_update()
                    else: self.save_user()  # caso ainda não existir usuário com o nick
            else:
                box = BoxLayout(orientation='vertical', padding=10, spacing=10)
                pop = Popup(title='Attention', content=box, size_hint=(None, None), size=(300, 180))
                label = Label(text='You need to set a fullname!')
                botao = Button(text='Ok', on_release=pop.dismiss, size_hint=(None, None), size=(50, 30))
                box.add_widget(label)
                box.add_widget(botao)
                pop.open()


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class BCIMenu(Screen):
    label_command = StringProperty('Live Command')

    def __init__(self, session, **kwargs):
        super(BCIMenu, self).__init__(**kwargs)
        self.session = session

    def change_to_acquisition(self, *args):
        self.manager.current = 'AcqMode'
        self.manager.transition.direction = 'left'

    def change_to_calibration(self, *args):
        self.manager.current = 'CalLoad'
        self.manager.transition.direction = 'left'

    def change_to_command(self, *args):
        self.manager.current = 'ControlMenu'
        self.manager.transition.direction = 'left'

    def close_and_back(self, *args):
        self.manager.current = 'Start'
        self.manager.transition.direction = 'right'

    def exit(self, *args):
        # popup usuário com este nickname já cadastrado, deseja sobrescrever as informações
        box = BoxLayout(orientation='vertical', padding=10, spacing=10, size_hint=(1, None))
        botoes = BoxLayout(padding=10, spacing=10, size_hint=(1, None), height=50)
        pop = Popup(title='Do you really want to exit ' + self.session.info.fullname + '?', content=box,
                    size_hint=(None, None), size=(300, 180))
        btnYes = Button(text='Yes', on_press=self.close_and_back, on_release=pop.dismiss)
        btnNo = Button(text='No', on_release=pop.dismiss)
        botoes.add_widget(btnYes)
        botoes.add_widget(btnNo)
        box.add_widget(botoes)
        pop.open()

    def update_screen(self, *args):
        print(self.session.info.nickname, self.session.info.flag)
        if self.session.info.is_dataset:
            #self.ids.acq_button.disabled = True
            self.label_command = 'Simu Command'
            # self.ids.box.remove_widget(self.ids.acq_button)
            # self.ids.command_button.text = 'Simu Command'

        else:
            #self.ids.acq_button.disabled = False
            self.label_command = 'Live Command'
            # self.ids.command_button.text = 'Live Command'
            # self.ids.box.add_widget(self.ids.acq_button)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class AcqMode(Screen):

    def __init__(self, session, **kwargs):  # layout
        super(AcqMode, self).__init__(**kwargs)
        self.session = session
        self.daisy = False

    def back_to_bci(self, *args):
        self.manager.current = 'BCIMenu'
        self.manager.transition.direction = 'right'

    def change_to_acq_protocol(self, *args):
        # print(self.session.acq.daisy, self.session.acq.sample_rate, self.session.acq.flag, self.session.acq.mode,
        #     self.session.acq.com_port, self.session.acq.ch_labels, self.session.acq.path_to_file, self.session.acq.board)
        self.session.acq.mode = 'simu' if self.ids.simulator.active else 'openbci'
        self.session.acq.sample_rate = self.ids.srate.value
        self.session.acq.flag_mode = True

        if self.session.acq.mode == 'simu':
            self.session.acq.board = None
            self.session.acq.com_port = None
            self.session.acq.daisy = None
            self.session.acq.ch_labels = self.ids.ch_labels.value
            self.session.acq.class_ids = list(map(int, self.ids.class_ids.text.split(' ')))
            self.session.acq.eeg_path = self.ids.eeg_path.value
            self.session.acq.dummy = None # self.ids.dummy.value
        else:
            self.session.acq.board = self.ids.acq_board.text
            self.session.acq.com_port = self.ids.com_port.value
            self.session.acq.daisy = self.ids.daisy.value
            self.session.acq.ch_labels = self.ids.ch_labels.value
            self.session.acq.class_ids = [1, 2]
            self.session.acq.eeg_path = None
            self.session.acq.dummy = None

        self.session.saveSession()
        self.manager.current = 'AcqProtocol'
        self.manager.transition.direction = 'left'

    def load_saved_settings(self):

        # print(self.session.acq.ch_labels)
        self.ids.ch_labels.value = self.session.acq.ch_labels
        self.ids.srate.value = self.session.acq.sample_rate

        if self.session.acq.mode == 'simu':

            self.ids.item_acq_board.disabled = True
            self.ids.acq_board.disabled = True
            self.ids.acq_board.text = '< Select >'
            self.ids.com_port.disabled = True
            self.ids.ch_labels.disabled = True
            self.ids.daisy.disabled = True

            self.ids.simulator.active = True
            self.ids.eeg_path.disabled = False
            self.ids.class_ids.disabled = False
            self.ids.eeg_path.value = self.session.acq.eeg_path
            self.ids.class_ids.text = '< Select >' if self.session.acq.class_ids is None else str(
                self.session.acq.class_ids[0]) + ' ' + str(self.session.acq.class_ids[1])
            self.update_settings()

        else:
            self.ids.simulator.active = False
            self.ids.eeg_path.disabled = True
            self.ids.eeg_path.value = ''

            self.ids.item_acq_board.disabled = False
            self.ids.acq_board.disabled = False
            self.ids.acq_board.text = self.session.acq.board
            self.ids.com_port.value = '' if self.session.acq.com_port is None else self.session.acq.com_port
            self.ids.daisy.value = self.session.acq.daisy

            if self.ids.com_port.value == '' or self.ids.acq_board.text == '< Select >':
                self.ids.btn_next_step_acq.disabled = True
            else: self.ids.btn_next_step_acq.disabled = False

            if self.ids.acq_board.text == '< Select >':
                self.ids.com_port.disabled = True
                self.ids.ch_labels.disabled = True
                self.ids.daisy.disabled = True
            else:
                self.ids.com_port.disabled = False
                self.ids.ch_labels.disabled = False
                self.ids.daisy.disabled = False

    def update_settings(self):
        if self.ids.simulator.active:
            self.ids.eeg_path.disabled = False
            self.ids.item_acq_board.disabled = True
            self.ids.acq_board.disabled = True
            self.ids.acq_board.text = '< Select >'
            self.ids.com_port.disabled = True
            self.ids.ch_labels.disabled = True
            self.ids.daisy.disabled = True

            if self.ids.eeg_path.value != '':
                try:
                    self.data, self.events, self.info = np.load(self.ids.eeg_path.value, allow_pickle=True) # load_pickle_data(self.ids.eeg_path.value)
                    self.ids.srate.value = self.info['fs']
                    self.ids.ch_labels.value = str(self.info['ch_labels'])
                    self.ids.item_class_ids.disabled = False
                    self.ids.class_ids.disabled = False
                    # print(self.data.shape)
                    if self.info['class_ids'] == [1, 2, 3, 4]:
                        self.ids.class_ids.values = ['< Select >', '1 2', '1 3', '1 4', '2 3', '2 4', '3 4']
                    elif self.info['class_ids'] == [1, 3]:
                        self.ids.class_ids.values = ['< Select >', '1 3']
                        self.ids.class_ids.text = '1 3'
                    else:
                        self.ids.class_ids.values = ['< Select >', '1 2']
                        self.ids.class_ids.text = '1 2'
                except:
                    self.ids.class_ids.text = '< Select >'
                    self.ids.item_class_ids.disabled = True
                    self.ids.class_ids.disabled = True

            self.ids.btn_next_step_acq.disabled = True if (self.ids.eeg_path.value=='' or self.ids.class_ids.text=='< Select >') else False

        else:
            self.ids.eeg_path.disabled = True
            self.ids.item_acq_board.disabled = False
            self.ids.acq_board.disabled = False

            self.ids.srate.value = 125 if self.ids.daisy.value else 250
            self.ids.btn_next_step_acq.disabled = True if (self.ids.com_port.value=='' or self.ids.acq_board.text=='< Select >') else False

            if self.ids.acq_board.text == '< Select >':
                self.ids.com_port.disabled = True
                self.ids.ch_labels.disabled = True
                self.ids.daisy.disabled = True
            else:
                self.ids.com_port.disabled = False
                self.ids.ch_labels.disabled = False
                self.ids.daisy.disabled = False

class Menu(GridLayout):
    pass


class SettingsScreens(ScreenManager):
    simulator = ObjectProperty(None)
    openbci = ObjectProperty(None)


class Simulator(Screen):
    eeg_path = StringProperty('')
    labels_path = StringProperty('')
    srate = NumericProperty(0)


class OpenBCI(Screen):
    pass


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class AcqProtocol(Screen):
    def __init__(self, session, **kwargs):  # layout
        super(AcqProtocol, self).__init__(**kwargs)
        self.session = session
        self.daisy = False

    def back_to_bci(self, *args):
        self.manager.current = 'AcqMode'
        self.manager.transition.direction = 'right'

    def run_acq(self, *args):
        self.session.acq.n_runs = self.ids.n_runs.value  # Número de Execuções - conjunto de tentativas
        self.session.acq.runs_interval = self.ids.runs_interval.value  # Intervalo de Tempo em segundos entre execuções
        self.session.acq.n_trials = self.ids.n_trials.value  # Número de tentativas a serem exibidas por execução (deve ser PAR)
        self.session.acq.cue_offset = self.ids.cue_offset.value  # Momento para o início da apresentação da dica (s) (a partir do início da tentativa)
        self.session.acq.cue_time = self.ids.cue_time.value  # Duração da apresentação da dica (s)
        self.session.acq.min_pause = self.ids.pause_min.value  # Pausa entre tentativas (s)
        self.session.acq.trial_duration = self.ids.trial_duration.value  # Duração total da tentativa (s)
        self.session.acq.flag_protocol = True
        self.session.saveSession()
        self.manager.current = 'AcqRun'
        self.manager.transition.direction = 'left'

    def update_settings(self):
        # atualiza a sessão do usuário ativo com as informações oriundas do formulário da UI acquisition_settings.kv
        self.ids.n_runs.value = self.session.acq.n_runs
        self.ids.runs_interval.value = self.session.acq.runs_interval
        self.ids.n_trials.value = self.session.acq.n_trials
        self.ids.cue_offset.value = self.session.acq.cue_offset
        self.ids.cue_time.value = self.session.acq.cue_time
        self.ids.pause_min.value = self.session.acq.min_pause
        self.ids.trial_duration.value = self.session.acq.trial_duration

    def update_trial_duration(self):
        self.ids.trial_duration.value = self.ids.cue_offset.value + self.ids.cue_time.value + self.ids.pause_min.value + 1


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class AcqRun(Screen):
    src = ["view/resources/cross2.png",
           "view/resources/left2.png",
           "view/resources/right2.png",
           "view/resources/blank.png",
           "view/resources/break.png"]
    fig_list = ListProperty(src)
    button_stream = StringProperty('Start Streaming')
    carousel = ObjectProperty(None)
    inst_prob_left = NumericProperty(0)
    inst_prob_right = NumericProperty(0)
    accum_color_left = ListProperty([1, 0, 0, 1])
    accum_color_right = ListProperty([0, 0, 1, 1])

    def __init__(self, session, **kwargs):
        super(AcqRun, self).__init__(**kwargs)
        self.session = session
        self.carousel.index = 3  # 3 == view/resources/blank.png
        self.stream_flag = False  # indica status do streaming (True = Operando; False = Parado)

    def back_to_acq(self, *args):
        self.manager.current = 'AcqProtocol'
        self.manager.transition.direction = 'right'

    def toggle_stream(self, *args):  #### step 1 - after press on Start Streaming ## chaveia o streaming
        if self.stream_flag:  # se True
            self.stream_stop()
            self.stop_stimulus()
        else:
            self.stream_start()

    def stream_start(self):  ### step 2
        # print(self.session.acq.dummy)
        self.sm = SampleManager(self.session.acq.sample_rate,
                                self.session.acq.com_port,
                                # self.session.dp.buf_len,
                                class_ids=self.session.acq.class_ids,
                                daisy=self.session.acq.daisy,
                                mode=self.session.acq.mode,
                                path=self.session.acq.eeg_path,
                                # labels_path=self.session.acq.path_to_labels_file,
                                # dummy=self.session.acq.daisy
                                )

        # print(self.sm.loadedData)
        # self.sm.daemon = True
        self.sm.stop_flag = False
        self.sm.start()  # inicia thread - chama def run() em sample_manager.py
        self.button_stream = 'Stop Streaming'  # seta o label do botão na tela cal_start 'Parar Transmiss\u00e3o'
        self.stream_flag = True
        self.start_stimulus()
        self.set_bar_default()

    def stream_stop(self):
        self.sm.stop_flag = True
        self.stream_flag = False
        self.sm.join()  ### encadeamento de chamada fica bloqueado até o objeto de encadeamento no qual foi chamado terminar
        self.button_stream = 'Start Streaming'
        self.set_bar_default()
        self.acq_info = {
            'fs': self.session.acq.sample_rate,
            'class_ids': list(np.unique(self.stim_list)),
            'trial_tcue': self.session.acq.cue_offset,
            'trial_tpause': self.session.acq.cue_offset + self.session.acq.cue_time,
            'trial_mi_time': self.session.acq.cue_time,
            'trials_per_class': (self.session.acq.n_trials * self.session.acq.n_runs) / 2,
            'eeg_channels': self.sm.loadedData.shape[0],
            'ch_labels': self.session.acq.ch_labels,
            'datetime': datetime.now().strftime('%d-%m-%Y_%Hh%Mm')
        }

        # self.save_data()  # save 1
        save_acq = AcqSavePopup(self.session, self.sm, self.acq_info)
        save_acq.open() # save 2, Popup mode

    def start_stimulus(self):
        self.epoch_counter = 0
        self.run_counter = 0
        self.generate_stim_list()
        self.start_run(None)

    def generate_stim_list(self):
        nt = self.session.acq.n_trials * self.session.acq.n_runs
        idA = self.session.acq.class_ids[0] * np.ones(int(nt / 2))  # original sem o int
        idB = self.session.acq.class_ids[1] * np.ones(int(nt / 2))  # original sem o int
        slist = np.concatenate([idA, idB])
        random.shuffle(slist)  # embaralha vetor
        self.stim_list = slist.astype(int)

    def stop_stimulus(self):
        Clock.unschedule(self.display_epoch)
        Clock.unschedule(self.start_run)
        Clock.unschedule(self.set_pause)
        Clock.unschedule(self.set_cue)
        Clock.unschedule(self.set_blank)
        self.carousel.index = 3

    def start_run(self, dt):
        self.run_epoch_counter = 0
        self.carousel.index = 3  # 3 para manipular as etapas e imagens do protocolo
        Clock.schedule_interval(self.display_epoch, self.session.acq.trial_duration)

    def stop_run(self):
        self.stop_stimulus()
        self.run_counter += 1
        if self.run_counter < self.session.acq.n_runs:
            Clock.schedule_once(self.start_run, self.session.acq.runs_interval)
            self.carousel.index = 4
        else:
            self.stream_stop()
            self.stop_stimulus()

    def display_epoch(self, dt):
        st = time.time()
        if self.run_epoch_counter < self.session.acq.n_trials:
            Clock.schedule_once(self.set_start_trial)
            Clock.schedule_once(self.set_cue, self.session.acq.cue_offset)
            Clock.schedule_once(self.set_blank, self.session.acq.cue_offset + self.session.acq.cue_time)
            #self.pause_time = self.session.acq.min_pause # random.uniform(self.session.acq.min_pause, self.session.acq.min_pause + 2)
            #Clock.schedule_once(self.set_pause, self.session.acq.cue_offset + self.session.acq.cue_time + self.pause_time)
        else:
            self.stop_run()
    def set_start_trial(self, dt):
        self.carousel.index = 0  # original comentado
        self.sm.MarkEvents(0)
        self.beep()

    def set_pause(self, dt):
        # print(self.session.acq.cue_offset, self.session.acq.cue_time, self.pause_time)
        pass

    def set_cue(self, dt):
        if self.stim_list[self.epoch_counter] == self.session.acq.class_ids[0]:
            self.carousel.index = self.session.acq.class_ids[0]  # original comentado
            #self.sm.event_list[-1, 1] = 1
            self.sm.MarkEvents(self.session.acq.class_ids[0]) #101
            anim_left = threading.Thread(target=self.animate_bar_left)
            anim_left.start()
        elif self.stim_list[self.epoch_counter] == self.session.acq.class_ids[1]:
            self.carousel.index = self.session.acq.class_ids[1]  # original comentado
            #self.sm.event_list[-1, 1] = 2
            self.sm.MarkEvents(self.session.acq.class_ids[1]) #102
            anim_right = threading.Thread(target=self.animate_bar_right)
            anim_right.start()
        self.epoch_counter += 1
        self.run_epoch_counter += 1

    def set_blank(self, dt):
        self.carousel.index = 3

    def beep(self):
        os.system('play --no-show-progress --null --channels 1 synth %s sine %f &' % (0.3, 500))

    def save_data(self):
        print('Saving data')
        acq_user_path = PATH_TO_SESSION + self.session.info.nickname + '/acqs/'
        if not os.path.isdir(acq_user_path): os.makedirs(acq_user_path)
        self.sm.SaveAll(self.acq_info, acq_user_path + 'data')
        # self.sm.SaveData(acq_user_path + 'data')
        # self.sm.SaveEvents(acq_user_path + 'events')
        self.update_user_path(PATH_TO_SESSION + self.session.info.nickname + '/acqs/')

    def update_user_path(self, path):
        pattern_data = re.compile(r"data_.*?\.npy")  # mask to get filenames
        pattern_events = re.compile(r"events_.*?\.npy")  # mask to get filenames
        data_files, events_files = [], []
        for root, dirs, files in os.walk(path):  # add .kv files of view/kv in kv_files vector
            data_files += [root + file_ for file_ in files if pattern_data.match(file_)]
            events_files += [root + file_ for file_ in files if pattern_events.match(file_)]
        try:
            last_data_file = max(data_files, key=os.path.getmtime).split('/')[-1]
            last_ev_file = max(events_files, key=os.path.getmtime).split('/')[-1]
            self.session.acq.path_to_eeg_data = path + last_data_file
            self.session.acq.path_to_eeg_events = path + last_ev_file
        except:
            self.session.acq.path_to_eeg_data = None
            self.session.acq.path_to_eeg_events = None
        self.session.saveSession()



    def set_bar_default(self):
        self.inst_prob_left = 50
        self.inst_prob_right = 50
        self.accum_color_left = [1, 0, 0, 1]
        self.accum_color_right = [0, 0, 1, 1]

    def animate_bar_left(self):
        ts = time.time()
        tf = 0
        while tf < self.session.acq.cue_time and self.stream_flag:
            ratio = tf / self.session.acq.cue_time
            self.inst_prob_left = 50 + ratio * 50
            # if self.inst_prob_left > 80: self.accum_color_left = [1, 1, 0, 1]
            # else: self.accum_color_left = [1, 0, 0, 1]
            self.inst_prob_right = 100 - self.inst_prob_left
            time.sleep(0.05)
            tf = (time.time() - ts)
        self.set_bar_default()

    def animate_bar_right(self):
        ts = time.time()
        tf = 0
        while tf < self.session.acq.cue_time and self.stream_flag:
            ratio = tf / self.session.acq.cue_time
            self.inst_prob_right = 50 + ratio * 50
            # if self.inst_prob_right > 80: self.accum_color_right = [1, 1, 0, 1]
            # else: self.accum_color_right = [0, 0, 1, 1]
            self.inst_prob_left = 100 - self.inst_prob_right
            time.sleep(0.05)
            tf = (time.time() - ts)
        self.set_bar_default()


class AcqSavePopup(Popup):
    def __init__(self, session, sm, acq_info, **kwargs):
        super(AcqSavePopup, self).__init__(**kwargs)
        self.session = session
        self.sm = sm
        self.acq_info = acq_info
        print('Saving data...')

    def save_acquisition(self, save_name):
        acq_user_path = PATH_TO_SESSION + self.session.info.nickname + '/acqs/'
        if not os.path.isdir(acq_user_path): os.makedirs(acq_user_path)
        self.sm.SaveAll(self.acq_info, acq_user_path + save_name)
        # self.sm.SaveData(acq_user_path + save_name + '_data')
        # self.sm.SaveEvents(acq_user_path + save_name + '_events')
        self.session.acq.path_to_eeg_data = acq_user_path + save_name + '_data.npy'
        self.session.acq.path_to_eeg_events = acq_user_path + save_name + '_events.npy'
        self.session.saveSession()
        print('Data saved!')


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class CalLoad(Screen):
    max_channels = NumericProperty(8)

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


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class ControlMenu(Screen):
    label_text = StringProperty('Command Menu')
    box = ObjectProperty(None)

    def __init__(self, session, **kwargs):
        super(ControlMenu, self).__init__(**kwargs)
        self.session = session
        # box1 = BoxLayout(id='caixa', size_hint_x=1, size_hint_y=0.5, padding=10, spacing=10, orientation='vertical')
        # self.label_msg = Label(id='label_msg', text=self.label_text, font_size=FONT_SIZE)
        # button_bars = Button(text="Bars", size=BUTTON_SIZE)
        # button_bars.bind(on_press=self.change_to_bars)
        # button_target = Button(text="Target", size=BUTTON_SIZE)
        # button_target.bind(on_press=self.change_to_target)
        # # button_ardrone = Button(text="Ardrone", size=BUTTON_SIZE)
        # # button_ardrone.bind(on_press=self.change_to_ardrone)
        # button_settings = Button(text="Settings", size=BUTTON_SIZE)
        # button_settings.bind(on_press=self.change_to_settings)
        # button_galaxy = Button(text="Galaxy Game", size=BUTTON_SIZE)
        # button_galaxy.bind(on_press=self.change_to_galaxy)
        # button_back = Button(text="Back", size=BUTTON_SIZE)
        # button_back.bind(on_press=self.back_to_bci)
        # box1.add_widget(self.label_msg)
        # box1.add_widget(button_settings)
        # # box1.add_widget(button_ardrone)
        # box1.add_widget(button_bars)
        # box1.add_widget(button_galaxy)
        # box1.add_widget(button_target)
        # box1.add_widget(button_back)
        # self.add_widget(box1)

    def change_to_target(self, *args):
        self.manager.current = 'TargetRun'
        self.manager.transition.direction = 'left'

    def change_to_bars(self, *args):
        self.manager.current = 'BarsRun'
        self.manager.transition.direction = 'left'

    def change_to_ardrone(self, *args):
        self.manager.current = 'DroneMenu'
        self.manager.transition.direction = 'left'

    def change_to_galaxy(self, *args):
        self.manager.current = 'GalaxyMenu'
        self.manager.transition.direction = 'left'

    def change_to_settings(self, *args):
        self.manager.current = 'ControlSettings'
        self.manager.transition.direction = 'left'

    def back_to_bci(self, *args):
        self.manager.current = 'BCIMenu'
        self.manager.transition.direction = 'right'

    def update_screen(self, *args):
        if self.session.info.is_dataset: self.label_text = "Simulator Menu"
        else: self.label_text = "Command Menu"


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class ControlSettings(Screen):
    def __init__(self, session, **kwargs):
        super(ControlSettings, self).__init__(**kwargs)
        self.session = session

    def back_to_control_menu(self, *args):
        self.manager.current = 'ControlMenu'
        self.manager.transition.direction = 'right'

    def save_config(self, *args):
        self.session.control.game_threshold = self.ids.game_threshold.value
        self.session.control.window_overlap = self.ids.window_overlap.value
        self.session.control.warning_threshold = self.ids.warning_threshold.value
        self.session.control.forward_speed = self.ids.forward_speed.value / 1000.0
        self.session.control.inst_prob = self.ids.inst_prob.value / 1000.0
        self.session.control.keyb_enable = self.ids.keyb_enable.value
        self.session.control.action_cmd1 = self.ids.action_cmd1.value
        self.session.control.action_cmd2 = self.ids.action_cmd2.value
        self.session.control.flag = True

        # self.session.acq.path_to_file = self.ids.eeg_path.value
        # self.session.acq.path_to_labels_file = self.ids.labels_path.value
        # self.session.acq.sample_rate = self.ids.srate.value
        # self.session.acq.dummy = self.ids.dummy_data.value

        self.session.saveSession()

    def update_settings(self):
        self.ids.game_threshold.value = self.session.control.game_threshold
        self.ids.window_overlap.value = self.session.control.window_overlap
        self.ids.warning_threshold.value = self.session.control.warning_threshold
        self.ids.forward_speed.value = self.session.control.forward_speed * 1000.0
        self.ids.inst_prob.value = self.session.control.inst_prob * 1000.0
        self.ids.keyb_enable.value = self.session.control.keyb_enable
        self.ids.action_cmd1.value = self.session.control.action_cmd1
        self.ids.action_cmd2.value = self.session.control.action_cmd2

        # if not self.session.acq.path_to_file is None: self.ids.eeg_path.value = self.session.acq.path_to_file
        # if not self.session.acq.path_to_labels_file is None: self.ids.labels_path.value = self.session.acq.path_to_labels_file
        # self.ids.srate.value = self.session.acq.sample_rate
        # if not self.session.acq.dummy is None: self.ids.dummy_data.value = self.session.acq.dummy


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class BarsRun(Screen):
    inst_prob_left = NumericProperty(0)
    accum_prob_left = NumericProperty(0)
    accum_color_left = ListProperty([1, 0, 0, 1]) # red
    inst_prob_right = NumericProperty(0)
    accum_prob_right = NumericProperty(0)
    accum_color_right = ListProperty([0, 0, 1, 1]) # blue
    label_on_toggle_button = StringProperty('Start')
    current_label = NumericProperty(None)
    label_color = ListProperty([0, 0, 0, 1]) # black
    wt = NumericProperty(0.0)

    def __init__(self, session, **kwargs):
        super(BarsRun, self).__init__(**kwargs)
        self.session = session
        self.stream_flag = False
        self.U1 = 0.0
        self.U2 = 0.0
        self.p = [0, 0]

    def back_to_control_menu(self, *args):
        self.manager.current = 'ControlMenu'
        self.manager.transition.direction = 'right'

    def toogle_stream(self, *args):
        if self.stream_flag: self.stream_stop()
        else: self.stream_start()

    def load_setup(self):
        self.ap = Approach()
        self.ap.loadSetup(PATH_TO_SESSION + self.session.info.nickname)

    def stream_stop(self):
        self.sm.stop_flag = True
        self.stream_flag = False
        self.sm.join()
        self.label_on_toggle_button = 'Start'
        self.clock_unscheduler()
        self.set_bar_default()
        res = GameDataPopup(self.session, self.sm.all_data)
        res.open()

    def stream_start(self):
        self.load_setup()
        self.limiar = self.session.control.game_threshold
        TTA = 5. # tempo de ação
        increment = self.session.acq.sample_rate * 0.1  # 10% srate (old increment = 25)
        ABUF_LEN = TTA * self.session.acq.sample_rate / increment
        cal_acc = self.ap.learner.get_results()  # old self.ap.accuracy
        self.delta_ref = cal_acc * TTA / (increment / self.session.acq.sample_rate)
        self.U1_local = collections.deque(maxlen=int(ABUF_LEN))
        self.U2_local = collections.deque(maxlen=int(ABUF_LEN))
        self.sm = SampleManager(
            self.session.acq.sample_rate,
            self.session.acq.com_port,
            buf_len=int(self.session.dp.buf_len),
            tmin=self.session.dp.epoch_start,
            tmax=self.session.dp.epoch_end,
            class_ids=self.session.acq.class_ids,
            mode=self.session.acq.mode,
            path=self.session.acq.eeg_path,
            # labels_path=self.session.acq.path_to_labels_file,
            # daisy=self.session.acq.daisy,
            # dummy=self.session.acq.dummy
        )
        self.sm.daemon = True
        self.sm.stop_flag = False
        self.sm.start()
        self.label_on_toggle_button = 'Stop'
        self.stream_flag = True
        self.clock_scheduler()

    def clock_scheduler(self):
        Clock.schedule_interval(self.get_probs, 1. / 20.)
        Clock.schedule_interval(self.update_accum_bars, self.session.control.window_overlap)
        if self.session.acq.mode == 'simu': # and not self.session.acq.dummy and not self.session.acq.path_to_labels_file == '':
            Clock.schedule_interval(self.update_current_label, 1. / 20.)

    def clock_unscheduler(self):
        Clock.unschedule(self.get_probs)
        Clock.unschedule(self.update_current_label)
        Clock.unschedule(self.update_accum_bars)

    def get_probs(self, dt):
        t, buf = self.sm.GetBuffData()
        # print(buf.shape[0], self.session.dp.buf_len)
        if buf.shape[0] == self.session.dp.buf_len:
            self.p = self.ap.classify_epoch(buf.T, 'prob')[0]
            if self.session.control.inst_prob: self.update_inst_bars()

    def update_inst_bars(self):

        if self.p is None: return
        p1 = self.p[0]
        p2 = self.p[1]
        u = p1 - p2
        if u > 0:
            self.inst_prob_left = int(math.floor(u * 100))
            self.inst_prob_right = 0
        else:
            self.inst_prob_right = int(math.floor(abs(u) * 100))
            self.inst_prob_left = 0

    def update_accum_bars(self, dt):
        if self.p is None: return
        p1 = self.p[0]
        p2 = self.p[1]
        u = p1 - p2
        if u >= 0:
            u1 = 1
            u2 = 0
        else:
            u1 = 0
            u2 = 1

        self.U1 = self.U1 + u1
        self.U2 = self.U2 + u2
        self.U1_local.append(self.U1)
        self.U2_local.append(self.U2)

        # print(u1, u2, self.U1, self.U2, self.U1_local, self.U2_local)

        delta1 = self.U1_local[-1] - self.U1_local[0]
        delta2 = self.U2_local[-1] - self.U2_local[0]

        BAR1 = 100 * (delta1 / self.delta_ref)
        BAR2 = 100 * (delta2 / self.delta_ref)

        BAR1_n = int(math.floor(BAR1))
        BAR2_n = int(math.floor(BAR2))

        print(BAR1_n, BAR2_n)

        if BAR1_n < self.limiar: self.accum_prob_left = BAR1_n
        if BAR2_n < self.limiar: self.accum_prob_right = BAR2_n

        self.map_probs(BAR1, BAR2)

    def map_probs(self, BAR1, BAR2):
        if BAR1 > self.limiar:
            os.system(self.session.control.action_cmd1)
            self.set_bar_default()
        elif BAR2 > self.limiar:
            os.system(self.session.control.action_cmd2)
            self.set_bar_default()
        else:
            pass
            # dont send any cmd

    def set_bar_default(self):
        self.accum_prob_left = 0
        self.accum_prob_right = 0

        self.inst_prob_left = 0
        self.inst_prob_right = 0

        self.U1_local.clear()
        self.U2_local.clear()

    def update_current_label(self, dt):
        self.current_label = self.sm.current_cmd


class GameDataPopup(Popup):
    def __init__(self, session, data, **kwargs):
        super(GameDataPopup, self).__init__(**kwargs)
        self.session = session
        self.data = data

    def save_data(self, game_name):
        path = PATH_TO_SESSION + self.session.info.nickname + '/' + 'bar_data_' + game_name + '.npy'
        save_npy_data(self.data, path, mode='w')


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class TargetRun(Screen):
    inst_prob_left = NumericProperty(0)
    accum_prob_left = NumericProperty(0)
    accum_color_left = ListProperty([0, 0, 1, 1])

    inst_prob_right = NumericProperty(0)
    accum_prob_right = NumericProperty(0)
    accum_color_right = ListProperty([0, 0, 1, 1])

    label_on_toggle_button = StringProperty('Start')

    game = ObjectProperty(None)

    current_label = NumericProperty(None)

    label_color = ListProperty([0, 0, 0, 1])

    wt = NumericProperty(0.0)

    def __init__(self, session, **kwargs):
        super(TargetRun, self).__init__(**kwargs)
        self.session = session
        self.U = 0.0
        self.p = [0, 0]
        self.stream_flag = False

    # BUTTON CALLBACKS
    def back_to_control_menu(self, *args):
        self.manager.current = 'ControlMenu'
        self.manager.transition.direction = 'right'

    def toogle_stream(self, *args):
        if self.stream_flag: self.stream_stop()
        else: self.stream_start()

    def load_setup(self):
        self.ap = Approach()
        self.ap.loadSetup(PATH_TO_SESSION + self.session.info.nickname)

    def stream_stop(self):
        if self.session.control.keyb_enable:
            self.game.keyb_enable = False
        else:
            self.sm.stop_flag = True
            self.sm.join()
            self.clock_unscheduler()
            self.set_bar_default()

        self.stream_flag = False
        self.label_on_toggle_button = 'Start'
        self.game.stop()

        res = GameResultsPopup(self.session, self.game.res_h)
        res.open()

    def stream_start(self):
        self.load_setup()
        if self.session.control.keyb_enable: self.game.keyb_enable = True
        else:
            self.sm = SampleManager(
                self.session.acq.sample_rate,
                self.session.acq.com_port,
                buf_len=self.session.dp.buf_len,
                tmin=self.session.dp.epoch_start,
                tmax=self.session.dp.epoch_end,
                class_ids=self.session.acq.class_ids,
                mode=self.session.acq.mode,
                path=self.session.acq.eeg_path,
                # labels_path=self.session.acq.path_to_labels_file,
                # daisy=self.session.acq.daisy,
                # dummy=self.session.acq.dummy
            )
            self.sm.daemon = True
            self.sm.stop_flag = False
            self.sm.start()
            self.clock_scheduler()

        self.stream_flag = True
        self.label_on_toggle_button = 'Stop'
        self.game.set_player_speed(self.session.control.forward_speed)
        self.game.setup()
        self.game.start(None)

    def clock_scheduler(self):
        Clock.schedule_interval(self.get_probs, 0) #, 1. / 20.
        Clock.schedule_interval(self.update_accum_bars, self.session.control.window_overlap)
        if self.session.acq.mode == 'simu' and not self.session.acq.dummy:
            Clock.schedule_interval(self.update_current_label, 1. / 20.)

    def clock_unscheduler(self):
        Clock.unschedule(self.get_probs)
        Clock.unschedule(self.update_current_label)
        Clock.unschedule(self.update_accum_bars)

    def get_probs(self, dt):
        tBuff, circBuff = self.sm.GetBuffData()
        if circBuff.shape[0] == self.session.dp.buf_len:
            # self.p = self.ap.applyModelOnEpoch(buf.T, 'prob')[0]
            self.p = self.ap.classify_epoch(circBuff.T, out_param='prob')[0]
            if self.session.control.inst_prob: self.update_inst_bars()

    def update_inst_bars(self):
        p1 = self.p[0]
        p2 = self.p[1]
        u = p1 - p2

        if u > 0:
            self.inst_prob_left = int(math.floor(u * 100))
            self.inst_prob_right = 0
        else:
            self.inst_prob_right = int(math.floor(abs(u) * 100))
            self.inst_prob_left = 0

    def update_accum_bars(self, dt):
        print('teste')
        p1 = self.p[0]
        p2 = self.p[1]
        u = p1 - p2
        self.U += u
        U1 = 100 * (self.U + self.session.control.game_threshold) / (2. * self.session.control.game_threshold)

        U2 = 100 - U1
        U1_n = int(math.floor(U1))
        U2_n = int(math.floor(U2))

        if U1_n > self.session.control.warning_threshold:
            self.accum_color_left = [1, 1, 0, 1]
        elif U2_n > self.session.control.warning_threshold:
            self.accum_color_right = [1, 1, 0, 1]
        else:
            self.accum_color_left = [1, 0, 0, 1]
            self.accum_color_right = [0, 0, 1, 1]

        if U1_n in range(101): self.accum_prob_left = U1_n
        if U2_n in range(101): self.accum_prob_right = U2_n

        self.map_probs(U1, U2)

    def map_probs(self, U1, U2):
        # print self.game.direction
        if U1 > 100:
            self.game.set_direction(-1)
            self.set_bar_default()
            # print self.game.direction
            return 0, 0
        elif U2 > 100:
            self.game.set_direction(1)
            self.set_bar_default()
            # print self.game.direction
            return 0, 0
        else:
            return U1, U2
            # dont send any cmd

    def set_bar_default(self):
        self.accum_prob_left = 0
        self.accum_prob_right = 0
        self.inst_prob_left = 0
        self.inst_prob_right = 0
        self.U = 0.0

    def update_current_label(self, dt):
        current_label = int(self.sm.current_playback_label[1])
        self.current_label = current_label


class Game(Widget):
    player = ObjectProperty(None)
    target = ObjectProperty(None)
    vel = NumericProperty(1)
    keyb_enable = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(Game, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(None, self)
        if not self._keyboard: return
        self.direction = 'up'
        self.direction_list = ['left', 'up', 'right', 'down']
        self.direction_idx = 1
        self.on_flag = False

    def on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'left':
            self.set_direction(-1)
        elif keycode[1] == 'right':
            self.set_direction(1)
        else:
            return False
        return True

    def set_player_speed(self, speed):
        self.forward_speed = speed

    def set_positions(self):
        max_width = int(self.parent.width)
        max_height = int(self.parent.height)
        self.target.pos = (random.randint(0, max_width), random.randint(0, max_height))
        self.player.pos = self.center

    def setup(self):
        self.res_h = [0]
        if self.keyb_enable: self._keyboard.bind(on_key_down=self.on_keyboard_down)

    def start(self, dt):
        self.target.t_color = [1, 1, 0, 1]
        self.set_positions()
        self.on_flag = True
        Clock.schedule_interval(self.check_if_won, 1. / 5.)
        Clock.schedule_interval(self.move_player, self.forward_speed)
        self.time_start = time.time()

    def stop(self):
        # unbind keyboard even if it wasnt before
        self._keyboard.unbind(on_key_down=self.on_keyboard_down)
        self.on_flag = False
        Clock.unschedule(self.check_if_won)
        Clock.unschedule(self.move_player)

    def check_if_won(self, dt):
        if self.player.collide_widget(self.target):
            self.target.t_color = [0, 1, 0, 1]
            Clock.schedule_once(self.start, 2)
            self.time_stop = time.time()
            self.res_h.append(self.time_stop - self.time_start)
            Clock.unschedule(self.check_if_won)
            Clock.unschedule(self.move_player)

    def set_direction(self, direction):
        # print 'changing by:', direction
        if (self.direction_idx == 0) and (direction == -1):
            self.direction_idx = 3
        elif (self.direction_idx == 3) and (direction == 1):
            self.direction_idx = 0
        else:
            self.direction_idx += direction
        self.direction = self.direction_list[self.direction_idx]
        self.move_player(None)

    def move_player(self, dt):
        l = self.player.width
        p0 = self.player.pos[0]
        p1 = self.player.pos[1]
        # print 'moving to:', self.direction
        if self.direction == 'right':
            x0 = p0
            y0 = p1 + l
            x1 = p0 + l
            y1 = p1 + l / 2
            x2 = p0
            y2 = p1
            if self.player.center_x <= int(self.parent.width) - 15: self.player.pos[0] += self.vel
        elif self.direction == 'left':
            x0 = p0 + l
            y0 = p1
            x1 = p0
            y1 = p1 + l / 2
            x2 = p0 + l
            y2 = p1 + l
            if self.player.center_x >= 15: self.player.pos[0] -= self.vel
        elif self.direction == 'up':
            x0 = p0
            y0 = p1
            x1 = p0 + l / 2
            y1 = p1 + l
            x2 = p0 + l
            y2 = p1
            if self.player.center_y <= int(self.parent.height) - 15: self.player.pos[1] += self.vel
        elif self.direction == 'down':
            x0 = p0 + l
            y0 = p1 + l
            x1 = p0 + l / 2
            y1 = p1
            x2 = p0
            y2 = p1 + l
            if self.player.center_y >= 15: self.player.pos[1] -= self.vel
        self.player.points = [x0, y0, x1, y1, x2, y2]


class GamePlayer(Widget):
    points = ListProperty([0] * 6)


class GameTarget(Widget):
    t_color = ListProperty([1, 1, 0, 1])


class GameResultsPopup(Popup):
    res = ListProperty([0])
    hits = NumericProperty(0)

    def __init__(self, session, results, **kwargs):
        super(GameResultsPopup, self).__init__(**kwargs)
        self.session = session
        if len(results) > 1:
            self.res = results[1:]
            self.hits = len(self.res)

    def save_results(self, game_name):
        path = PATH_TO_SESSION + self.session.info.nickname + '/' + 'game_results_' + game_name + '.npy'
        r = np.array(self.res)
        save_npy_data(r, path, mode='w')


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class GalaxyMenu(Screen):
    label_on_toggle_button = StringProperty('Start')
    current_label = NumericProperty(None)
    wt = NumericProperty(0.0)

    def __init__(self, session, **kwargs):
        super(GalaxyMenu, self).__init__(**kwargs)
        self.session = session
        self.stream_flag = False
        self.U1 = 0.0
        self.U2 = 0.0
        self.p = [0, 0]

    def change_to_settingsGalaxy(self, *args):
        self.manager.current = 'GalaxySettings'
        self.manager.transition.direction = 'left'

    # def change_to_playGalaxy(self, *args):
    #     self.manager.current = 'GalaxyPlay'
    #     self.manager.transition.direction = 'left'

    def back_to_control_menu(self, *args):
        self.manager.current = 'ControlMenu'
        self.manager.transition.direction = 'right'

    def toogle_stream(self, *args):
        if self.stream_flag: self.stream_stop()
        else: self.stream_start()

    def load_setup(self):
        self.ap = Approach()
        self.ap.loadSetup(PATH_TO_SESSION + self.session.info.nickname)

    def stream_stop(self):
        self.sm.stop_flag = True
        self.stream_flag = False
        self.sm.join()

        self.gp.stop_flag = True
        self.gp.Stop()
        self.gp.join()

        self.label_on_toggle_button = 'Start'
        self.clock_unscheduler()
        # self.set_bar_default()
        # res = GameDataPopup(self.session, self.sm.all_data)
        # res.open()

    def stream_start(self):
        self.load_setup()
        self.limiar = self.session.control.game_threshold
        TTA = 5. # tempo de ação
        increment = self.session.acq.sample_rate * 0.1  # 10% srate (old increment = 25)
        ABUF_LEN = TTA * self.session.acq.sample_rate / increment
        cal_acc = self.ap.learner.get_results()  # old self.ap.accuracy
        print('current setup accuracy :', round(cal_acc*100, 2))
        self.delta_ref = cal_acc * TTA / (increment / self.session.acq.sample_rate)
        self.U1_local = collections.deque(maxlen=int(ABUF_LEN))
        self.U2_local = collections.deque(maxlen=int(ABUF_LEN))
        self.sm = SampleManager(
            self.session.acq.sample_rate,
            self.session.acq.com_port,
            buf_len=int(self.session.dp.buf_len),
            tmin=self.session.dp.epoch_start,
            tmax=self.session.dp.epoch_end,
            class_ids=self.session.acq.class_ids,
            mode=self.session.acq.mode,
            path=self.session.acq.eeg_path,
            # labels_path=self.session.acq.path_to_labels_file,
            # daisy=self.session.acq.daisy,
            # dummy=self.session.acq.dummy
        )
        self.sm.daemon = True
        self.sm.stop_flag = False
        self.sm.start()

        self.gp = GalaxyPlay(self.session)
        self.gp.daemon = True
        self.gp.stop_flag = False
        self.gp.start()

        self.label_on_toggle_button = 'Stop'
        self.stream_flag = True
        self.clock_scheduler()

    def clock_scheduler(self):
        # Clock.schedule_interval(self.galaxy_screen2, 30)
        Clock.schedule_interval(self.get_probs, 1. / 20.)
        # Clock.schedule_interval(self.update_command, 2)
        # Clock.schedule_interval(self.update_accum_bars, self.session.control.window_overlap)
        # if self.session.acq.mode == 'simu': # and not self.session.acq.dummy and not self.session.acq.path_to_labels_file == '':
        #     Clock.schedule_interval(self.update_current_label, 1. / 20.)

    def clock_unscheduler(self):
        # Clock.unschedule(self.galaxy_screen2)
        Clock.unschedule(self.get_probs)
        # Clock.unschedule(self.update_command)
        # Clock.unschedule(self.update_current_label)
        # Clock.unschedule(self.update_accum_bars)

    def get_probs(self, dt):
        self.sm.current_cmd = 1 if self.gp.asteroidx == 350 else 2
        t, buf = self.sm.GetBuffData()
        # print(buf.shape, self.session.dp.buf_len)
        if buf.shape[0] == self.session.dp.buf_len:
            self.p = self.ap.classify_epoch(buf.T, 'prob')[0]
            # if self.session.control.inst_prob: self.update_inst_bars()
            p1 = self.p[0]
            p2 = self.p[1]
            u = p1 - p2
            if u > 0:
                pyautogui.hotkey("a", interval=0.1)
            else:
                pyautogui.hotkey("d", interval=0.1)


    def update_command(self, dt):
        if self.p is None: return
        p1 = self.p[0]
        p2 = self.p[1]
        u = p1 - p2
        if u > 0:
            pyautogui.hotkey("a", interval=0.1)
        else:
            pyautogui.hotkey("d", interval=0.1)

    def update_current_label(self, dt):
        # print(self.gp.asteroidx)
        self.sm.current_cmd = 1 if self.gp.asteroidx == 350 else 2

class GalaxyPlay(threading.Thread):

    def __init__(self, session):
        super(GalaxyPlay, self).__init__()
        self.stop_flag = False
        self._stopper = threading.Event()
        self.asteroidx = random.choice([0, 350])
        self.session = session

    def define_pygame(self):
        pygame.init()
        # Inicial
        self.default_font = pygame.font.Font('freesansbold.ttf', 32)
        self.instruc = pygame.font.Font('freesansbold.ttf', 32)
        self.instruc2 = pygame.font.Font('freesansbold.ttf', 32)
        self.ready = pygame.font.Font('freesansbold.ttf', 32)
        self.go = pygame.font.Font('freesansbold.ttf', 32)
        self.inst = pygame.image.load("galaxy_game/instruc2.png")
        self.spacet = pygame.mixer.music.load('galaxy_game/spacet.ogg')
        # pygame.mixer.music.play()
        self.startim = pygame.font.Font('freesansbold.ttf', 32)

        # nave
        self.nave = pygame.image.load("galaxy_game/nave.png")
        self.x = 346  # largura
        self.y = 480  # altura
        self.move = pygame.mixer.Sound('galaxy_game/woosh.ogg')
        self.ded = pygame.mixer.Sound('galaxy_game/ded.ogg')

        # colisãoscore
        self.colisoes = 0
        self.colidx = 300
        self.colidy = 100
        self.colidim = pygame.font.Font('freesansbold.ttf', 32)

        # Tempo
        self.timev = 0
        self.timeim = pygame.font.Font('freesansbold.ttf', 32)
        self.timex = 300
        self.timey = 300

        # score
        self.batida = 0
        self.scorev = 0
        self.scoreim = pygame.font.Font('freesansbold.ttf', 32)
        self.textx = 300
        self.texty = 200

        # asteroid
        self.asteroidim = pygame.image.load("galaxy_game/asteroid3.png")
        ##self.asteroidx = random.randint(0,5)
        ##self.asteroidy = random.randint(-150,-130)
        self.asteroidx = random.choice([0, 350])
        self.asteroidy = -330 # -130
        self.astemusic = pygame.mixer.Sound('galaxy_game/asteroid.ogg')

        # blit() - adiciona na superficie da tela

        self.sair = False
        self.play = False  # partida já iniciada?
        self.game_over = False  # fim de partida ?

        self.fps = 33  # controla o tempo de jogo também

    def run(self):
        self.define_pygame()
        self.tela = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.space = pygame.image.load("galaxy_game/space2.png")
        pygame.display.set_caption('teste')

        # pygame.display.update()
        self.pontua = False
        while not self.sair: #and not self.stop_flag:
            # print(self.stop_flag)

            for event in pygame.event.get():  # captura todos os eventos que ocorrem na tela do jogo
                # print(event)
                if event.type == pygame.QUIT:
                    self.sair = True

            # Tela inicial
            # if self.play == False and self.game_over == False:
            #     self.tela.blit(self.space, (0, 0))  # imagem de fundo
            #     self.instrucoes(self.x, self.y)
            #     self.start_button(self.x, self.y)
            #     pygame.display.update()
            #     if event.type == pygame.KEYDOWN:
            #         if event.key == pygame.K_SPACE:
            #             self.play = True

            # self.tela.blit(self.space, (0, 0))  # imagem de fundo
            pygame.display.update()

            if self.colisoes != 5: self.play = True

            if self.game_over == True and self.play == False:
                self.tela.fill((0, 0, 0))  # preenche com uma cor sólida
                self.show_score(self.textx, self.texty)
                self.timer(self.x, self.y)
                self.colid(self.x, self.y)
                self.clock.tick(self.fps)  # limita os frames por segundo
                try:
                    pygame.display.update()  # atualiza tela
                except:
                    pass
                # break

            if self.game_over == False and self.play == True:
                self.nave = pygame.image.load("galaxy_game/nave.png")
                self.timev += 1
                self.timev2 = math.trunc(self.timev / self.fps)
                # print(self.timev, self.timev2)
                self.tela.fill((0, 0, 0))
                self.tela.blit(self.space, (0, 0))
                if self.timev2 <= 1: self.ready_(self.x, self.y)
                if self.timev2 >= 3: self.asteroidy += (500/self.session.dp.buf_len)
                # nave_mechanics
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.x -= 10 # 7
                        self.move.play()

                    if event.key == pygame.K_d:
                        self.x += 10 #7
                        self.move.play()

                if self.x <= 0: self.x = 0
                elif self.x >= 670: self.x = 670 #670

                # asteroid_mechanics
                if self.asteroidy >= 400:
                    if -9 >= self.distancex or self.distancex >= 389: self.scorev += 1
                    self.colisoes += 1
                    self.asteroidx = random.choice([0, 350])
                    self.astemusic.play()
                    self.asteroidy = -330 # -130
                    self.pontua = True

                # colisão
                self.distancex = math.pow(self.x - self.asteroidx, 1)
                self.distancey = math.pow(self.y - self.asteroidy, 1)

                if self.distancey <= 330 and -9 <= self.distancex <= 389:
                    self.nave = pygame.image.load("galaxy_game/explosion.png")
                    self.x = self.asteroidx + 230

                if self.colisoes == 5: # self.timev2 == 132:
                    self.game_over = True
                    self.play = False

                    # self.tela.fill((0, 0, 0))  # preenche com uma cor sólida
                    # self.show_score(self.textx, self.texty)
                    # self.timer(self.x, self.y)
                    # self.colid(self.x, self.y)
                    # self.clock.tick(self.fps)  # limita os frames por segundo
                    # try:
                    #     pygame.display.update()  # atualiza tela
                    # except:
                    #     pass

                # Score
                # if self.distancey == -22 and -9 >= self.distancex: self.scorev += 1
                # if self.distancey == -22 and self.distancex >= 389: self.scorev += 1

                # if self.asteroidy >= 499 and -9 >= self.distancex: self.scorev += 1
                # if self.asteroidy >= 499 and self.distancex >= 389: self.scorev += 1

                # if (self.pontua and -9 >= self.distancex) or (self.pontua and self.distancex >= 389):
                #     self.scorev += 1
                #     self.pontua = False

                # print(self.asteroidy, self.distancex)

                self.colid(self.x, self.y)
                self.show_score(self.x, self.y)

                self.asteroid(self.x, self.y)
                self.player(self.x, self.y)
                self.clock.tick(33)
                pygame.display.update()

        pygame.quit()

    def Stop(self):
        print('Streaming stopped. Closing game')
        pygame.quit()
        self._stopper.set()  # alterado de _stop para _stopper

    def Stopped(self):
        print('Streaming stopped. Closing game')
        pygame.quit()
        return self._stopper.isSet()  # alterado de _stop para _stopper

    def player(self, x, y):
        self.tela.blit(self.nave, (x, y))

    def asteroid(self, x, y):
        self.tela.blit(self.asteroidim, (self.asteroidx, self.asteroidy))

    def show_score(self, x, y):
        score = self.scoreim.render("Score: " + str(self.scorev), True, (255, 255, 255))
        self.tela.blit(score, (x, y))

    def start_button(self, x, y):
        startb = self.startim.render("Aperte Espaço para Iniciar", True, (255, 255, 255,))
        self.tela.blit(startb, (350, 550))

    def instrucoes(self, x, y):
        instrucim = self.default_font.render("Desvie dos Asteroides", True, (255, 255, 255,))
        instrucim2 = self.default_font.render("para marcar pontos", True, (255, 255, 255,))
        self.tela.blit(instrucim, (50, 50))
        self.tela.blit(instrucim2, (50, 100))
        self.tela.blit(self.inst, (50, 200))

    def ready_(self, x, y):
        readyim = self.instruc.render("READY", True, (255, 255, 255,))
        self.tela.blit(readyim, (350, 250))

    def timer(self, x, y):
        time = self.timeim.render("Tempo: " + str(math.trunc(self.timev / 33)), True, (255, 255, 255))
        self.tela.blit(time, (self.timex, self.timey))

    def colid(self, x, y):
        colider = self.colidim.render("Asteroides: " + str(self.colisoes), True, (255, 255, 255))
        self.tela.blit(colider, (self.colidx, self.colidy))


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class GalaxySettings(Screen):
    def __init__(self, session, **kwargs):
        super(GalaxySettings, self).__init__(**kwargs)
        self.session = session

    def back_to_galaxy_menu(self, *args):
        self.manager.current = 'GalaxyMenu'
        self.manager.transition.direction = 'right'

    def save_config(self, *args):
        self.session.control.game_threshold = self.ids.game_threshold.value
        self.session.control.window_overlap = self.ids.window_overlap.value
        self.session.control.warning_threshold = self.ids.warning_threshold.value
        self.session.control.forward_speed = self.ids.forward_speed.value / 1000.0
        self.session.control.inst_prob = self.ids.inst_prob.value / 1000.0
        self.session.control.keyb_enable = self.ids.keyb_enable.value
        self.session.control.action_cmd1 = self.ids.action_cmd1.value
        self.session.control.action_cmd2 = self.ids.action_cmd2.value
        self.session.control.flag = True
        self.session.saveSession()

    def update_settings(self):
        self.ids.game_threshold.value = self.session.control.game_threshold
        self.ids.window_overlap.value = self.session.control.window_overlap
        self.ids.warning_threshold.value = self.session.control.warning_threshold
        self.ids.forward_speed.value = self.session.control.forward_speed * 1000.0
        self.ids.inst_prob.value = self.session.control.inst_prob * 1000.0
        self.ids.keyb_enable.value = self.session.control.keyb_enable
        self.ids.action_cmd1.value = self.session.control.action_cmd1
        self.ids.action_cmd2.value = self.session.control.action_cmd2


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# class GalaxyPlay(Screen):
#     def __init__(self, session, **kwargs):
#         super(GalaxyPlay, self).__init__(**kwargs)
#         self.session = session
#
#     def test(self):
#         pyautogui.hotkey("a", interval=0.1)
#         pyautogui.hotkey("d", interval=0.1)


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# class DroneMenu(Screen):
#     def __init__(self, session, **kwargs):
#         super(DroneMenu, self).__init__(**kwargs)
#         self.session = session
#
#         box1 = BoxLayout(size_hint_x=1, size_hint_y=0.5, padding=10, spacing=10, orientation='vertical')
#
#         self.label_msg = Label(text="ARDrone Control Menu", font_size=FONT_SIZE)
#
#         button_start = Button(text="Start", size=BUTTON_SIZE)
#         button_start.bind(on_press=self.change_to_run)
#
#         button_simulator = Button(text="Start Simulator", size=BUTTON_SIZE)
#         button_simulator.bind(on_press=self.start_simulator)
#
#         button_settings = Button(text="Settings", size=BUTTON_SIZE)
#         button_settings.bind(on_press=self.change_to_drone_settings)
#
#         button_back = Button(text="Back", size=BUTTON_SIZE)
#         button_back.bind(on_press=self.back_to_control_menu)
#
#         box1.add_widget(self.label_msg)
#
#         box1.add_widget(button_start)
#         box1.add_widget(button_simulator)
#         box1.add_widget(button_settings)
#         box1.add_widget(button_back)
#
#         self.add_widget(box1)
#
#     def change_to_run(self, *args):
#         self.manager.current = 'DroneRun'
#         self.manager.transition.direction = 'left'
#
#     def change_to_drone_settings(self, *args):
#         self.manager.current = 'DroneSettings'
#         self.manager.transition.direction = 'left'
#
#     def back_to_control_menu(self, *args):
#         self.manager.current = 'ControlMenu'
#         self.manager.transition.direction = 'right'
#
#     def start_simulator(self, *args):
#         PATH_TO_ROS = '/home/vboas/codes/tum_simulator_ws/devel/setup.bash'
#         os.system('roslaunch ardrone_tutorials keyboard_controller_simu_goal.launch &')


# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# class DroneSettings(Screen):
#     def __init__(self, session, **kwargs):
#         super(DroneSettings, self).__init__(**kwargs)
#         self.session = session
#
#     def change_to_drone(self, *args):
#         self.manager.current = 'DroneMenu'
#         self.manager.transition.direction = 'right'
#
#     def save_config(self, *args):
#         ids = self.ids
#         self.session.control.game_threshold = ids.game_threshold.value
#         self.session.control.window_overlap = ids.window_overlap.value / 1000.0
#         self.session.control.warning_threshold = ids.warning_threshold.value
#         self.session.control.forward_speed = ids.forward_speed.value / 1000.0
#         self.session.control.inst_prob = ids.inst_prob.value / 1000.0
#         self.session.control.keyb_enable = ids.keyb_enable.value
#
#         self.session.control.action_cmd1 = ids.action_cmd1.value
#         self.session.control.action_cmd2 = ids.action_cmd2.value
#
#         self.session.control.flag = True
#         self.session.saveSession()
#
#     def update_settings(self):
#         ids = self.ids
#
# DRONE_VEL = 1
# K = 1
# I = 1
# D_TO_TARGET = 10
# TARGET_POS_ARR = [[0, 0], [-20, 0], [-20, 20], [20 + D_TO_TARGET, 20]]  # simu1
# CMD_LIST = [1, 2, 2]

# TARGET_POS_ARR = [[0, 0], [20, 0], [20, 20], [-20, 20]] # simu2
# CMD_LIST = [2, 1, 1]
# TARGET_POS_ARR = [[0, 20], [-20, 20], [-20, 0]] # simu3
# CMD_LIST = [1, 1]

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
# class DroneRun(Screen):
#     inst_prob_left = NumericProperty(0)
#     accum_prob_left = NumericProperty(0)
#     accum_color_left = ListProperty([1, 0, 0, 1])
#
#     inst_prob_right = NumericProperty(0)
#     accum_prob_right = NumericProperty(0)
#     accum_color_right = ListProperty([0, 0, 1, 1])
#
#     label_on_toggle_button = StringProperty('Start')
#
#     current_label = NumericProperty(None)
#
#     label_color = ListProperty([0, 0, 0, 1])
#
#     wt = NumericProperty(0.0)
#
#     def __init__(self, session, **kwargs):
#         super(DroneRun, self).__init__(**kwargs)
#         self.session = session
#
#         self.stream_flag = False
#         self.p = None
#         self.U1 = 0.0
#         self.U2 = 0.0
#
#     # BUTTON CALLBACKS
#     def change_to_drone(self, *args):
#         self.manager.current = 'DroneMenu'
#         self.manager.transition.direction = 'right'
#
#     def toogle_stream(self, *args):
#         if self.stream_flag:
#             self.stream_stop()
#         else:
#             self.stream_start()
#
#     def start_drone(self):
#         # pass
#         # Hardware:
#         from processing.ardrone_ros import ARDrone
#         self.drone = ARDrone()
#
#     def stream_stop(self):
#         self.sm.stop_flag = True
#         self.stream_flag = False
#         self.sm.join()
#         self.label_on_toggle_button = 'Start'
#         self.clock_unscheduler()
#         self.set_bar_default()
#         # self.save_results()
#         self.drone.stop()
#         self.drone.land()
#         self.drone.reset()
#         game_time = time.time() - self.game_start_time
#         results = np.array([self.pos_history, game_time])
#         res = DroneResultsPopup(self.session, results, self.sm.all_data)
#         res.open()
#
#         # global I
#         # if self.bad_run: res.save_results('run' + str(I) + 'bad')
#         # else: res.save_results('run' + str(I))
#         # I += 1
#         # if I < 20:
#         #     time.sleep(4)
#         #     self.stream_start()
#
#     def stream_start(self):
#         self.lock_check_pos = False
#         self.drone.stop()
#         self.bad_run = False
#         self.cmd_list = iter(CMD_LIST)
#         self.target_pos_arr = iter(TARGET_POS_ARR)
#         self.update_target_area()
#
#         self.load_approach()
#
#         TTA = 10.
#         ABUF_LEN = TTA * self.session.acq.sample_rate / self.session.control.window_overlap
#         self.delta_ref = self.ap.accuracy * ABUF_LEN
#         self.U1_local = collections.deque(maxlen=ABUF_LEN)
#         self.U2_local = collections.deque(maxlen=ABUF_LEN)
#
#         self.sm = SampleManager(self.session.acq.com_port,
#                                 self.session.dp.buf_len, daisy=self.session.acq.daisy,
#                                 mode=self.session.acq.mode,
#                                 path=self.session.acq.path_to_file,
#                                 labels_path=self.session.acq.path_to_labels_file,
#                                 dummy=self.session.acq.dummy)
#
#         self.sm.daemon = True
#         self.sm.stop_flag = False
#         self.sm.start()
#         self.label_on_toggle_button = 'Stop'
#         self.stream_flag = True
#         self.pos_history = np.array([0, 0])
#         self.clock_scheduler()
#         self.drone.takeoff()
#         self.game_start_time = time.time()
#
#     def clock_scheduler(self):
#         Clock.schedule_once(self.move_drone_forward, 2)
#         Clock.schedule_interval(self.get_probs, 1. / 20.)
#         Clock.schedule_interval(self.update_accum_bars,
#                                 float(self.session.control.window_overlap) / self.session.acq.sample_rate)
#         Clock.schedule_interval(self.store_pos, .2)
#         Clock.schedule_interval(self.check_pos, 1. / 10.)
#
#         if self.session.acq.mode == 'simu' and not self.session.acq.dummy:
#             pass
#             Clock.schedule_interval(self.update_current_label, 1. / 5.)
#
#     def clock_unscheduler(self):
#         Clock.unschedule(self.get_probs)
#         Clock.unschedule(self.update_current_label)
#         Clock.unschedule(self.update_accum_bars)
#         Clock.unschedule(self.store_pos)
#         Clock.unschedule(self.check_pos)
#
#     def get_probs(self, dt):
#         t, buf = self.sm.GetBuffData()
#         if buf.shape[0] == self.session.dp.buf_len:
#             self.p = self.ap.classify_epoch(buf.T, 'prob')[0]
#             if self.session.control.inst_prob: self.update_inst_bars()
#
#     def update_inst_bars(self):
#         if self.p is None: return
#
#         p1 = self.p[0]
#         p2 = self.p[1]
#         u = p1 - p2
#
#         if u >= 0:
#             u = 1
#         else:
#             u = -1
#
#         if u > 0:
#             self.inst_prob_left = int(math.floor(u * 100))
#             self.inst_prob_right = 0
#         else:
#             self.inst_prob_right = int(math.floor(abs(u) * 100))
#             self.inst_prob_left = 0
#
#     def update_accum_bars(self, dt):
#         if self.p is None: return
#
#         p1 = self.p[0]
#         p2 = self.p[1]
#         u = p1 - p2
#         print(u)
#
#         if u >= 0:
#             u1 = 1
#             u2 = 0
#         elif u < 0:
#             u1 = 0
#             u2 = 1
#         else:
#             return
#
#         print(u1, u2)
#
#         self.U1 = self.U1 + u1
#         self.U2 = self.U2 + u2
#         self.U1_local.append(self.U1)
#         self.U2_local.append(self.U2)
#
#         delta1 = self.U1_local[-1] - self.U1_local[0]
#         delta2 = self.U2_local[-1] - self.U2_local[0]
#
#         BAR1 = 100 * (delta1 / self.delta_ref)
#         BAR2 = 100 * (delta2 / self.delta_ref)
#
#         BAR1_n = int(math.floor(BAR1))
#         BAR2_n = int(math.floor(BAR2))
#
#         if BAR1_n < 100: self.accum_prob_left = BAR1_n
#         if BAR2_n < 100: self.accum_prob_right = BAR2_n
#
#         self.map_probs(BAR1, BAR2)
#
#     def map_probs(self, U1, U2):
#         if (U1 > 100) or (U2 > 100):
#             self.drone.stop()
#             self.set_bar_default()
#             # self.sm.clear_buffer()
#             self.sm.current_cmd = 0
#             Clock.schedule_once(self.move_drone_forward, 2)
#             if U1 > 100:
#                 self.drone.set_direction('left')
#             else:
#                 self.drone.set_direction('right')
#             self.update_target_area()
#             self.lock_check_pos = False
#         elif self.sm.current_cmd == 0:
#             if U1 > U2:
#                 self.sm.winning = 1
#             else:
#                 self.sm.winning = 2
#             # dont send any cmd
#
#     def move_drone_forward(self, dt):
#         self.drone.set_forward_vel(DRONE_VEL)
#
#     def set_bar_default(self):
#         self.accum_prob_left = 0
#         self.accum_prob_right = 0
#         self.inst_prob_left = 0
#         self.inst_prob_right = 0
#         self.p = None
#         self.U1_local.clear()
#         self.U2_local.clear()
#
#     def update_current_label(self, dt):
#         self.current_label = self.sm.current_cmd
#
#     def load_approach(self):
#         self.ap = Approach()
#         self.ap.loadSetup(PATH_TO_SESSION + self.session.info.nickname)
#
#     def store_pos(self, dt):
#         new = [self.drone.pos_x, self.drone.pos_y]
#         self.pos_history = np.vstack([self.pos_history, new])
#
#     def check_pos(self, dt):
#         if self.lock_check_pos:
#             # print('locked')
#             return
#         pos = [self.drone.pos_x, self.drone.pos_y]
#         target_area = self.target_area
#         if (target_area[0] < pos[0] < target_area[2]) and (target_area[1] < pos[1] < target_area[3]):
#             print('entrou na area')
#             try:
#                 self.sm.current_cmd = next(self.cmd_list)
#                 # self.sm.clear_buffer()
#                 # self.sm.jump_playback_data()
#                 self.set_bar_default()
#                 self.lock_check_pos = True
#             except StopIteration:
#                 self.stream_stop()
#
#         else:
#             if abs(pos[0]) > 35 or abs(pos[1]) > 35:
#                 self.bad_run = True
#                 self.stream_stop()
#
#     def update_target_area(self):
#         try:
#             target_pos = next(self.target_pos_arr)
#             targ_yaw = self.drone.target_yaw
#             if targ_yaw == 270:
#                 self.target_area = [
#                     target_pos[0] - 100,
#                     target_pos[1] - D_TO_TARGET,
#                     target_pos[0] + 100,
#                     target_pos[1] + 100,
#                 ]
#             elif targ_yaw == 360 or targ_yaw == 0:
#                 self.target_area = [
#                     target_pos[0] - 100,
#                     target_pos[1] - 100,
#                     target_pos[0] + D_TO_TARGET,
#                     target_pos[1] + 100,
#                 ]
#             if targ_yaw == 90:
#                 self.target_area = [
#                     target_pos[0] - 100,
#                     target_pos[1] - 100,
#                     target_pos[0] + 100,
#                     target_pos[1] + D_TO_TARGET,
#                 ]
#             if targ_yaw == 180:
#                 self.target_area = [
#                     target_pos[0] - D_TO_TARGET,
#                     target_pos[1] - 100,
#                     target_pos[0] + 100,
#                     target_pos[1] + 100,
#                 ]
#
#         except StopIteration:
#             self.stream_stop()
#
#
# class DroneResultsPopup(Popup):
#     def __init__(self, session, results, data, **kwargs):
#         super(DroneResultsPopup, self).__init__(**kwargs)
#         self.session = session
#         self.res = results
#         self.data = data
#
#     def save_results(self, game_name):
#         path_res = (PATH_TO_SESSION + self.session.info.nickname + '/' + 'game_results_' + game_name + '.npy')
#         path_data = (PATH_TO_SESSION + self.session.info.nickname + '/' + 'game_data_' + game_name + '.npy')
#         r = np.array(self.res)
#         save_npy_data(r, path_res, mode='w')
#         save_npy_data(r, path_data, mode='w')
#

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
