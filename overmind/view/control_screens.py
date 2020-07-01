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
# import kivy.garden.bar  # from kivy.garden.bar import Bar
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
        self.U1 = 0.0 # U1 e U2 registram o acumulador histórico durante toda a operacao das classes A e B respectivamente
        self.U2 = 0.0
        self.p = [0, 0]

    def back_to_control_menu(self, *args):
        self.manager.current = 'ControlMenu'
        self.manager.transition.direction = 'right'

    def toogle_stream(self, *args):
        if self.stream_flag: self.stream_stop() # Se jogo rodando (stream_flag==True) --> para
        else: self.stream_start() # Se jogo parado (stream_flag==False) --> inicia

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
        self.limiar = self.session.control.game_threshold # 80
        TTA = 5. # tempo de ação (segundos)
        inc_rate = 0.1 # 10% srate (porcentagem do incremento)
        increment = self.session.acq.sample_rate * inc_rate  # (old increment = 25) # incremento em amostras
        ABUF_LEN = TTA * self.session.acq.sample_rate / increment # largura do buffer CIRCULAR externo A ( para Fs=250, 1250 / 25 = 50)
        cal_acc = self.ap.learner.get_results()  # old self.ap.accuracy
        self.delta_ref = cal_acc * TTA / inc_rate # cal_acc * TTA / (increment / self.session.acq.sample_rate) #  80 * 5 / 0.1 (METADE DA ACURÁCIA DE CALIBRACAO)

        self.U1_local = collections.deque(maxlen=int(ABUF_LEN)) # U1_local e U2_local registram os acumuladores deslizantes (instantaneo) das classes A e B respectivamente (usados para verificaćao de envio de comando)
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
        Clock.schedule_interval(self.get_probs, 0.05) # delta_t = 1/20 = 0.05s = 50ms (20x por segundo) # agenda a invocacao do método get_prob para a cada 50ms 0.05s

        ##### controla o tempo em que o buffer A é analisado para envio ou não do comando (qto maior o tempo self.session.control.window_overlap, menor a sobreposicao com o ultimo buffer
        Clock.schedule_interval(self.update_accum_bars, 0.5) # , self.session.control.window_overlap) # /1000 ? # executado após cada execucao de get_probs
        if self.session.acq.mode == 'simu': # and not self.session.acq.dummy and not self.session.acq.path_to_labels_file == '':
            Clock.schedule_interval(self.update_current_label, 0.05)

    def clock_unscheduler(self):
        Clock.unschedule(self.get_probs) # cancela o agendamento do método get_prob
        Clock.unschedule(self.update_current_label)
        Clock.unschedule(self.update_accum_bars)

    def get_probs(self, dt):
        t, buf = self.sm.GetBuffData() # tira a foto dos buffers atuais
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
        p1 = self.p[0] # captura prob classe A
        p2 = self.p[1] # captura prob classe B
        u = p1 - p2 # u > 0 se p1>p2 ; u < 0 se p1<p2 ; u == 0 se p1==p2
        if u >= 0: # se prob classe A maior ou igual que prob classe B, u >= 0
            u1 = 1
            u2 = 0
        else: # se prob classe B maior, u < 0
            u1 = 0
            u2 = 1

        self.U1 = self.U1 + u1
        self.U2 = self.U2 + u2
        self.U1_local.append(self.U1)
        self.U2_local.append(self.U2)

        # print(u1, u2, self.U1, self.U2, self.U1_local, self.U2_local)

        delta1 = self.U1_local[-1] - self.U1_local[0] # calcula a variacao dentro do buffer_local_local U1 da primeira até a ultima avaliacao
        delta2 = self.U2_local[-1] - self.U2_local[0] # calcula a variacao dentro do buffer_local U2_local da primeira até a ultima avaliacao

        # delta1 + delta2 serao igual a delta_ref, de modo que eles expressam a soma das probabilidades

        # controla o aumento da barra esquerda conforme a variacao delta1 para classe A (U1_local) e a variacao de referencia data pela metade da acurácia de calibracao
        BAR1 = 100 * (delta1 / self.delta_ref) #
        BAR2 = 100 * (delta2 / self.delta_ref) #

        BAR1_n = int(math.floor(BAR1))
        BAR2_n = int(math.floor(BAR2))

        # print(BAR1_n, BAR2_n, delta1, delta2)

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
            pass # dont send any cmd

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
    res = ListProperty([0])
    hits = NumericProperty(0)

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

    def espera(self, dt):
        print('...')
        Clock.schedule_interval(self.get_probs, 1./2)
        
    def clock_scheduler(self):
        Clock.schedule_once(self.espera, 5)
        # Clock.schedule_interval(self.galaxy_screen2, 30)
        #Clock.schedule_interval(self.get_probs, 1./3)
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
            # print(u)
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
        self.inst = pygame.image.load(os.path.dirname(__file__) + "/../game/galaxy/instruc2.png")
        self.spacet = pygame.mixer.music.load(os.path.dirname(__file__) + '/../game/galaxy/spacet.ogg')
        # pygame.mixer.music.play()
        self.startim = pygame.font.Font('freesansbold.ttf', 32)

        # nave
        self.nave = pygame.image.load(os.path.dirname(__file__) + "/../game/galaxy/nave.png")
        self.x = 346  # largura
        self.y = 480  # altura
        self.move = pygame.mixer.Sound(os.path.dirname(__file__) + '/../game/galaxy/woosh.ogg')
        self.ded = pygame.mixer.Sound(os.path.dirname(__file__) + '/../game/galaxy/ded.ogg')

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
        self.asteroidim = pygame.image.load(os.path.dirname(__file__) + "/../game/galaxy/asteroid3.png")
        ##self.asteroidx = random.randint(0,5)
        ##self.asteroidy = random.randint(-150,-130)
        self.asteroidx = random.choice([0, 350])
        self.asteroidy = -330 # -130
        self.astemusic = pygame.mixer.Sound(os.path.dirname(__file__) + '/../game/galaxy/asteroid.ogg')

        # blit() - adiciona na superficie da tela

        self.sair = False
        self.play = False  # partida já iniciada?
        self.game_over = False  # fim de partida ?

        self.fps = 40  # também controla o tempo de jogo

    def run(self):
        self.define_pygame()
        self.tela = pygame.display.set_mode((800, 600))
        self.clock = pygame.time.Clock()
        self.space = pygame.image.load(os.path.dirname(__file__) + "/../game/galaxy/space2.png")
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

            if self.colisoes != 4: self.play = True

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
                self.nave = pygame.image.load(os.path.dirname(__file__) + "/../game/galaxy/nave.png")
                self.timev += 1
                self.timev2 = math.trunc(self.timev / self.fps)
                # print(self.timev, self.timev2)
                self.tela.fill((0, 0, 0))
                self.tela.blit(self.space, (0, 0))
                if self.timev2 <= 1: self.ready_(self.x, self.y)
                if self.timev2 >= 3: self.asteroidy += (600/self.session.dp.buf_len)
                # nave_mechanics
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_a:
                        self.x -= 10 # 7
                        #self.move.play()

                    if event.key == pygame.K_d:
                        self.x += 10 #7
                        #self.move.play()

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
                    self.nave = pygame.image.load(os.path.dirname(__file__) + "/../game/galaxy/explosion.png")
                    self.x = self.asteroidx + 230

                if self.colisoes == 4: # self.timev2 == 132:
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
