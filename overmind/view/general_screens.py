import os
import numpy as np
from kivy.app import App
from kivy.uix.screenmanager import Screen
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.popup import Popup
from kivy.core.window import Window
from kivy.properties import ObjectProperty, StringProperty
from processing.utils import PATH_TO_SESSION

# @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
class StartScreen(Screen):  # layout - A classe implementa Screen o gerenciador de tela kivy (screenmanager)
    login = ObjectProperty(None)  # var python para armazenar objeto kivy
    label_msg = StringProperty('')  # var python para armazenar valor string oriundo do kivy
    session_text = StringProperty('')

    def __init__(self, session, **kwargs):  # session_header recebe a sessão_do_usuário ativo
        super(StartScreen, self).__init__(**kwargs)  # atribui a lista de argumentos (Keywordsargs) à superclasse kivy Screen implementada
        self.session = session  # contém a sessão do usuário e seus atributos e métodos

    def change_to_register(self, *args):
        self.manager.current = 'Register'
        self.manager.transition.direction = 'left'

    def change_to_bci(self, *args):
        self.manager.current = 'BCIMenu'
        self.manager.transition.direction = 'left'

    def on_pre_enter(self, *args):
        # print('User atual ', self.session.info.nickname, ' logado? >>', self.session.info.flag)
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

    def update_screen(self): # chamada quando o usuário volta para a tela principal
        self.session.info.flag = False
        self.session.saveSession()
        print(self.session.info.nickname, 'saiu ({})'.format(self.session.info.flag))
        self.label_msg = ''
        self.session_text = ''

    def check_login(self, *args):
        sname = self.login.text  # self.ids.usuario.text

        if not os.path.isdir(PATH_TO_SESSION): os.makedirs(PATH_TO_SESSION)

        if sname == '':  # if no login is provided, use latest modified folder in userdata/session
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
                print(PATH_TO_SESSION + sname)
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

    def enable_dataset(self, *args): # chamada ao acionar o interruptor belongs_dataset
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

    def update_subject_values(self, *args): # chamada ao selecionar um dataset no field_dataset
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

    def update_exist_user(self, *args):
        self.session.saveSession()
        box = BoxLayout(orientation='vertical', padding=10, spacing=10)
        pop = Popup(title='Warning', content=box, size_hint=(None, None), size=(300, 180))
        label = Label(text='Info about  ' + self.nick + '  already updated.')
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
        btnYes = Button(text='Yes', on_press=self.update_exist_user, on_release=pop.dismiss)
        btnNo = Button(text='No', on_release=pop.dismiss)
        botoes.add_widget(btnYes)
        botoes.add_widget(btnNo)
        box.add_widget(botoes)
        pop.open()

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
                        if os.path.isdir(PATH_TO_SESSION + self.nick): self.popup_update() # caso usuário ja existe, confirma atualização de dados
                        else: self.save_user()  # caso ainda não existir usuário com o nick, salva dados do novo usuário (tipo dataset)

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
                    if os.path.isdir(PATH_TO_SESSION + self.nick): self.popup_update() # caso usuário ja existe, confirma atualização de dados
                    else: self.save_user()  # caso ainda não existir usuário com o nick, salva dados do novo usuário
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
        pop = Popup(title='Do you really want to exit ' + self.session.info.fullname + '?', content=box, size_hint=(None, None), size=(300, 180))
        btnYes = Button(text='Yes', on_press=self.close_and_back, on_release=pop.dismiss)
        btnNo = Button(text='No', on_release=pop.dismiss)
        botoes.add_widget(btnYes)
        botoes.add_widget(btnNo)
        box.add_widget(botoes)
        pop.open()

    def update_screen(self, *args):
        # print(self.session.info.nickname, self.session.info.flag)
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

