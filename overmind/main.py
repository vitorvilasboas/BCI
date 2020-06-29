import re
import os
import view.template
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager
from view.session_info import UserSession
import view.general_screens as gscr
import view.acq_screens as ascr
import view.calib_screens as kscr
import view.control_screens as cscr

def load_all_kv_files(start="view/kv/"): #Load all .kv files
    pattern = re.compile(r".*?\.kv")
    kv_files = []
    for root, dirs, files in os.walk(start): # add .kv files of view/kv in kv_files vector
        kv_files += [root + "/" + file_ for file_ in files if pattern.match(file_)]
    for file_ in kv_files:
        Builder.load_file(file_) # load .kv files added

class OverMind(App):
    def build(self):
        user_session = UserSession()
        # CREATE SCREENS
        start_screen = gscr.StartScreen(user_session, name='Start')
        register = gscr.Register(user_session, name='Register')
        bci_menu = gscr.BCIMenu(user_session, name='BCIMenu')
        acq_mode = ascr.AcqMode(user_session, name='AcqMode')
        acq_protocol = ascr.AcqProtocol(user_session, name='AcqProtocol')
        acq_run = ascr.AcqRun(user_session, name='AcqRun')
        cal_load = kscr.CalLoad(user_session, name='CalLoad')
        cal_settings = kscr.CalSettings(user_session, name='CalSettings')
        control_menu = cscr.ControlMenu(user_session, name='ControlMenu')
        control_settings = cscr.ControlSettings(user_session, name='ControlSettings')
        bars_run = cscr.BarsRun(user_session, name='BarsRun')
        target_run = cscr.TargetRun(user_session, name='TargetRun')
        galaxy_menu = cscr.GalaxyMenu(user_session, name='GalaxyMenu')
        galaxy_settings = cscr.GalaxySettings(user_session, name='GalaxySettings')
        # galaxy_play = cscr.GalaxyPlay(user_session, name='GalaxyPlay')
        # drone_run = cscr.DroneRun(user_session, name='DroneRun')
        # drone_menu = cscr.DroneMenu(user_session, name='DroneMenu')
        # drone_settings = cscr.DroneSettings(user_session, name='DroneSettings')

        # ADD SCREENS TO SCREEN MANAGER
        sm = ScreenManager() # instance a new layout manager (gerenciador de layout)

        sm.add_widget(start_screen)
        sm.add_widget(register)
        sm.add_widget(bci_menu)
        sm.add_widget(acq_mode)
        sm.add_widget(acq_protocol)
        sm.add_widget(acq_run)
        sm.add_widget(cal_settings)
        sm.add_widget(cal_load)
        sm.add_widget(control_menu)
        sm.add_widget(control_settings)
        sm.add_widget(bars_run)
        sm.add_widget(target_run)
        sm.add_widget(galaxy_menu)
        sm.add_widget(galaxy_settings)
        # sm.add_widget(galaxy_play)
        # sm.add_widget(drone_run)
        # sm.add_widget(drone_menu)
        # sm.add_widget(drone_settings)

        sm.current = 'Start' # define the first current screen
        return sm

if __name__ == "__main__":
    # try:
    #     load_all_kv_files()
    #     OverMind().run()
    # except Exception as e: print(e)

    load_all_kv_files()
    OverMind().run()
