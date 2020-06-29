# -*- coding: utf-8 -*-


teste = [1,2,3,4,5,6,7,8,9]

teste = iter(teste)

next(teste)

from view.session_info import UserSession
session = UserSession()
session.info.nickname = 'vitor'
session.loadSession()

print
print('eeg_path:', session.acq.eeg_path)

print('flag =', session.control.flag)

print('window_overlap =', session.control.window_overlap) 
print('game_threshold =', session.control.game_threshold)
print('warning_threshold =', session.control.warning_threshold)
print('inst_prob =', session.control.inst_prob)

print('forward_speed =', session.control.forward_speed) 
print('keyb_enable =', session.control.keyb_enable)

print('action_cmd1 =', session.control.action_cmd1) 
print('action_cmd2 =', session.control.action_cmd2)

session.info.nickname = 'vitor'
session.saveSession()

# print(session.info.nickname, session.info.fullname, session.info.is_dataset, session.info.age, 
#       session.info.gender, session.info.ds_name, session.info.ds_subject, session.acq.sample_rate,)
# print('::: Acquisition Header :::\n', f'path_to_eeg_data = {session.acq.path_to_eeg_data}\n',
#       f'path_to_eeg_events = {session.acq.path_to_eeg_events}\n', f'dummy = {session.acq.dummy}\n', f'sample_rate = {session.acq.sample_rate}\n')
# print('::: DataProcessing Header :::\n', f'buf_len = {session.dp.buf_len}\n',
#       f'epoch_start = {session.dp.epoch_start}\n', f'epoch_end = {session.dp.epoch_end}\n')



# import collections
# import math
# ABUF_LEN = math.ceil(35.76970443349754)
# U1_local = collections.deque(maxlen=ABUF_LEN)  
