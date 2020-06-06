import pickle # manipular estruturas de dados em formato binário
from processing.utils import PATH_TO_SESSION

class UserSession:
    def __init__(self):
        self.info = InfoHeader()            # GENERAL SETTINGS
        self.acq = AcquisitionHeader()      # ACQUISITION SETTINGS
        self.dp = DataProcessingHeader()    # DATA PROCESSING SETTINGS
        self.precal = PreCalHeader()        # PRE-CALIBRATION SETTINGS
        self.control = ControlHeader()      # CONTROL SETTINGS

    def saveSession(self):
        path = PATH_TO_SESSION + self.info.nickname + '/' + 'session_info.pkl' # escrita binária wb (write byte)
        with open(path, 'wb') as file_name: pickle.dump(self.__dict__, file_name) # salva o dicionário da sessão em arquivo binário .pickle

    def loadSession(self):
        #print(self.info.nickname)
        path = PATH_TO_SESSION + self.info.nickname + '/' + 'session_info.pkl' # leitura binária 'rb' (read byte)
        with open(path, 'rb') as file_name: load_obj = pickle.load(file_name) # carrega em load_obj os dados binários
        self.__dict__.update(load_obj) # atualiza todos os atributos da sessão carregada com os dados em load_obj

class InfoHeader:
    def __init__(self):
        self.flag = False # controla usuário logado ou não
        self.is_dataset = False # diz se o usuário cadastrado pertence a um dataset publico ou nao
        self.nickname = None
        self.fullname = None
        self.save_date = None
        self.age = None
        self.gender = None
        self.pnd = False
        self.ds_name = None # usado somente para usuários de data sets públicos (is_dataset == True)
        self.ds_subject = None # usado somente para usuários de data sets públicos (is_dataset == True)

class AcquisitionHeader:
    def __init__(self):
        self.flag_mode = False # True ao salvar AcqMode
        self.flag_protocol = False
        self.mode = None
        self.board = 'OpenBCI Cyton'
        self.com_port = None
        self.daisy = None
        self.ch_labels = None
        self.sample_rate = None
        self.eeg_path = None  # simulator
        self.class_ids = None # simulator
        self.dummy = None  # simulator
        self.n_runs = None
        self.n_trials = None
        self.cue_offset = None
        self.cue_time = None
        self.min_pause = None
        self.trial_duration = None
        self.runs_interval = None

class DataProcessingHeader:
    def __init__(self):
        self.flag_load = False  # True ao salvar CalLoad
        self.flag_setup = False

        self.eeg_path = None
        self.eeg_info = None
        self.class_ids = None
        self.cross_val = False
        self.n_folds = None
        self.test_perc = None

        self.auto_cal = False
        self.n_iter = None
        self.f_low = None
        self.f_high = None
        self.epoch_start = None
        self.epoch_end = None
        self.buf_len = None
        self.csp_nei = None
        self.sb_clf = None
        self.overlap = True

        self.channels = None
        self.final_clf = None
        self.filt_approach = None
        self.f_order = None  # if filt_approach == 'IIR':
        self.sb_method = False
        self.n_sbands = None # if sb_method:

        # self.max_amp = None
        # self.max_mse = None
        # self.ch_labels = None
        # self.sample_rate = None
        # self.new_path = False
        # self.train_events_path = None
        # self.test_data_path = None
        # self.test_events_path = None
        # self.max_channels = None
        # self.trials_per_class = None
        # self.trial_tcue = None
        # self.trial_tpause = None
        # self.trial_mi_time = None

class ControlHeader:
    def __init__(self):
        self.flag = False
        self.game_threshold = None
        self.window_overlap = None
        self.warning_threshold = None
        self.forward_speed = None
        self.inst_prob = None
        self.keyb_enable = None
        self.action_cmd1 = None
        self.action_cmd2 = None
        
class PreCalHeader:
    def __init__(self):
        self.flag = False
        self.ch_energy_left = None
        self.ch_energy_right = None
        self.total_time = None
        self.relax_time = None
        self.sign_direction = None
        self.plot_flag = None
