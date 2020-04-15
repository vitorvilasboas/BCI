from processing.processor import Approach
from processing.utils import PATH_TO_SESSION

class Controller:
    def __init__(self, smanager, session):
        self.session = session
        self.ap = self.load_approach()
        self.sm = smanager
        self.clear_states()

    def clear_states(self):
        self.U = 0

    def get_probs(self):
        t, buf = self.sm.GetBuffData()
        # if buf.shape[0] == self.session.dp.buf_len:
        p = self.ap.classify_epoch(buf.T, 'prob')[0]
        return p

    def update_cumulative(self, p):
        if p is None: return
        p1 = p[0]
        p2 = p[1]
        u = p1 - p2
        self.U += u
        return self.U

    def generate_cmd(self):
        p = self.get_probs()
        U = self.update_cumulative(p)
        U1 = 100 * (U + self.session.control.game_threshold) / (2. * self.session.control.game_threshold)
        U2 = 100 - U1
        if U1 > 100: cmd = 1
        elif U2 > 100: cmd = 2
        else: cmd = 0
        return cmd, [U1, U2], p

    def load_approach(self):
        ap = Approach()
        ap.loadSetup(PATH_TO_SESSION + self.session.info.nickname)
        return ap