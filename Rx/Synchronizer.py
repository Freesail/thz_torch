import numpy as np
import queue
import threading
from collections import deque

DEBUG = False


class Synchronizer:
    def __init__(self, dst_queue, sample_freq=1e3, mode='cal',
                 cal_syn_t=5e-3, cal_frame_t=5e-1, cal_reset_t=8e-1,
                 ):
        self.src_queue = queue.Queue(maxsize=0)
        self.mode_queue = queue.Queue(maxsize=0)
        self.dst_queue = dst_queue
        self.mode = mode
        self.fs = sample_freq

        self.cal_syn_t = cal_syn_t
        self.cal_syn_horizon = int(cal_syn_t * self.fs)
        self.cal_frame_t = cal_frame_t
        self.cal_frame_horizon = int(cal_frame_t * self.fs)
        self.cal_reset_t = cal_reset_t
        self.cal_reset_horizon = int(cal_reset_t * self.fs)
        self.cal_queue = deque([0.0] * self.cal_syn_horizon, maxlen=self.cal_syn_horizon)
        self.cal_refill = True

        self.thread = threading.Thread(target=self._synchronizer)

    def start(self):
        print('Synchronizer starts with %s mode' % self.mode)
        self.thread.start()

    def _synchronizer(self):
        while True:
            self.switch_mode()
            if self.mode == 'cal':
                self.cal_synchronizer()
            elif self.mode == 'data':
                self.data_synchronizer()
            else:
                assert False

    def switch_mode(self):
        try:
            mode = self.mode_queue.get(block=False)
            print('Synchronizer mode: %s' % self.mode)
        except queue.Empty:
            pass

    def cal_synchronizer(self):
        if self.cal_refill:
            for i in range(self.cal_syn_horizon):
                self.cal_queue.append(self.src_queue.get(block=True, timeout=None))
            self.cal_refill = False

        cal_diff = np.diff(list(self.cal_queue)) * self.fs

        if np.all(cal_diff < -20):
            cal_frame = list(self.cal_queue)
            for i in range(self.cal_frame_horizon - self.cal_syn_horizon):
                cal_frame.append(self.src_queue.get(block=True, timeout=None))
            self.dst_queue.put(('cal', cal_frame), block=True, timeout=None)

            for i in range(self.cal_reset_horizon):
                self.src_queue.get(block=True, timeout=None)
            self.cal_refill = True
        else:
            self.cal_queue.append(self.src_queue.get(block=True, timeout=None))

    def data_synchronizer(self):
        pass

