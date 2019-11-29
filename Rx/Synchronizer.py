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
        self.dst_queue = dst_queue
        self.mode = mode
        self.fs = sample_freq

        self.cal_syn_t = cal_syn_t
        self.cal_syn_horizon = int(cal_syn_t * self.fs)
        self.cal_frame_t = cal_frame_t
        self.cal_frame_horizon = int(cal_frame_t * self.fs)
        self.cal_reset_t = cal_reset_t
        self.cal_reset_horizon = int(cal_reset_t * self.fs)


    def _synchronizer(self):
        while True:
            if self.mode == 'cal':
                self.cal_synchronizer()
            elif self.mode == 'data':
                self.data_synchronizer()

    def cal_synchronizer(self):
        cal_queue = deque([0.0] * self.cal_syn_horizon, maxlen=self.cal_syn_horizon)
        for i in range(self.cal_syn_horizon):
            cal_queue.append(self.src_queue.get(block=True, timeout=None))
        while True:
            # TODO: detect cal frame
            if True:
                cal_frame = list(cal_queue)
                for i in range(self.cal_frame_horion - self.cal_syn_horizon):
                    cal_frame.append(self.src_queue.get(block=True, timeout=None))
                self.dst_queue.put(('cal', cal_frame), block=True, timeout=None)

                for i in range(self.cal_reset_horizon):
                    self.src_queue.get(block=True, timeout=None)
                break
            else:
                cal_queue.append(self.src_queue.get(block=True, timeout=None))

    def data_synchronizer(self):
        pass
