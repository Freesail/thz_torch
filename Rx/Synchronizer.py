import numpy as np
import queue
import threading
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import linregress
import ruptures as rpt

DEBUG = False


class Synchronizer:
    def __init__(self, sample_freq=1e3, mode='data',
                 cal_syn_t=5e-3, cal_frame_t=5e-1, cal_reset_t=8e-1,
                 frame_header=(1, 1, 1, 0), bit_rate=50, frame_bits=8, data_reset_t=8e-1,
                 syn_threshold=0.15):
        self.src_queue = queue.Queue(maxsize=0)
        self.mode_queue = queue.Queue(maxsize=0)
        self.header_queue = queue.Queue(maxsize=0)
        self.dst_queue = None

        self.mode = mode
        self.fs = sample_freq

        self.cal_syn_t = cal_syn_t
        self.cal_syn_horizon = int(cal_syn_t * self.fs)
        self.cal_frame_t = cal_frame_t
        self.cal_frame_horizon = int(cal_frame_t * self.fs)
        self.cal_reset_t = cal_reset_t
        self.cal_reset_horizon = int(cal_reset_t * self.fs)

        self.data_syn_t = len(frame_header) / bit_rate
        self.data_syn_horizon = int(self.data_syn_t * self.fs) + 1
        self.data_frame_t = (len(frame_header) + frame_bits) / bit_rate
        self.data_frame_horizon = int(self.data_frame_t * self.fs) + 1
        self.data_reset_t = data_reset_t
        self.data_reset_horizon = int(data_reset_t * self.fs)

        self.syn_queue_len = max(self.cal_syn_horizon, self.data_syn_horizon)
        self.syn_queue = deque([0.0] * self.syn_queue_len, maxlen=self.syn_queue_len)
        self.syn_threshold = syn_threshold
        self.refill = True

        self.v_header = None
        self.thread = threading.Thread(target=self._synchronizer)

    def start(self):
        print('Synchronizer: starts with %s mode' % self.mode)
        self.thread.start()

    def _synchronizer(self):
        while True:
            self.switch_mode()
            if self.mode == 'cal':
                self.cal_synchronizer()
            elif self.mode in ['data', 'tx_cal', 'record']:
                self.frame_synchronizer()
            else:
                assert False

    def switch_mode(self):
        try:
            mode = self.mode_queue.get(block=False)
            if mode != self.mode:
                self.mode = mode
                print('Synchronizer: switch mode to - %s' % self.mode)
        except queue.Empty:
            pass

    def get_v_header(self):
        try:
            self.v_header = self.header_queue.get(block=False)
            v_shift = np.zeros_like(self.v_header)
            v_shift[1:] = self.v_header[0:-1]
            self.syn_threshold = np.mean(np.abs(self.v_header - v_shift))
            print('new syn threshold: %.3f' % self.syn_threshold)
            print('Synchronizer: frame header updated')
        except queue.Empty:
            pass

    def refill_syn_queue(self):
        for i in range(self.syn_queue_len):
            self.syn_queue.append(self.src_queue.get(block=True, timeout=None))
        self.refill = False

    def cal_synchronizer(self):
        if self.refill:
            self.refill_syn_queue()

        cal_syn = list(self.syn_queue)[-self.cal_syn_horizon:]
        cal_diff = np.diff(cal_syn) * self.fs

        if np.all(cal_diff < -5):  # -20 before
            cal_frame = cal_syn
            for i in range(self.cal_frame_horizon - self.cal_syn_horizon):
                cal_frame.append(self.src_queue.get(block=True, timeout=None))
            self.dst_queue.put(('cal', cal_frame), block=True, timeout=None)

            for i in range(self.cal_reset_horizon):
                self.src_queue.get(block=True, timeout=None)
            self.refill = True
        else:
            self.syn_queue.append(self.src_queue.get(block=True, timeout=None))

    def align_by_cpd(self, s1, s2):
        t = np.arange(s1.shape[0])
        ones = np.ones_like(t)

        s1 = np.column_stack((s1.reshape(-1, 1), t, ones))
        algo1 = rpt.Dynp(model='linear').fit(s1)
        bkp1 = algo1.predict(n_bkps=3)
        print(bkp1)

        s2 = np.column_stack((s2.reshape(-1, 1), t, ones))
        algo2 = rpt.Dynp(model='linear').fit(s2)
        bkp2 = algo2.predict(n_bkps=3)
        print(bkp2)

        shift = np.array(bkp1) - np.array(bkp2)
        shift = shift[shift >= 0]
        return int(np.argmax(np.bincount(shift))), bkp1, bkp2

        # return np.argmin(s1) - np.argmin(s2)

    def frame_synchronizer(self):
        self.get_v_header()
        if self.v_header is None:
            print('Synchronizer: need frame header for data_txcal_record sync')
        else:
            if self.refill:
                self.refill_syn_queue()

            data_syn = list(self.syn_queue)[-self.data_syn_horizon:]
            e = np.mean(np.abs(self.v_header - np.array(data_syn)))

            if self.mode == 'tx_cal':
                syn_threshold = self.syn_threshold * 5.0
            else:
                syn_threshold = self.syn_threshold * 2.0

            # print(e)
            if e < syn_threshold:
                # CPD for syn
                if self.mode == 'tx_cal':
                    pre_cpd = np.array(data_syn)

                    shift, bkp_m, bkp_p = self.align_by_cpd(
                        s1=pre_cpd,
                        s2=self.v_header
                    )
                    for _ in range(shift):
                        data_syn.append(self.src_queue.get(block=True, timeout=None))
                        data_syn.pop(0)
                    print('shift timesteps: %d' % shift)

                    post_cpd = np.array(data_syn)
                    plot_data = {
                        'pre_cpd': pre_cpd,
                        'post_cpd': post_cpd,
                        'p': self.v_header,
                        'bkp_m': bkp_m,
                        'bkp_p': bkp_p
                    }
                    with open('./result/sync/cpd_plot.pkl', 'wb') as f:
                        import pickle
                        pickle.dump(plot_data, f)

                plt.figure()
                plt.plot(self.v_header)
                plt.plot(data_syn)
                np.savetxt('./result/sync/sync.csv', data_syn, delimiter=',')
                plt.savefig('./result/sync/sync.png')
                plt.close()

                data_frame = data_syn
                if self.mode == 'tx_cal':
                    self.dst_queue.put((self.mode, np.array(data_syn)), block=True, timeout=None)

                for i in range(self.data_frame_horizon - self.data_syn_horizon):
                    data_frame.append(self.src_queue.get(block=True, timeout=None))

                if self.mode != 'tx_cal':
                    self.dst_queue.put((self.mode, data_frame), block=True, timeout=None)

                for i in range(self.data_reset_horizon):
                    self.src_queue.get(block=True, timeout=None)
                self.refill = True
            else:
                self.syn_queue.append(self.src_queue.get(block=True, timeout=None))
