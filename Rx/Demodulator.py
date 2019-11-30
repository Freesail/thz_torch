import numpy as np
import threading
import queue
from scipy import optimize, integrate, interpolate
import matplotlib.pyplot as plt


class Demodulator:
    def __init__(self, sample_freq=1e3):
        self.src_queue = queue.Queue(maxsize=0)
        self.fs = sample_freq

        try:
            self.pyro_params = np.genfromtxt('./result/cal/pyro_params.csv', delimiter=',')
        except OSError:
            self.pyro_params = None

        self.thread = threading.Thread(target=self.demodulate)

    def start(self):
        self.thread.start()

    def demodulate(self):
        while True:
            mode, frame = self.src_queue.get(block=True, timeout=None)
            if mode == 'data':
                self.data_demodulate(frame)
            elif mode == 'cal':
                self.cal_demodulate(frame)

    def cal_demodulate(self, frame):
        t = np.arange(0, len(frame)) / self.fs
        v_step = frame

        popt, pcov = optimize.curve_fit(self.pyro_v_step, t, v_step, p0=[1.0, 50.0, 5.0, 0.0])

        v_step_fit = self.pyro_v_step(t, *popt)
        plt.figure()
        plt.plot(t, v_step, t, v_step_fit)
        plt.savefig('./result/cal/v_step.png')
        np.savetxt('./result/cal/v_step.csv', np.stack((t, v_step, v_step_fit), axis=-1), delimiter=',')
        np.savetxt('./result/cal/vstep_params.csv', popt, delimiter=',')

        self.pyro_params = np.array([popt[1] + popt[2], popt[1] * popt[2], popt[0] * (popt[1] - popt[2])])
        np.savetxt('./result/cal/pyro_params.csv', self.pyro_params, delimiter=',')
        print('pyro calibration done.')

    def data_demodulate(self, frame):
        pass

    @staticmethod
    def pyro_v_step(x, p0, p1, p2, p3):
        return p0 * (np.exp(-p1 * x) - np.exp(-p2 * x)) + p3

    # TODO: correct typo in paper
    def pyro_ode(self, x, t, dpdt):
        dx0dt = x[1]
        dx1dt = - (self.pyro_params[0] * x[1] + self.pyro_params[1] * x[0] + self.pyro_params[2] * dpdt(t))
        return [dx0dt, dx1dt]
