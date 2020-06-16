import numpy as np
import threading
import queue
from scipy import optimize, integrate, interpolate
import matplotlib.pyplot as plt
import pickle

from Rx.Demodulator import Demodulator


class TxCalDemodulator(Demodulator):
    def __init__(self, *args, **kwargs):
        super(TxCalDemodulator, self).__init__(*args, **kwargs)

    def torch_ode(self, T, t, We, ke, Ce):
        b = 1.311e-13
        dTdt = (We(t) - ke * (T - self.Te) - b * (T ** 4)) / Ce
        return dTdt

    def bit_predict(self, t, T0, v0, dvdt0, We, ke, Ce, dt=1e-4):
        We_func = lambda x: We

        assert t[0] == 0.0
        t_grid = np.linspace(0, t[-1], int((t[-1] - t[0]) / dt) + 1)

        T_ode = integrate.odeint(self.torch_ode, T0, t_grid, args=(We_func, ke, Ce))
        T = T_ode[:, 0]
        T_end = T[-1]

        nor_rx_power = np.zeros(t_grid.shape, dtype=np.float64)
        for i in range(nor_rx_power.shape[0]):
            nor_rx_power[i] = self.normalized_rx_power_func(T[i])

        dnor_rx_power_dt = interpolate.interp1d(t_grid[:-1], np.diff(nor_rx_power) / dt,
                                                kind='quadratic', bounds_error=False, fill_value='extrapolate')

        v_ode = integrate.odeint(self.pyro_ode, [v0, dvdt0], t, args=(dnor_rx_power_dt,))
        v = v_ode[:, 0]

        v_end = v[-1]
        dvdt_end = v_ode[-1, 1]

        return v, T_end, v_end, dvdt_end, (t_grid, T, nor_rx_power)

    def header_predict(self, We, ke, Ce):
        n = len(self.frame_header)
        spb = round(self.fs / self.bit_rate)
        t_header = np.linspace(start=0, stop=1.0 / self.bit_rate * n, num=spb * n + 1)
        v_header = np.zeros(shape=(spb * n + 1,))
        t = np.linspace(start=0, stop=1.0 / self.bit_rate, num=spb + 1)

        T0, v0, dvdt0 = self.Te, 0.0, 0.0
        for i in range(n):
            s_idx = i * spb
            e_idx = (i + 1) * spb + 1

            v_bit, T0, v0, dvdt0, info = self.bit_predict(t, T0, v0, dvdt0, self.frame_header[i] * We, ke, Ce)
            v_header[s_idx:e_idx] = v_bit
        return v_header, t_header

    def tx_cal(self, npop, alpha, x_init, lb, ub, max_iter=100, sigma=20):
        gt = np.genfromtxt('./result/sync/sync.csv', delimiter=',')

        x = x_init
        sigma = (ub - lb) / sigma
        for i in range(max_iter):
            N = np.random.rand(npop, 1)
            R = np.zeros(npop)
            for j in range(npop):
                x_try = x + sigma * N[j]
                pred, _ = self.header_predict(We=0.8978, ke=1.033e-3, Ce=x_try)
                R[j] = np.mean(np.abs(pred - gt))
            A = (R - np.mean(R)) / np.std(R)
            x = x - alpha * (npop * sigma) * np.dot(N.T, A)
            print(x)

        plt.figure()
        plt.plot(pred)
        plt.plot(gt)
        plt.savefig('./result/tx_cal/tx_cal.png')
        plt.close()


if __name__ == '__main__':
    cfg = {
        'fs': 1e3,
        'channel_id': 'single',
        'channel_range': 1000,
        'bit_rate': 50,
        'frame_header': (1, 1, 1, 0),
        'frame_bits': 8,
    }

    test = TxCalDemodulator(
        header_queue=queue.Queue(maxsize=0),
        sample_freq=cfg['fs'],
        bit_rate=cfg['bit_rate'],
        frame_header=cfg['frame_header'],
        frame_bits=cfg['frame_bits'],
        channel_id=cfg['channel_id'],
        channel_range=cfg['channel_range']
    )

    test.tx_cal(npop=30,
                alpha=0.005,
                x_init=np.array([1.9e-5]),
                lb=np.array([1.0e-5]),
                ub=np.array([3.0e-5]),
                max_iter=10)
