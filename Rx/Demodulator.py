import numpy as np
import threading
import queue
from scipy import optimize, integrate, interpolate
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
import pickle


class Demodulator:
    def __init__(self, header_queue, sample_freq=1e3,
                 channel_id='single', channel_range=2000, Te=293.15,
                 frame_header=(1, 1, 1, 0), bit_rate=50, frame_bits=8):
        self.src_queue = queue.Queue(maxsize=0)
        self.header_queue = header_queue

        self.fs = sample_freq
        self.channel_id = channel_id
        self.channel_range = channel_range
        self.channel_info = self.get_channel_info()
        self.Te = Te
        self.frame_header = frame_header
        self.bit_rate = bit_rate
        self.frame_bits = frame_bits

        try:
            self.pyro_params = np.genfromtxt('./result/cal/pyro_params.csv', delimiter=',')
        except OSError:
            self.pyro_params = None

        try:
            self.tx_params = np.genfromtxt('./result/tx_cal/tx_params.csv', delimiter=',')
        except OSError:
            # We, ke, Ce
            self.tx_params = np.array([0.8978, 1.033e-3, 1.9e-5])

        try:
            with open('./result/rx_func/%s_%s.pkl' % (channel_id, channel_range), 'rb') as f:
                self.normalized_rx_power_func = pickle.load(f)
        except FileNotFoundError:
            self.normalized_rx_power_func = self.get_normalized_rx_power_func()
            with open('./result/rx_func/%s_%s.pkl' % (channel_id, channel_range), 'wb') as f:
                pickle.dump(self.normalized_rx_power_func, f)

        try:
            header_queue.put(np.genfromtxt('./result/header/header_pred.csv', delimiter=','))
        except OSError:
            pass

        self.thread = threading.Thread(target=self.demodulate)

    def start(self):
        print('Demodulator: starts')
        self.thread.start()

    def demodulate(self):
        while True:
            mode, frame = self.src_queue.get(block=True, timeout=None)
            if mode == 'data':
                self.data_demodulate(frame)
            elif mode == 'cal':
                self.cal_demodulate(frame)
            elif mode == 'tx_cal':
                self.tx_cal_demodulate(frame)

    def tx_cal_demodulate(self, frame):
        print('Demodulator: tx_cal frame received')
        # params
        npop = 30
        alpha = 5e-3
        lb = np.array([-1e5, -1e5, 1e-5])
        ub = np.array([1e5, 1e5, 3e-5])
        num_iter = 10
        sigma = 20
        dim = [2]

        print('old tx_params: ', self.tx_params)
        x = self.tx_params[dim]
        lb = lb[dim]
        ub = ub[dim]
        sigma = (ub - lb) / sigma
        for _ in tqdm(range(num_iter)):
            N = np.random.rand(npop, 1)
            R = np.zeros(npop)
            for j in range(npop):
                x_try = x + sigma * N[j]
                pred, _ = self.header_predict(We=self.tx_params[0], ke=self.tx_params[1], Ce=x_try)
                R[j] = np.mean(np.abs(pred - frame))
            A = (R - np.mean(R)) / np.std(R)
            x = x - alpha * (npop * sigma) * np.dot(N.T, A)
        self.tx_params[dim] = x
        np.savetxt('./result/tx_cal/tx_params.csv', self.tx_params, delimiter=',')
        print('new tx_params: ', self.tx_params)
        print('tx calibration done.')

        self.header_update()

    def header_update(self):
        v_header, t_header = self.header_predict(self.tx_params[0], self.tx_params[1], self.tx_params[2])
        plt.figure()
        plt.plot(t_header, v_header)
        plt.savefig('./result/header/header_pred.png')
        plt.close()
        np.savetxt('./result/header/header_pred.csv', v_header, delimiter=',')
        self.header_queue.put(v_header)
        print('header update done.')

    def cal_demodulate(self, frame):
        print('Demodulator: cal frame received')
        t = np.arange(0, len(frame)) / self.fs
        v_step = frame

        popt, pcov = optimize.curve_fit(self.pyro_v_step, t, v_step, p0=[1.0, 50.0, 5.0, 0.0])

        v_step_fit = self.pyro_v_step(t, *popt)
        plt.figure()
        plt.plot(t, v_step, t, v_step_fit)
        plt.savefig('./result/cal/v_step.png')
        plt.close()
        np.savetxt('./result/cal/v_step.csv', np.stack((t, v_step, v_step_fit), axis=-1), delimiter=',')
        np.savetxt('./result/cal/vstep_params.csv', popt, delimiter=',')

        self.pyro_params = np.array([popt[1] + popt[2], popt[1] * popt[2], popt[0] * (popt[1] - popt[2])])
        np.savetxt('./result/cal/pyro_params.csv', self.pyro_params, delimiter=',')
        print('v_step fitting result: ', popt)
        print('pyro calibration done.')

        # TODO: send header to syn
        self.header_update()

    def simulate_frame(self, n_frame, save_to='./result/simulate/dataset.pkl'):
        n = len(self.frame_header) + self.frame_bits
        spb = round(self.fs / self.bit_rate)
        t = np.linspace(start=0, stop=1.0 / self.bit_rate, num=spb + 1)

        dataset = {'x': [], 'y': []}
        for _ in tqdm(range(n_frame)):
            T_init, v_init, dvdt_init = self.Te, 0.0, 0.0
            bits = np.random.randint(2, size=n)
            bits[:len(self.frame_header)] = self.frame_header
            dataset['y'].append(bits)
            v_frame = []
            for i in range(n):
                bit = bits[i]
                we = bit * self.tx_params[0]
                v_bit, T_init, v_init, dvdt_init, info = \
                    self.bit_predict(t, T_init, v_init, dvdt_init, we, self.tx_params[1], self.tx_params[2])
                v_frame.append(v_bit[:-1])
            dataset['x'].append(np.array(v_frame))

        dataset['x'], dataset['y'] = np.array(dataset['x']), np.array(dataset['y'])
        with open(save_to, 'wb') as f:
            pickle.dump(dataset, f)

    def data_demodulate(self, frame):
        print('Demodulator: data frame received')
        n = len(self.frame_header) + self.frame_bits
        spb = round(self.fs / self.bit_rate)
        t = np.linspace(start=0, stop=1.0 / self.bit_rate, num=spb + 1)
        T_init, v_init, dvdt_init = self.Te, 0.0, 0.0
        digits = ()

        t_frame = np.linspace(start=0, stop=1.0 / self.bit_rate * n, num=spb * n + 1)
        v_f1 = np.zeros(t_frame.shape)
        v_f0 = np.zeros(t_frame.shape)
        T_f1 = np.zeros(t_frame.shape)
        T_f0 = np.zeros(t_frame.shape)
        pnor_f1 = np.zeros(t_frame.shape)
        pnor_f0 = np.zeros(t_frame.shape)

        for i in range(n):
            s_idx = i * spb
            e_idx = (i + 1) * spb + 1

            vr = frame[s_idx:e_idx]
            v1, T1_end, v1_end, dvdt1_end, info1 = \
                self.bit_predict(t, T_init, v_init, dvdt_init, self.tx_params[0], self.tx_params[1], self.tx_params[2])

            v0, T0_end, v0_end, dvdt0_end, info0 = \
                self.bit_predict(t, T_init, v_init, dvdt_init, 0.0, self.tx_params[1], self.tx_params[2])

            bit = self.sequence_matching(vr, v1, v0)
            digits += (bit,)

            if bit == 1:
                T_init, v_init, dvdt_init = T1_end, v1_end, dvdt1_end
            else:
                T_init, v_init, dvdt_init = T0_end, v0_end, dvdt0_end
            v_init = vr[-1]

            v_f1[s_idx:e_idx] = v1
            v_f0[s_idx:e_idx] = v0
            T_f1[s_idx:e_idx] = interpolate.interp1d(info1[0], info1[1], bounds_error=True)(t)
            T_f0[s_idx:e_idx] = interpolate.interp1d(info0[0], info0[1], bounds_error=True)(t)
            pnor_f1[s_idx:e_idx] = interpolate.interp1d(info1[0], info1[2], bounds_error=True)(t)
            pnor_f0[s_idx:e_idx] = interpolate.interp1d(info0[0], info0[2], bounds_error=True)(t)

            if i == (len(self.frame_header) - 1):
                if digits != self.frame_header:
                    print('Demodulator: wrong data frame header detected')
                    break

        print(digits[len(self.frame_header):])

        plt.figure()
        plt.plot(t_frame, v_f1, label='v1')
        plt.plot(t_frame, v_f0, label='v0')
        plt.plot(t_frame, frame, label='measured')
        plt.savefig('./result/frame/v.png')
        plt.close()

        plt.figure()
        plt.plot(t_frame, T_f1, label='T1')
        plt.plot(t_frame, T_f0, label='T0')
        plt.savefig('./result/frame/T.png')
        plt.close()

        plt.figure()
        plt.plot(t_frame, pnor_f1, label='P1')
        plt.plot(t_frame, pnor_f0, label='P0')
        plt.savefig('./result/frame/P.png')
        plt.close()

    def sequence_matching(self, vr, v1, v0, mode='l1'):
        if mode == 'l1':
            e1 = np.sum(np.abs(vr - v1))
            e0 = np.sum(np.abs(vr - v0))
        else:
            pass

        if e1 > e0:
            return 0
        else:
            return 1

    def bit_predict(self, t, T0, v0, dvdt0, We, ke, Ce, dt=1e-4, torch_solver='ode'):
        assert t[0] == 0.0
        t_grid = np.linspace(0, t[-1], int((t[-1] - t[0]) / dt) + 1)

        if torch_solver == 'ode':
            T_ode = integrate.odeint(self.torch_ode, T0, t_grid, args=(lambda x: We, ke, Ce))
            T = T_ode[:, 0]
        elif torch_solver == 'piecewise_series':
            T = self.torch_ode_piecewise_series_solv(t_grid, T0, We, ke, Ce)
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

    def torch_ode(self, T, t, We, ke, Ce):
        b = 1.311e-13

        dTdt = (We(t) - ke * (T - self.Te) - b * (T ** 4)) / Ce
        return dTdt

    def torch_ode_series_solv(self, t, T0, We, ke, Ce, order=3):
        b = 1.311e-13
        Te = self.Te

        a1 = (We + ke * (Te - T0) - b * (T0 ** 4)) / Ce
        a2 = - a1 * (ke + 4 * b * (T0 ** 3)) / (2 * Ce)
        a3 = (- a2 * (ke + 4 * b * (T0 ** 3)) - 6 * (a1 ** 2) * b * (T0 ** 2)) / (3 * Ce)

        if order == 3:
            return T0 + a1 * t + a2 * (t ** 2) + a3 * (t ** 3)
        elif order == 2:
            return T0 + a1 * t + a2 * (t ** 2)
        else:
            assert False

    def torch_ode_piecewise_series_solv(self, t, T0, We, ke, Ce, order=3, piece=3e-3):
        assert t[0] == 0
        eps = (t[1] - t[0]) / 3
        y = [T0]
        for i in range(1, t.shape[0]):
            t_mod = t[i] % piece
            if (t_mod < eps) or (t_mod > piece - eps):
                T = self.torch_ode_series_solv(piece, T0, We, ke, Ce, order)
                T0 = T
                y.append(T)
            else:
                T = self.torch_ode_series_solv(t_mod, T0, We, ke, Ce, order)
                y.append(T)
        return np.array(y)

    def get_channel_info(self):
        bpf = np.genfromtxt('./data/filter/%s.csv' % self.channel_id, delimiter=',', dtype=np.float32)
        bpf_trans = interpolate.interp1d(bpf[:, 0], bpf[:, 1], bounds_error=True)

        atmos = np.genfromtxt('./data/atmos/%s.csv' % self.channel_range, delimiter=',', dtype=np.float32)
        atmos_trans = interpolate.interp1d(atmos[:, 0], atmos[:, 1], bounds_error=True)

        max_wl, min_wl = bpf[:, 0].max(), bpf[:, 0].min()
        wavelength = np.concatenate((bpf[:, 0], atmos[:, 0]))
        wavelength = np.sort(wavelength[(wavelength <= max_wl) & (wavelength >= min_wl)])

        def channel_trans(wl):
            # 0.72 for ZnSe lense
            return 0.72 * 0.72 * atmos_trans(wl) * bpf_trans(wl) * bpf_trans(wl)

        return channel_trans, wavelength
        # return lambda x: 1.0, np.arange(0.1, 50, 0.1)

    def planck_equ(self, wl, T):
        c1 = 1.1910428661813628e8
        c2 = 1.438776749195487e4
        # W/Sr/m^2/um
        return c1 / (wl ** 5) / (np.exp(c2 / wl / T) - 1)

    def get_rx_power(self, T, integrate_method='trapz'):

        channel_trans, wavelength = self.channel_info

        def rx_spectrum(wl):
            return self.planck_equ(wl, T) * channel_trans(wl)

        if integrate_method == 'trapz':
            x = wavelength
            y = rx_spectrum(wavelength)
            return np.trapz(y, x=x) * np.pi * 0.8 * 2.89e-6
        elif integrate_method == 'quad':
            result = integrate.quad(rx_spectrum, wavelength[0], wavelength[-1])
            return result[0] * np.pi * 0.8 * 2.89e-6

    def get_normalized_rx_power_func(self, dT=0.5):
        p_step = self.get_rx_power(1023)
        T = np.arange(250, 1100, dT)
        p_nor = []
        for i in T:
            p_nor.append(self.get_rx_power(i) / p_step)
        func = interpolate.interp1d(T, p_nor, bounds_error=True)
        return func

    @staticmethod
    def pyro_v_step(x, p0, p1, p2, p3):
        return p0 * (np.exp(-p1 * x) - np.exp(-p2 * x)) + p3

    def pyro_ode(self, x, t, dpdt):
        dx0dt = x[1]
        dx1dt = - (self.pyro_params[0] * x[1] + self.pyro_params[1] * x[0] + self.pyro_params[2] * dpdt(t))
        return [dx0dt, dx1dt]


if __name__ == '__main__':
    cfg = {
        'fs': 1e3,
        'channel_id': 'single',
        'channel_range': 1000,
        'bit_rate': 50,
        'frame_header': (1, 1, 1, 0),
        'frame_bits': 8,
    }

    test = Demodulator(
        header_queue=queue.Queue(maxsize=0),
        sample_freq=cfg['fs'],
        bit_rate=cfg['bit_rate'],
        frame_header=cfg['frame_header'],
        frame_bits=cfg['frame_bits'],
        channel_id=cfg['channel_id'],
        channel_range=cfg['channel_range']
    )
    # import matplotlib.pyplot as plt
    #
    # d = Demodulator()
    # print(d.get_power(1023, integrate_method='trapz'))
    # wl = np.arange(1, 20, 0.1)
    # p = d.planck_equ(wl, 1023)
    # p = p * np.pi * 0.8 * 2.89e-6
    #
    # plt.plot(wl, p)
    # plt.show()

    # t = np.arange(0, 50e-3, 1e-3)
    # y0 = 293.15
    # w = 0.8978
    #
    # y = d.torch_ode_piecewise_series_solv(t, y0, w, order=3)
    # # print(y)
    # plt.plot(t, y)
    # plt.xlim([0, 50e-3])
    # plt.ylim([200, 1100])
    # plt.show()
