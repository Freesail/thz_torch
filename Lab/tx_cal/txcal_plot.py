from Rx.Demodulator import Demodulator
import queue
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    cfg = {
        'fs': 2e3,
        'channel_id': 'single',
        'channel_range': 1000,
        'bit_rate': 50,
        'frame_header': (1, 1, 1, 0),
        'frame_bits': 50,
    }

    dem = Demodulator(
        header_queue=queue.Queue(maxsize=0),
        sample_freq=cfg['fs'],
        bit_rate=cfg['bit_rate'],
        frame_header=cfg['frame_header'],
        frame_bits=cfg['frame_bits'],
        channel_id=cfg['channel_id'],
        channel_range=cfg['channel_range']
    )

    with open('./result/sync/cpd_plot.pkl', 'rb') as f:
        frame = pickle.load(f)['post_cpd']

    # def func()
    #
    # result = dem.tx_cal_demodulate(frame)
    # with open('./result/tx_cal/trace.pkl', 'wb') as f:
    #     pickle.dump(result, f)
    #
    # trace, pred_new, pred_old = result

    with open('./result/tx_cal/trace.pkl', 'rb') as f:
        trace, pred_new, pred_old = pickle.load(f)

    # contour
    lb = np.array([0.80, -1e5, 1.5e-5])
    ub = np.array([0.96, 1e5, 3.1e-5])

    # x = np.arange(lb[0], ub[0], (ub[0] - lb[0])/50)
    # y = np.arange(lb[2], ub[2], (ub[2] - lb[2])/50)
    # X, Y = np.meshgrid(x, y)
    # Z = np.zeros_like(X)
    # for i in tqdm(range(X.shape[0])):
    #     for j in range(X.shape[1]):
    #         pred, _ = dem.header_predict(We=X[i, j], ke=dem.tx_params[1], Ce=Y[i, j])
    #         Z[i, j] = np.mean(np.abs(pred - frame))
    # with open('./result/tx_cal/contour.pkl', 'wb') as f:
    #     pickle.dump((X, Y, Z), f)

    with open('./result/tx_cal/contour.pkl', 'rb') as f:
        X, Y, Z = pickle.load(f)

    s = 10
    params = {'axes.labelsize': s, 'xtick.labelsize': s, 'ytick.labelsize': s, 'legend.fontsize': 9}
    plt.rcParams.update(params)

    fig = plt.figure(figsize=(11.5, 4))

    plt.subplot(121)
    plt.contourf(X, Y, Z, levels=10, cmap='summer')
    plt.xlim([0.8, 0.95])
    plt.ylim([1.5e-5, 2.9e-5])
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-5, -5), useMathText=True)
    plt.xlabel('Emitter input power, $W_R$ (W)')
    plt.ylabel('Emitter heat capacity, $C_e$ (J/K)')
    plt.scatter(trace[1:-1, 0], trace[1:-1, 1], c='k', s=30, marker='o')
    plt.scatter(trace[0, 0], trace[0, 1], c='r', s=50, marker='^')
    plt.scatter(trace[-1, 0], trace[-1, 1], c='w', s=50, marker='^')
    plt.plot(trace[:, 0], trace[:, 1], c='k', linestyle='--')
    plt.title('a\n', loc='left', weight='bold')

    plt.subplot(122)
    t = np.linspace(0, 80, 161)
    plt.xlim(0, 80)
    plt.ylim(-2, 0)
    plt.plot(t, frame, 'r', label='measurement')
    plt.plot(t, pred_new, 'k', label='prediction (optimized)')
    plt.plot(t, pred_old, 'b', label='prediction (original)')
    plt.xlabel('Time, $t$ (ms)')
    plt.ylabel('Sensor output voltage, $v$ (V)')
    plt.legend(loc='lower left', framealpha=1.0)

    ax = plt.gca()
    for i in [20, 40, 60, 80]:
        if i != 80:
            ax.axvline(i, c='grey', ls='--', lw=1.0)
            plt.text(x=i - 2.5, y=-0.15, s='1', c='grey')
        else:
            plt.text(x=i - 2.5, y=-0.15, s='0', c='grey')

    plt.title('b\n', loc='left', weight='bold')

    plt.savefig('./result/tx_cal/txcal.png', bbox_inches='tight', dpi=500)
    plt.close()


