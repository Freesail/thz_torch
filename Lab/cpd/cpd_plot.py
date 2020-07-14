import pickle
import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('./result/sync/cpd_plot.pkl', 'rb') as f:
        data = pickle.load(f)
    print(data['bkp_m'], data['bkp_p'])

    # rpt.show.display(data['pre_cpd'], data['bkp_m'], figsize=(10, 6))
    # plt.show()
    t = np.linspace(0, 80, 161)

    s = 7
    params = {'axes.labelsize': s, 'xtick.labelsize': s, 'ytick.labelsize': s, 'legend.fontsize': s}
    plt.rcParams.update(params)

    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(6, 3.5))
    plt.subplots_adjust(hspace=.2)

    bkp_m = data['bkp_m'][:-1]
    ax = axs[0]
    ax.plot(t, data['pre_cpd'], color='r', label='unaligned')
    ax.scatter(t[bkp_m], data['pre_cpd'][bkp_m], s=30, c='r', marker='*')
    for v in t[bkp_m]:
        ax.axvline(v, c='r', ls='--')
    ax.set_ylim(-2, 0)
    ax.set_ylabel('Measured sensor voltage\n$v$ (V)')

    bkp_p = data['bkp_p'][:-1]
    ax = axs[1]
    ax.plot(t, data['p'], color='k')
    ax.scatter(t[bkp_p], data['p'][bkp_p], c='k', s=30, marker='*')
    for v in t[bkp_p]:
        ax.axvline(v, c='k', ls='--')
    ax.set_ylim(-2, 0)
    ax.set_ylabel('Predicted sensor voltage\n$v$ (V)')

    ax = axs[0]
    ax.plot(t, data['post_cpd'], color='b', label='aligned')
    ax.scatter(t[bkp_p], data['post_cpd'][bkp_p], c='b', s=30, marker='*')
    for v in t[bkp_p]:
        ax.axvline(v, c='b', ls='--')
    ax.legend()

    # ax.set_ylim(-2, 0)
    # ax.set_ylabel('Measured sensor voltage\naligned, $v$ (V)')

    ax = axs[-1]
    ax.set_xlim(-1, 81)
    ax.set_xlabel('Time, $t$ (ms)')

    plt.savefig('./result/sync/cpd.png', bbox_inches='tight', dpi=500)
    plt.close()
