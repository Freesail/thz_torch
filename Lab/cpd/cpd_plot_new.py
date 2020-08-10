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

    plt.figure(figsize=(6, 2.5))

    plt.ylim(-2, 0)
    plt.ylabel('Measured and predicted\nsensor output voltage $v$ (V)')

    plt.xlim(0, 80)
    plt.xlabel('Time, $t$ (ms)')

    bkp_p = data['bkp_p'][:-1]
    plt.plot(t, data['p'], color='k', label='predicted')
    plt.scatter(t[bkp_p], data['p'][bkp_p], c='k', s=30, marker='*')

    bkp_m = data['bkp_m'][:-1]
    plt.plot(t, data['pre_cpd'], color='r', label='sychronized after\nmatching phase')
    plt.scatter(t[bkp_m], data['pre_cpd'][bkp_m], s=30, c='r', marker='*')

    plt.plot(t, data['post_cpd'], color='b', label='sychronized after\njitter-reduction phase')
    plt.scatter(t[bkp_p], data['post_cpd'][bkp_p], c='b', s=30, marker='*')
    plt.legend(loc='lower left', framealpha=1.0)

    ax = plt.gca()
    for i in [20, 40, 60, 80]:
        if i != 80:
            ax.axvline(i, c='grey', ls='--', lw=1.0)
            plt.text(x=i - 2.5, y=-0.15, s='1', c='grey')
        else:
            plt.text(x=i - 2.5, y=-0.15, s='0', c='grey')

    plt.savefig('./result/sync/cpd.png', bbox_inches='tight', dpi=500)
    plt.close()
