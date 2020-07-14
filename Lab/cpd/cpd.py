import numpy as np
import ruptures as rpt
import matplotlib.pyplot as plt

if __name__ == '__main__':
    signal = np.genfromtxt('./result/sync/sync.csv', delimiter=',')
    t = np.arange(signal.shape[0])
    signal = np.column_stack((signal.reshape(-1, 1), t))
    print(signal.shape)
    algo = rpt.Dynp(model='linear').fit(signal)
    my_bkps = algo.predict(n_bkps=3)
    print(my_bkps)

    rpt.show.display(signal, my_bkps, figsize=(10, 6))
    plt.show()
