import pickle
import numpy as np
import os

if __name__ == '__main__':
    base_path = '../ber_2500/'
    file = '2000mm_125bps_v1.pkl'

    with open(os.path.join(base_path, 'gt_%s' % file), 'rb') as f:
        gt = pickle.load(f)

    with open(os.path.join(base_path, 'offline_%s' % file), 'rb') as f:
        offline = pickle.load(f)

    gt = gt[:, 4:]
    offline = offline[:, 4:]
    # print(gt.shape[0])
    print(np.sum(offline == gt) / gt.shape[0] / gt.shape[1])
