import pickle
import numpy as np

if __name__ == '__main__':
    file = '2000mm_50bps_v2.pkl'
    with open('gt_%s' % file, 'rb') as f:
        gt = pickle.load(f)

    with open('offline_%s' % file, 'rb') as f:
        offline = pickle.load(f)

    gt = gt[:, 4:]
    offline = offline[:, 4:]
    # print(gt.shape[0])
    print(np.sum(offline == gt) / gt.shape[0] / gt.shape[1])
