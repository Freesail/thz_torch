import pickle
import numpy as np

with open('gt_2000mm_50bps.pkl', 'rb') as f:
    gt = pickle.load(f)

with open('offline_2000mm_50bps.pkl', 'rb') as f:
    offline = pickle.load(f)

print(np.sum(offline == gt)/gt.shape[0]/gt.shape[1])

offline[0, 0] = 1 - offline[0, 0]
print(np.sum(offline == gt)/gt.shape[0]/gt.shape[1])