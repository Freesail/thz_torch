import os
import numpy as np

src_dir = '../data/atmos_raw/'

dst_dir = '../data/atmos/'

all = os.listdir(src_dir)
print(all)

for file in all:
    name = src_dir + file
    a = np.genfromtxt(name, dtype=np.float32, skip_header=6)
    a[:, 0] = 1E4 / a[:, 0]

    a[a[:, 1] > 1, 1] = 1
    a[a[:, 1] < 0, 1] = 0

    dst_name = (dst_dir + file).replace('txt', 'csv')
    np.savetxt(dst_name, a, delimiter=',', fmt='%1.5e')

    # break