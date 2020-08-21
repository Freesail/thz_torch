import pickle
import os

# with open('./2000mm_50bps_v1.pkl', 'rb') as f:
#     data = pickle.load(f)
#
# print(data['params'][0])

for name in os.listdir(path='./'):
    if name[:6] == '2000mm':
        with open(name, 'rb') as f:
            data = pickle.load(f)
        print(data['params'][0])