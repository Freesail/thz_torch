import pickle

with open('./2000mm_50bps_v2.pkl', 'rb') as f:
    data = pickle.load(f)

print(data['params'][0])