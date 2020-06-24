import pickle
import matplotlib.pyplot as plt

if __name__ == '__main__':
    with open('./result/simulate/dataset.pkl', 'rb') as f:
        dataset = pickle.load(f)

    X, Y = dataset['x'], dataset['y']
    i = 1
    print(X[i], Y[i])
    sx = X[i].flatten()
    plt.plot(sx)
    plt.show()




