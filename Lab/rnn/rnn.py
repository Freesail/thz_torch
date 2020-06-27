import pickle
import os

import torch
from torch.optim import Adam
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split


class ThzTorchDataset(Dataset):
    def __init__(self, pkl_file, device):
        self.device = device
        with open(pkl_file, 'rb') as f:
            self.data = pickle.load(f)
        for k, v in self.data.items():
            self.data[k] = torch.from_numpy(v).float().to(device)

        self.preprocess()
        self.x_size = self.data['x'].size()[-1]
        self.p_size = self.data['params'].size()[-1]

    def preprocess(self):
        # choose dims
        self.data['params'] = self.data['params'][:, 2:]

        # normalize
        self.x_mu, self.x_std = self.data['x'].mean(), self.data['x'].std()
        self.data['x'] = (self.data['x'] - self.x_mu) / self.x_std

        self.p_mu, self.p_std = self.data['params'].mean(dim=0), self.data['params'].std(dim=0)
        self.data['params'] = (self.data['params'] - self.p_mu) / self.p_std

        # print(mu.size(), std.size())
        # print(self.data['params'])
        # assert False

    def __len__(self):
        return self.data['x'].shape[0]

    def __getitem__(self, idx):
        sample = {}
        for k, v in self.data.items():
            sample[k] = v[idx]
        return sample


class ThzTorchTestset(Dataset):
    def __init__(self, path, device, trainset, datafile='dataset.pkl', labelfile='labels.pkl'):
        self.device = device
        self.path = path
        self.trainset = trainset
        with open(os.path.join(path, datafile), 'rb') as f:
            self.data = pickle.load(f)
        with open(os.path.join(path, labelfile), 'rb') as f:
            self.data['y'] = pickle.load(f)

        for k, v in self.data.items():
            self.data[k] = torch.from_numpy(v).float().to(device)

        self.preprocess()
        # self.x_size = self.data['x'].size()[-1]
        # self.p_size = self.data['params'].size()[-1]

    def preprocess(self):
        # choose dims
        self.data['params'] = self.data['params'][:, 2:]

        # normalize
        self.data['x'] = (self.data['x'] - self.trainset.x_mu) / self.trainset.x_std
        self.data['params'] = (self.data['params'] - self.trainset.p_mu) / self.trainset.p_std

    def __len__(self):
        return self.data['x'].shape[0]

    def __getitem__(self, idx):
        sample = {}
        for k, v in self.data.items():
            sample[k] = v[idx]
        return sample


class ThzTorchModel(nn.Module):
    def __init__(self,
                 x_size,
                 p_size,
                 lstm_hidden_size,
                 mlp_hidden_size,
                 device):
        super(ThzTorchModel, self).__init__()

        self.x_size = x_size
        self.p_size = p_size
        self.lstm_hidden_size = lstm_hidden_size
        self.mlp_hidden_size = mlp_hidden_size
        self.device = device

        self.lstm = nn.LSTM(
            input_size=x_size,
            hidden_size=lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )

        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_size + p_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, 1),
            nn.Sigmoid(),
        )

        self.to(device)

    def forward(self, x, params=None):
        lstm_out, _ = self.lstm(x)
        # print(lstm_out.size())
        lstm_out = lstm_out.contiguous().view(-1, self.lstm_hidden_size)
        # print(lstm_out.size())

        seq_len = x.size()[1]
        # print(params)
        params = torch.repeat_interleave(params, repeats=seq_len, dim=0)
        # print(params.size())

        mlp_in = torch.cat([lstm_out, params], dim=-1)
        mlp_out = self.mlp(mlp_in)
        return mlp_out.view(-1, seq_len)


def train_model(datapath, testpath, n_epoch, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # data
    data_set = ThzTorchDataset(
        pkl_file=datapath,
        device=device
    )
    data_size = len(data_set)
    train_size = int(data_size * 0.8)

    train_set, val_set = random_split(data_set, [train_size, data_size - train_size])
    test_set = ThzTorchTestset(testpath, device, data_set)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )

    val_loader = DataLoader(
        val_set, batch_size=len(val_set), shuffle=True
    )

    test_loader = DataLoader(
        test_set, batch_size=len(test_set), shuffle=True
    )

    # model
    model = ThzTorchModel(
        x_size=data_set.x_size,
        p_size=data_set.p_size,
        lstm_hidden_size=64,
        mlp_hidden_size=32,
        device=device
    )

    # opt
    loss_fn = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=3e-4)

    for epoch in range(1, n_epoch + 1):
        print('Epoch %d' % epoch)
        # train
        print('TRAIN')
        for i, batch in enumerate(train_loader):
            outputs = model(batch['x'], batch['params'])
            loss = loss_fn(outputs, batch['y'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                preds = (outputs > 0.5).float()
                batch_acc = torch.sum(preds == batch['y']).float() / preds.numel()
                print('batch %d - loss: %.3f | acc: %.3f' % (i, loss.item(), batch_acc.item()))
        # val
        print('VAL')
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                outputs = model(batch['x'], batch['params'])
                loss = loss_fn(outputs, batch['y'])
                preds = (outputs > 0.5).float()
                acc = torch.sum(preds == batch['y']).float() / preds.numel()
                print('val - loss: %.3f | acc: %.3f' % (loss.item(), acc.item()))

        # test
        print('TEST')
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                outputs = model(batch['x'], batch['params'])
                loss = loss_fn(outputs, batch['y'])
                preds = (outputs > 0.5).float()
                acc = torch.sum(preds == batch['y']).float() / preds.numel()
                print('val - loss: %.3f | acc: %.3f' % (loss.item(), acc.item()))


if __name__ == '__main__':
    train_model(1, 2)
