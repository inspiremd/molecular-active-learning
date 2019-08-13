import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim


class TwoLayerNet(nn.Module):
    def __init__(self, feat_size):
        super(TwoLayerNet, self).__init__()
        self.ln1 = nn.Linear(feat_size, 100)
        self.ln2 = nn.Linear(100, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.ln2(self.relu(self.ln1(x)))



class Trainer():
    def __init__(self, model, feats, data_loader, loss):
        self.device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')

        self.model = model(feats).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        self.data_loader = data_loader
        self.loss = loss(reduce='mean')

    def train_epoch(self):
        for i, x, y in enumerate(self.data_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            y_pred = self.model(x)
            loss = self.loss(y,y_pred)
            loss.backward()
            self.optimizer.step()

    def train(self, epochs=1):
        for i in range(epochs):
            self.train_epoch()
