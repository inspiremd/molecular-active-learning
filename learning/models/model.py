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
    def __init__(self, model, optimizer, loss):
        self.device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
        self.model = model
        self.optimizer = optimizer
        self.loss = loss


    @classmethod
    def load_trainer(self, model, pt_file):
        device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')

        pt = torch.load(pt_file)
        model = model(pt['feats']).to(device)
        model.load_state_dict(pt['model_state'])
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(pt['optimizer_state'])
        loss = pt['loss_f']

        return Trainer(model, optimizer, loss)

    @classmethod
    def create_new_trainer(self, model, feats, loss):
        device = torch.device('cuda:0' if torch.cuda.is_avaliable() else 'cpu')
        model = model(feats).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss = loss(reduce='mean')

        return Trainer(model, optimizer, loss)

    def train_epoch(self, data_loader):
            for i, x, y in enumerate(data_loader):
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = self.model(x)
                loss = self.loss(y,y_pred)
                loss.backward()
                self.optimizer.step()

    def train(self, data_loader, epochs=1):
        for i in range(epochs):
            self.train_epoch(data_loader)
