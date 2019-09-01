import torch
import torch.cuda
import torch.nn as nn
import torch.optim as optim


class TwoLayerNet(nn.Module):
    def __init__(self, feat_size):
        super(TwoLayerNet, self).__init__()
        self.feat_size = feat_size
        self.ln1 = nn.Linear(feat_size, 100)
        self.mu = nn.Linear(100, 1)
        self.sigma = nn.Linear(100, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.ln1(x))
        return self.mu(x)


class Trainer():
    def __init__(self, model, optimizer, loss, config_dir):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        self.dir = config_dir

    @classmethod
    def load_trainer(self, model, pt_file, config_dir):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        pt = torch.load(pt_file)
        model = model(pt['feats']).to(device)
        model.load_state_dict(pt['model_state'])
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        optimizer.load_state_dict(pt['optimizer_state'])
        loss = pt['loss_f']

        return Trainer(model, optimizer, loss, config_dir)

    @classmethod
    def create_new_trainer(self, model, feats, loss, config_dir):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model = model(feats).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss = loss(reduce='mean')

        return Trainer(model, optimizer, loss, config_dir)

    def checkpont(self):
        torch.save({
            'model_state' : self.model.state_dict(),
            'optimizer_state' : self.model.state_dict(),
            'feats' : self.model.feat_size,
            'loss_f' : self.loss
        }, self.dir + "checkpoint.pt")


    def train_epoch(self, data_loader):
        for i, (x, y) in enumerate(data_loader):
            x = x.float().to(self.device)
            y = y.float().to(self.device)

            y_pred = self.model(x)
            loss = self.loss(y, y_pred)
            loss.backward()
            self.optimizer.step()

    def train(self, data_loader, epochs=10):
        for i in range(epochs):
            self.train_epoch(data_loader)
        self.checkpont()
