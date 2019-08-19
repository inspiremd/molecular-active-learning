import argparse
import learning.models.model as models
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store_true')
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    return parser.parse_args()

def get_data_loader():
    from sklearn.datasets import make_regression
    X,y = make_regression(10000, n_features=10, n_targets=1, n_informative=3)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    train_loader = DataLoader(TensorDataset(X,y), pin_memory=True, num_workers=2, batch_size=128)
    test_loader  = DataLoader(TensorDataset(X,y), pin_memory=True, num_workers=2, batch_size=128)

    return train_loader, test_loader

def main(args):
    print("learning stuff.")
    trainer = models.Trainer.create_new_trainer(models.TwoLayerNet, 10, nn.MSELoss, args.o)
    train_loader, test_loader = get_data_loader()
    if args.f:
        trainer.train(train_loader, epochs=10)


if __name__ == '__main__':
    args = get_args()
    main(args)