import argparse

import torch
import torch.nn as nn
from mpi4py import MPI
from torch.utils.data import DataLoader, TensorDataset

import learning.models.model as models



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store_true')
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--smiles_file', type=str, required=True)
    parser.add_argument('--mpi', action='store_true')
    return parser.parse_args()


def get_data_loader():
    from sklearn.datasets import make_regression
    X, y = make_regression(10000, n_features=10, n_targets=1, n_informative=3)
    X = torch.from_numpy(X)
    y = torch.from_numpy(y)
    train_loader = DataLoader(TensorDataset(X, y), pin_memory=True, num_workers=2, batch_size=128)
    test_loader = DataLoader(TensorDataset(X, y), pin_memory=True, num_workers=2, batch_size=128)

    return train_loader, test_loader


def main(args):
    if args.mpi:
        # comm = MPI.COMM_WORLD
        # size = comm.Get_size()
        # rank = comm.Get_rank()
        print("not implemented.")
        exit(0)
    else:
        print("Single-user non-MPI mode.")
        trainer = models.Trainer.create_new_trainer(models.TwoLayerNet, 10, nn.MSELoss, args.o)
        train_loader, test_loader = get_data_loader()
        if args.f:
            trainer.train(train_loader, epochs=10)


if __name__ == '__main__':
    args = get_args()
    main(args)
