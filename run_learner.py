import argparse
import glob
import time

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import learning.models.model as models


def read_smile_file(f):
    df = pd.read_csv(f, header=None, names=['smile', 'name'], sep=' ')
    return df


class RunData(object):
    def __init__(self):
        self.smile = None
        self.name = None
        self.mmgbsa = None
        self.dock = None
        self.min = None

    def to_tuples(self):
        res = []
        if self.dock is not None:
            res.append((self.name, self.smile, "Dock", self.dock))
        if self.mmgbsa is not None:
            res.append((self.name, self.smile, "MMGBSA", self.mmgbsa))
        if self.min is not None:
            res.append((self.name, self.smile, "Minimization", self.min))
        if len(res) == 0:
            return None
        else:
            return res


class Agg(object):
    def __init__(self, df, i):
        self.i = i
        self.df = df

        self.logs = {}  # index by name of molecule

    def scan(self):
        scanned_dirs = glob.glob(self.i + "*/")
        for dir in scanned_dirs:
            mol_name = dir.split("/")[-2]

            if mol_name in self.logs.keys():
                mol_data = self.logs[mol_name]
            else:
                mol_data = RunData()
                try:
                    mol_data.smile = self.df[self.df.name == mol_name].iloc[0].loc['smile']
                except:
                    print("Error looking up", mol_name)
                mol_data.name = mol_name

            metrics_csv = pd.read_csv(dir + "metrics.csv")
            if mol_data.dock is None and "Dock" in metrics_csv.columns:
                mol_data.dock = metrics_csv.loc[:, 'Dock'].iloc[0]
            if mol_data.dock is None and "Minimize" in metrics_csv.columns:
                mol_data.min = metrics_csv.loc[:, 'Minimize'].iloc[0]
            if mol_data.dock is None and "MMGBSA" in metrics_csv.columns:
                mol_data.mmgbsa = metrics_csv.loc[:, 'MMGBSA'].iloc[0]

            self.logs[mol_name] = mol_data

    def log(self):
        res = []
        for _, log in self.logs.items():
            res.append(log.to_tuples())
        res = list(filter(lambda x: x is not None, res))
        res = [y for x in res for y in x]
        res = list(zip(*res))
        print(res)
        try:
            df = pd.DataFrame(list(zip(*res)), columns=['name', 'smile', 'property', 'value'])
            return df
        except AssertionError:
            return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', action='store_true')
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--smiles_file', type=str, required=True)
    parser.add_argument('--mpi', action='store_true')
    parser.add_argument('--min_start', type=int, required=False, default=50000)
    parser.add_argument('--feature_df', type=str, required=True)
    return parser.parse_args()


def get_data_loader(df, smiles, features):

    df = pd.merge(df, features, on='name', how='inner')
    X = df.drop(['name', 'smile', 'property', 'value'], axis=1)
    X = torch.from_numpy(np.array(X.apply(lambda x : pd.to_numeric(x, errors='coerce'), axis=1)).astype(np.float32))
    y = torch.from_numpy(np.array(pd.to_numeric(df.value, errors='coerce')).astype(np.float32))

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
        print("This module will run persistantly and log continuously. Please exit with CTRL-C")
        print("Single-user non-MPI mode.")
        print("Loading input directory {}".format(args.data_path))
        print("This program will stall until {} files have been loaded.".format(args.min_start))

        smiles_input = read_smile_file(args.smiles_file)
        assert (smiles_input.shape[1] == 2)
        print("Loaded smiles input file with {} smiles".format(smiles_input.shape[0]))
        agregator = Agg(smiles_input, args.data_path)

        for i in range(10000):
            df = None
            while df is None or df.shape[0] < args.min_start:
                agregator.scan()
                df = agregator.log()
                print("Loaded {} molecules so far...")
                if df.shape[0] < args.min_start:
                    print("{} not met yet. Sleeping for 60 seconds.".format(args.min_start))
                    time.sleep(60)

            print("Loaded {} simulation properties... featurizing moleclues now...".format(df.shape[0]))
            smiles_loaded = list(set(df.smile.tolist()))
            feature_df = pd.read_csv(args.feature_df)

            train_loader, test_loader = get_data_loader(df, smiles_loaded, feature_df)

            trainer = models.Trainer.create_new_trainer(models.TwoLayerNet, 10, nn.MSELoss, args.o)
            if args.f:
                trainer.train(train_loader, epochs=10)
            trainer.checkpont(file_prefix=str(df.shape[0]))


if __name__ == '__main__':
    args = get_args()
    main(args)
