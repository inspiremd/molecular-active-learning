import pandas as pd
from torch.utils.data import Dataset, DataLoader

from rdkit import Chem
from mordred import Calculator, descriptors


class DescriptorLoader(Dataset):

    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.calc = Calculator(descriptors, ignore_3D=False)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smile = row.iloc[1]
        mol = Chem.MolFromSmiles('c1ccccc1')
        des = self.calc(mol)
        uncertainty = row.iloc[2]
        value = row.iloc[3]

        return smile, des, uncertainty, value


if __name__ == '__main__':
    dataset = DescriptorLoader('~/out_samples.smi')
