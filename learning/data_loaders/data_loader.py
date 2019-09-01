import pandas as pd
from rdkit import Chem
from torch.utils.data import Dataset

from learning.featurizer.DescriptorFeatureizer import DescriptorFeaturizer


class GeneralMolDataset(Dataset):

    def __init__(self, csv_file, mol_featuerizer=DescriptorFeaturizer):
        """
        General Molecular Dataset
        :param csv_file: single row, where the column headers are [Dock], [Minimize], and/or [MMGBSA]
        :param mol_featuerizer: A learning.featureizer.Featureizer object
        """
        self.data = pd.read_csv(csv_file)
        self.calc = mol_featuerizer(ignore_3D=False)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        smile = row.iloc[1]
        mol = Chem.MolFromSmiles('c1ccccc1')
        des = self.calc(mol)
        uncertainty = row.iloc[4]
        value = row.iloc[2]
        method = row.iloc[3]

        return smile, des, uncertainty, value
