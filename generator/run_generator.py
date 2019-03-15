import numpy as np
from Generator import Generator
import pandas as pd
import argparse


def main():
    data = np.array(pd.read_table("moses_cleaned.tab", header=None, names=['id', 'smiles'])['smiles']).reshape(-1,1)
    generator = Generator(data).generator
    for i in generator():
        print(i)



if __name__ == '__main__':
    main()
