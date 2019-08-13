import numpy as np
from Generator import Generator
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_location", type=str)
    parser.add_argument("--n", type=int)
    parser.add_argument("--data", type=str)

    return parser.parse_args()

def main(args):
    tb = pd.read_csv(args.data, header=None, sep='\t', names=['smiles'])['smiles']

    data = np.array(tb).reshape(-1,1)
    generator = Generator(data).generator

    with open(args.file_location, 'w') as f:
        for i in generator():
            f.write(i)
            f.write('\n')



if __name__ == '__main__':
    args = get_args()
    main(args)
