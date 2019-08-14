import numpy as np
from Generator import Generator
import pandas as pd
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str)
    parser.add_argument('-i', action='store_true')
    parser.add_argument("-n", type=int)
    parser.add_argument("-d", type=str)

    return parser.parse_args()

def main(args):
    tb = pd.read_csv(args.d, header=None, sep='\t', names=['smiles'])['smiles']

    data = np.array(tb).reshape(-1,1)
    generator = Generator(data).generator

    with open(args.file_location, 'w') as f:
        for i in generator():
            f.write(i)
            f.write('\n')



if __name__ == '__main__':
    args = get_args()
    if args.i:
        main(args)
    else:
        print("Error: uncertainty sampling not supported yet. Can only run -i")
        exit()
