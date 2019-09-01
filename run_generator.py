import argparse

import numpy as np
import pandas as pd

from generator import Generator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", type=str, help='output path', required=True)
    parser.add_argument('-i', action='store_true')
    parser.add_argument("-n", type=int, help='number of samples to provide', required=True)
    parser.add_argument("-d", type=str, help='path to database csv or file', required=False)
    parser.add_argument("-s", type=str, help="learner file directory", required=True)

    return parser.parse_args()


def main(args):
    if args.d is not None:
        tb = pd.read_csv(args.d, header=None, sep='\t', names=['smiles'])['smiles']
    else:
        from sklearn.datasets import make_regression
        X, _ = make_regression(n_samples=100000, n_features=2, n_targets=1)
        tb = pd.DataFrame(X)

    data = np.array(tb).reshape(-1, 1)
    generator = Generator.Generator(data).generator

    with open(args.o + "run.txt", 'w') as f:
        count = 0
        for i in generator():
            if count >= args.n:
                break
            f.write(str(i[0]))
            f.write('\n')
            count += 1


if __name__ == '__main__':
    args = get_args()
    if args.i:
        main(args)
    else:
        print("Error: uncertainty sampling not supported yet. Can only run -i")
        exit()
