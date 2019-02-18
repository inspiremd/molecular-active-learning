import numpy as np
from Generator import Generator
from sklearn import datasets

def main():
    data = datasets.load_breast_cancer(return_X_y=False)['data']
    generator = Generator(data).generator



if __name__ == '__main__':
    main()
