import argparse
import pandas as pd
import glob
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-o', type=str, required=True)
    parser.add_argument('-s', type=str, required=True)
    return parser.parse_args()

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
            res.append((self.name, "Dock", self.dock))
        if self.mmgbsa is not None:
            res.append((self.name, "MMGBSA", self.mmgbsa))
        if self.min is not None:
            res.append((self.name, "Minimization", self.min))
        if len(res) == 0:
            return None
        else:
            return res

class Agg(object):
    def __init__(self, df, i, o):
        self.i = i
        self.o = o
        self.df = df

        self.logs = {} #index by name of molecule

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
        for _,log in self.logs.items():
            res.append(log.to_tuples())
        res = list(filter(lambda x : x is not None, res))
        res = [y for x in res for y in x]
        res = list(zip(*res))
        print(res)
        try:
            df = pd.DataFrame(list(zip(*res)), columns=['name', 'property', 'value'])
        except AssertionError:
            return False
        df.to_hdf(self.o + "out.hdf", key='data')
        return True

def run_agg(i, o, s):
    df = read_smile_file(s)
    agger = Agg(df, i, o)

    while True:
        try:
            print("Scanning.")
            agger.scan()
            res = agger.log()
            if res:
                print("Log success.")
            else:
                print("Log failure.")
        except KeyboardInterrupt:
            print("Interupt. Exiting.")
            exit(0)
        print("Sleeping...")
        time.sleep(60)

if __name__ == '__main__':
    args = parse_args()
    run_agg(args.i, args.o, args.s)
