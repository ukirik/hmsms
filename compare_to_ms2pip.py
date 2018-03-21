import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Parse and compare MS2PIP results')
parser.add_argument('-f', '--files', nargs='+', help='files contaning peptides')

def getDataFrame(files):
    _getdf = lambda f: pd.read_csv(f, index_col=0)
    frames = [_getdf(f) for f in files]
    return pd.concat(frames)

if __name__ == '__main__':
    args = parser.parse_args()
    df = getDataFrame(args.files)
