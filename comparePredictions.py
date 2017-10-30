import argparse
import sys
import collections
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from ast import literal_eval

parser = argparse.ArgumentParser(description='Compare predictions from HMM model')
parser.add_argument('-i', '--input', help='input_files', nargs='+', required=True)
parser.add_argument('-n', '--names', help='optional names for the input files', nargs='+', default=None)
parser.add_argument('-o', '--output-folder', help='where to save the plots', default='.')
args = parser.parse_args()

if args.names and len(args.input) != len(args.names):
    sys.exit("Number of names and files do not match!")


class PredictionContainer(object):

    __slots__ = ['name', 'seqs', 'charges', 'scores']

    def __init__(self, name, tuples):
        self.name = name
        self.seqs = set()

        seqs, zs, scores = zip(*tuples)
        self.seqs.update(seqs)
        self.charges = [int(z) for z in zs]
        self.scores = [float(s) for s in scores]


def _getAACounts(seq):
    c = collections.Counter(seq)
    return {aa: c[aa] for aa in seq}

#TODO Annontate the plots
def plotLengthDistribution(data_series, file):

    dfdata = {name: [len(s) for s in pc.seqs] for name, pc in data_series.items()}
    df = pd.DataFrame.from_dict(dfdata, orient='index')
    df = df.transpose()

    min_l = df.min(numeric_only=True).min()
    max_l = df.max(numeric_only=True).max()
    df.plot.hist(bins=np.arange(min_l, max_l + 1, 1), alpha=0.5)

    plt.savefig(file, bbox_inches='tight', format='pdf')
    plt.close()


def plotChargeDistribution(data_series, file):
    dfdata = {name: pc.charges for name, pc in data_series.items()}
    df = pd.DataFrame.from_dict(dfdata, orient='index')
    df = df.transpose()

    fig, axes = plt.subplots(nrows=2, ncols=1)
    fig.suptitle("Distribution of charge states")
    df.plot.kde(ax=axes[0])
    axes[1].hist(df.values, histtype='bar')
    plt.savefig(file, bbox_inches='tight', format='pdf')
    plt.close()

def plotScoreDistribution(data_series, file):
    dfdata = {name: pc.scores for name, pc in data_series.items()}
    df = pd.DataFrame.from_dict(dfdata, orient='index')
    df = df.transpose()

    fig, axes = plt.subplots(nrows=2, ncols=1)
    df.plot.hist(ax=axes[0], bins=100)
    #df.plot.kde(ax=axes[0], xlim=(-50, 400))
    df.plot.box(ax=axes[1], vert=False)
    fig.suptitle("Distribution of Andromeda scores")
    plt.savefig(file, bbox_inches='tight', format='pdf')
    plt.close()


def plotChargeLengthBivariate(data_series, file):
    for name, ds in data_series.items():
        lens = [len(s) for s in ds.seqs]
        zs = ds.charges
        dfdata = list(zip(lens, zs))
        data = np.array(dfdata, dtype=np.dtype('int,int'))
        df = pd.DataFrame(data, columns=["length", "z"])
        ax = sns.jointplot(x="length", y="z", data=df)
        plt.title(f'Hexbin density plot for {name}')
        plt.savefig(file, bbox_inches='tight', format='pdf')
        plt.close()

def plotAADistribution(data_series, file):

    def getDF(series):
        d = {i: _getAACounts(s) for i, s in enumerate(series.seqs)}
        df = pd.DataFrame.from_dict(d, orient='index')
        df.sort_index(axis=1, inplace=True)
        return df

    dataframes = {name: getDF(pc) for name, pc in data_series.items()}
    nplots = len(dataframes)

    fig, axes = plt.subplots(nrows=nplots, ncols=1)
    fig.suptitle("Distribution of AA residues")
    for i, d in enumerate(dataframes):
        df = dataframes[d]
        df.plot.box(ax=axes[i], logy=True)

    plt.savefig(file, bbox_inches='tight', format='pdf')
    plt.close()

def plotAAs(data_series, file):

    d = {name: {i: _getAACounts(s) for i, s in enumerate(pc.seqs)} for name, pc in data_series.items()}
    df = pd.concat(map(pd.DataFrame, d.values()), keys=d.keys()).stack().unstack(0)
    df.index.names = ['aa', None]
    df.reset_index(level=['aa'], inplace=True)
    dd = pd.melt(df, id_vars=['aa'], value_vars=['high', 'low'], var_name='corr')
    g = sns.boxplot(x='aa', y='value', data=dd, hue='corr')
    plt.title(f'Amino acid residue distribution')
    plt.savefig(file, bbox_inches='tight', format='pdf')
    plt.close()

def plotMPT(data_series, file):

    def getMPClasses(seqs, zs):
        classes = collections.defaultdict(int)
        for i, seq in enumerate(seqs):
            counts = collections.defaultdict(int, _getAACounts(seq))
            if zs[i] > counts['R'] + counts['K'] + counts['H']:
                classes['Mobile'] += 1
            elif counts['K'] + counts['H'] > zs[i] - counts['R'] > 0:
                classes['Partial'] += 1
            elif counts['R'] > zs[i]:
                classes['Nonmobile'] += 1
            else:
                AssertionError('Should not happen!')

        return classes

    d = {name: getMPClasses(pc.seqs, pc.charges) for name, pc in data_series.items()}
    df = pd.DataFrame.from_dict(d)
    df = df.reindex(["Mobile", "Partial", "Nonmobile"])
    df = df.transpose()
    df = df.div(df.sum(axis=1), axis=0)
    df.plot.barh(stacked=True)
    plt.ylabel("Number of peptides")
    plt.title(f'Distribution of predictions into groups according to MPT')
    plt.savefig(file, bbox_inches='tight', format='pdf')


if __name__ == '__main__':

    data = {}
    for i, somefile in enumerate(args.input):
        with open(somefile, 'r') as f:
            entries = [literal_eval(line.strip()) for line in f]
            index = args.names[i] if args.names else f"File{i}"
            data[index] = PredictionContainer(index, entries)

    with PdfPages("figures.pdf") as pdf:
        plotLengthDistribution(data, pdf)
        plotChargeDistribution(data, pdf)
        # plotChargeLengthBivariate(data, pdf)
        plotAAs(data, pdf)
        plotScoreDistribution(data, pdf)
        plotMPT(data, pdf)


