import argparse
import sys
import pickle
import common_utils
import ms_utils
import itertools
import tqdm
import collections
import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from hmm_model import FragmentHMM


parser = argparse.ArgumentParser(description='Compare predictions from HMM model')
parser.add_argument('-m', '--models', help='models of different order to compare', nargs='+', required=True)
parser.add_argument('-n', '--names', help='optional names for the input files', nargs='+', default=None)
parser.add_argument('-t', '--test-files', help='optional names for the input files', nargs='+', required=True)
parser.add_argument('-o', '--outfile', help='name of the output CSV file', default='comparison.csv')
args = parser.parse_args()

if args.names and len(args.models) != len(args.names):
    sys.exit("Number of names and files do not match!")

if len(args.models) == 1:
    sys.exit("Nothing to compare, only one model given!")

def getIonProbs(model, z, seq):
    return model.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)

def _zbin(x):
    return x if int(x) < 4 else '4+'

def _lbin(l):
    if 7 <= l < 12:
        return '7-11'
    elif 12 <= l < 17:
        return '12-16'
    elif 17 <= l < 22:
        return '17-21'
    elif 22 <= l < 27:
        return '22-26'
    elif 27 <= l:
        return '27+'

if __name__ == '__main__':

    data = {}
    models = []
    names = []
    for i, picklefile in enumerate(args.models):
        m = pickle.load(open(picklefile, 'rb'))
        models.append(m.finalizeModel(alpha=256))
        index = args.names[i] if args.names else f"Model_{i}"
        names.append(index)

    file_gen = common_utils.yield_open(filenames=args.test_files)
    testdata = itertools.chain.from_iterable(file_gen)
    results = []
    nlines = None
    for line in itertools.islice(testdata, nlines):
        try:
            tokens = line.rstrip('\r\n').split('\t')
            z, seq, score, y_ions, y_ints, b_ions, b_ints, y_frac = tokens
            if len(y_ints) < 3:
                continue

            y_ints = [float(i) for i in y_ints.split(' ')]
            y_ions = y_ions.split(' ')

            preds = [getIonProbs(m, int(z), seq) for m in models]
            d = dict()
            d['exp'] = {i: ii for i, ii in zip(y_ions, y_ints)}
            for i,ions, probs in enumerate(preds):
                d[names[i]] = zip(ions, probs)

            df = pd.DataFrame.from_dict(d)
            p = df.corr(method='pearson')

            res = {
                'seq': seq,
                'charge': z,
                'z_bin': _zbin(z),
                'peplen': len(seq),
                'l_bin': _lbin(len(seq)),
                'mpt_class': ms_utils.getMPClass(seq, z),
                # 'exp_ints': df['exp'].to_json(),
                # 'pred_ints': df['model'].to_json(),
                'y_frac': float(y_frac),
                'a_score':float(score)
            }

            for i,n in enumerate(names):
                res[n] = p.loc[n][0]

            results.append(res)

        except ValueError as e:
            print("Unexpected number of tokens found on line, skipping this entry!")
            e.args += (line,)
            continue

    d = pd.DataFrame(results)
    # d = d[['seq', 'charge', 'z_bin', 'peplen', 'l_bin', 'y_frac', 'a_score', 'mpt_class', 'pearsons', 'pearsonz', 'spearman', 'exp_ints', 'pred_ints']]
    d.reindex(columns=['seq', 'charge', 'z_bin', 'peplen', 'l_bin', 'y_frac', 'a_score', 'mpt_class'] + names)
    d.to_csv(args.outfile)
