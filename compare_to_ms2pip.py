import argparse
import collections
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff() #http://matplotlib.org/faq/usage_faq.html (interactive mode)
from itertools import islice
import seaborn as sns
from enum import Enum

parser = argparse.ArgumentParser(description='Parse and compare MS2PIP results')
parser.add_argument('-f', '--files', nargs='+', required=True, help='files containing MS2PIP predictions')
parser.add_argument('-m', '--model', required=True, help='model to use for comparing predictions')
parser.add_argument('-a', '--alpha', default=128, help='alpha used to finalize model')
parser.add_argument('-t', '--test_file', required=True, help='tsv file containing peptides to predict. Important that the same peptides are used for both models')


def getDataFrame(files):
    _getdf = lambda f: pd.read_csv(f, index_col=0)
    frames = [_getdf(f) for f in files]
    return pd.concat(frames)


def get_ms2pip_pred(df, key):
    mask = (df.spec_id.values == key) & (df.ion.values == 'y')
    temp = df[mask]

    ysum = temp['prediction'].sum()
    ions = []
    pred = []
    for tup in temp.itertuples():
        ions.append(str(tup.ion) + str(tup.ionnumber))
        pred.append(tup.prediction / ysum)

    return ions, pred


def get_top_frags(zipped, threshold):
    assert threshold > 0.0
    import operator
    ig = operator.itemgetter(1)
    temp = sorted(zipped, key=ig, reverse=True)

    frags = []
    cumm = 0.0
    for ion, ion_int in temp:
        if cumm > threshold:
            break
        frags.append(ion)
        cumm += ion_int

    return frags


def jacc(x, y, threshold=0.9, verbose=False):
    assert isinstance(x, zip)
    assert isinstance(y, zip)

    xfrags = get_top_frags(x, threshold)
    yfrags = get_top_frags(y, threshold)

    assert len(xfrags) > 0
    assert len(yfrags) > 0

    if verbose:
        print(f'experimental: {xfrags}')
        print(f'predicted: {yfrags}')

    intersect = len(set.intersection(set(xfrags), set(yfrags)))
    union = len(set.union(set(xfrags), set(yfrags)))
    return intersect / float(union)


def _bin_z(charge):
    if int(charge) < 4:
        return str(charge)
    else:
        return '4+'


def _getbin(l):
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


def _getAACounts(seq):
    c = collections.Counter(seq)
    return {aa: c[aa] for aa in seq}


class MPT(Enum):
    MOBILE = 1
    PARTIAL = 2
    NONMOBILE = 3


def _getMPClass(seq, z):
    counts = collections.defaultdict(int, _getAACounts(seq))
    if isinstance(z, str):
        z = int(z)

    if z > counts['R'] + counts['K'] + counts['H']:
        return MPT.MOBILE
    elif counts['K'] + counts['H'] >= z - counts['R'] > 0:
        return MPT.PARTIAL
    elif counts['R'] >= z:
        return MPT.NONMOBILE
    else:
        raise AssertionError(f"Should not happen! z={z}, h={counts['H']}, k={counts['K']}, r={counts['R']}")


if __name__ == '__main__':
    args = parser.parse_args()
    ms2pip_df = getDataFrame(args.files)

    picklefile = open(args.model, 'rb')
    model = pickle.load(picklefile)
    model = model.finalizeModel(alpha=128)

    dd = {}
    counter = 0
    rej_hmsms = 0
    rej_ms2pip = 0

    with open(args.test_file, 'r') as f:
        for line in islice(f, 40000):
            try:
                tokens = line.rstrip('\r\n').split('\t')
                z, seq, score, y_ions, y_ints, b_ions, b_ints, y_frac = tokens
                if len(y_ions) == 0 or len(y_ints) == 0:
                    continue

                y_ints = [float(i) for i in y_ints.split(' ')]
                y_ions = y_ions.split(' ')
                z = int(z)
            except ValueError as e:
                print("Unexpected number of tokens found on line!")
                e.args += (line,)
                raise

            if len(y_ints) < 3:
                continue
            if z == 1:
                continue
            if len(re.findall('[KR]', seq)) > 2:
                # print(seq)
                continue

            #             print('='*40)
            #             print(f'Processing predictions for {seq} +{z}')

            ions, probs = model.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)
            ions2, preds = get_ms2pip_pred(ms2pip_df, f'{seq}_{z}')

            # Pearsons corr
            d = {'exp': {int(i[1:]): ii for i, ii in zip(y_ions, y_ints)},
                 'hmsms': {int(i[1:]): p for i, p in zip(ions, probs)},
                 'ms2pip': {int(i[1:]): p for i, p in zip(ions2, preds)}
                 }

            df = pd.DataFrame.from_dict(d)
            corr = df.corr(method='pearson')

            #             print(df)
            #             print()
            #             print(corr)

            r_hmsms = corr.iat[0, 1]
            r_ms2pip = corr.iat[0, 2]

            # Jaccard's similarity
            try:
                j_hmsms = jacc(zip(y_ions, y_ints), zip(ions, probs), threshold=0.7)
            except AssertionError as ae:
                print(seq, ions, probs)
                rej_hmsms += 1
                j_hmsms = np.nan

            try:
                j_ms2pip = jacc(zip(y_ions, y_ints), zip(ions2, preds), threshold=0.7)
            except AssertionError as ae:
                print(seq, ions2, preds)
                rej_ms2pip += 1
                j_ms2pip = np.nan

            #             print()
            #             print(f'Jaccard: HMSMS={j_hmsms}, MS2PIP={j_ms2pip}')
            #             input("enter to continue...")

            dd[counter] = {
                'seq': seq,
                'charge': _bin_z(z),
                'peplen': _getbin(len(seq)),
                'mpt_class': _getMPClass(seq, z),
                'corr': r_hmsms,
                'jacc': j_hmsms,
                'model': 'HMSMS'
            }
            counter += 1

            dd[counter] = {
                'seq': seq,
                'charge': _bin_z(z),
                'peplen': _getbin(len(seq)),
                'mpt_class': _getMPClass(seq, z),
                'corr': r_ms2pip,
                'jacc': j_ms2pip,
                'model': 'MS2PIP'
            }
            counter += 1

    df = pd.DataFrame.from_dict(dd, orient='index')
    # df_long = pd.melt(df, id_vars=['seq', 'charge', 'peplen', 'model', 'mpt_class'], var_name="threshold")
    print(df)
    print(f'Rejected peptides: HMSMS={rej_hmsms} MS2PIP={rej_ms2pip}')

    ax = sns.factorplot(data=df, x='peplen', y='jacc', linewidth=2, hue="model", col="charge",
                        order=['7-11', '12-16', '17-21', '22-26', '27+'],
                        col_order=['2', '3', '4+'], kind="box", legend_out=True)

    ax.despine(offset=10, trim=True)
    # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_axis_labels(x_var='Peptide length bins', y_var='Jaccard similarity ratio')
    plt.subplots_adjust(top=0.85)
    ax.fig.suptitle('Jaccard similarity performance at 60% threshold')  # can also get the figure from plt.gcf()
    ax.savefig('jacc_ms2pip_z.pdf')

    ax = sns.factorplot(data=df, x='peplen', y='corr', linewidth=2, hue="model", col="charge",
                        order=['7-11', '12-16', '17-21', '22-26', '27+'],
                        col_order=['2', '3', '4+'], kind="box", legend_out=True)

    ax.despine(offset=10, trim=True)
    # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_axis_labels(x_var='Peptide length bins', y_var='Correlation')
    plt.subplots_adjust(top=0.85)
    ax.fig.suptitle("Pearson's correlation performance")  # can also get the figure from plt.gcf()
    ax.savefig('corr_ms2pip_z.pdf')

    ax = sns.factorplot(data=df, x='peplen', y='jacc', linewidth=2, hue="model", col="mpt_class",
                        order=['7-11', '12-16', '17-21', '22-26', '27+'],
                        kind="box", legend_out=True)

    ax.despine(offset=10, trim=True)
    # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_axis_labels(x_var='Peptide length bins', y_var='Jaccard similarity ratio')
    plt.subplots_adjust(top=0.85)
    ax.fig.suptitle('Jaccard similarity performance at 60% threshold')  # can also get the figure from plt.gcf()
    ax.savefig('jacc_ms2pip_mpt.pdf')

    ax = sns.factorplot(data=df, x='peplen', y='jacc', linewidth=2, hue="model", col="mpt_class",
                        order=['7-11', '12-16', '17-21', '22-26', '27+'],
                        kind="box", legend_out=True)

    ax.despine(offset=10, trim=True)
    # plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
    ax.set_axis_labels(x_var='Peptide length bins', y_var='Correlation')
    plt.subplots_adjust(top=0.85)
    ax.fig.suptitle("Pearson's correlation performance")  # can also get the figure from plt.gcf()
    ax.savefig('corr_ms2pip_mpt.pdf')