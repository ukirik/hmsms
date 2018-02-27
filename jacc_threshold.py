import pickle
import argparse
import common_utils
import itertools
import tqdm
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()


def get_top_frags(zipped, threshold):
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
        return charge
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


parser = argparse.ArgumentParser(description='Evaluates the difference between model and mock for optimal Jacc similarity')
parser.add_argument('--model', help='path to model to use')
parser.add_argument('--mock', help='path to corresponding mock model')
parser.add_argument('--test_files', nargs='+', help='files to check corr on ')
parser.add_argument('-n', '--max_spectra', type=int, default=None, help='number of predictions to run')

if __name__ == '__main__':
    args = parser.parse_args()
    model_pickle = args.model
    mock_pickle = args.mock
    file_gen = common_utils.yield_open(filenames=args.test_files)
    testdata = itertools.chain.from_iterable(file_gen)
    thresholds = np.linspace(0.25, 0.95, n=15)

    with open(model_pickle, 'rb') as picklefile, open(mock_pickle, 'rb') as mockfile:
        model = pickle.load(picklefile)
        mockmodel = pickle.load(mockfile)

        np.seterr(invalid='ignore')

        model = model.finalizeModel(alpha=args.alpha)
        mockmodel = mockmodel.finalizeModel(alpha=args.alpha)
        nlines = args.max_spectra if args.max_spectra > 0 else None
        dd = {}
        counter = 0
        for line in tqdm.tqdm(itertools.islice(testdata, nlines), total=nlines):
            try:
                tokens = line.rstrip('\r\n').split('\t')
                z, seq, score, y_ions, y_ints, b_ions, b_ints, y_frac = tokens
                if len(y_ints) < 3:
                    continue
                if int(z) == 1:
                    continue
                if len(re.findall('[KR]', seq)) > 2:
                    # print(seq)
                    continue

                y_ints = [float(i) for i in y_ints.split(' ')]
                y_ions = y_ions.split(' ')

                ii, it = model.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)
                sims = [jacc(zip(y_ions, y_ints), zip(ii, it), threshold=t) for t in thresholds]

                temp = dict()
                temp['seq'] = seq
                temp['charge'] = _bin_z(z)
                temp['peplen'] = _getbin(len(seq))
                temp['model'] = 'real'

                for i, t in enumerate(thresholds):
                    temp[f'j{t:.2f}'] = sims[i]

                dd[counter] = temp
                counter += 1

                ii, it = mockmodel.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)
                mocks = [jacc(zip(y_ions, y_ints), zip(ii, it), threshold=t) for t in thresholds]

                temp = dict()
                temp['seq'] = seq
                temp['charge'] = _bin_z(z)
                temp['peplen'] = _getbin(len(seq))
                temp['model'] = 'mock'

                for i, t in enumerate(thresholds):
                    temp[f'j{t:.2f}'] = mocks[i]

                dd[counter] = temp
                counter += 1
            except ValueError as e:
                print("Unexpected number of tokens found on line!")
                e.args += (line,)
                raise

        df = pd.DataFrame.from_dict(dd, orient='index')
        df_long = pd.melt(df, id_vars=['seq', 'charge', 'peplen', 'model'], var_name="Similarity")
        # print(df_long.head())

        import seaborn as sns
        sns.factorplot(x="Similarity", y="value", hue="model", data=df_long, size=12, ci='sd')