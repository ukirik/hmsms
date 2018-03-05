import pickle
import argparse
import common_utils
import itertools
import tqdm
import re
import collections
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import seaborn as sns


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


def baseline():

    def _validateSpectraPair(spectra, minlength):
        prev = None
        assert len(spectra) == 2

        for s in spectra:
            try:
                score, yi, yw, bi, bw, yf = s
                if len(yi) < minlength:
                    return False
                if prev == s:
                    return False
            except Exception as e:
                print(f"Unexpected number of tokens found!")
                e.args += (str(s),)
                raise

            prev = s
        return True

    def _getSpectraPair(seq, data):
        valid_spectra = data
        ntry = 1
        minlen = 5
        spectra = random.sample(valid_spectra, 2)
        try:
            while not _validateSpectraPair(spectra, minlen):
                ntry += 1
                if ntry > 5:
                    valid_spectra = [d for d in data if len(d[1]) > minlen]
                    if len(valid_spectra) < 2:
                        print(f"Not enough valid spectra for {seq}")
                        return None

                spectra = random.sample(valid_spectra, 2)

        except Exception as e:
            print(f"Exception occurred during processing {seq}")
            e.args += (str(data),)
            raise

        return spectra

    def _getParser(infile):
        print('Processing the spectra...')
        from MQParser import MQParser
        with open(infile, 'r') as f:
            parser = MQParser()
            reader = pd.read_csv(f, delimiter='\t', iterator=True, chunksize=100000)

            for index, chunk in enumerate(reader):
                df = pd.DataFrame(chunk)
                if not index:
                    # Creates valid column names for pandas
                    colnames = [col.strip().replace(" ", "_") for col in df.columns]

                df.columns = colnames
                loop = parser.processChunk(df, True) # Input assumed to be sorted

                if not loop:  # check if FDR threshold is met
                    break

            print("Finished parsing spectra")
            print("FDR={}, pos={}, neg={}".format(parser.fdr, parser.pos, parser.neg))
            t = args.spectra_threshold
            filtered_spectra = [key for key in parser.getKeys() if len(list(parser.getDataAsTuple(key))) > t]
            print(f'{len(filtered_spectra)} peptides have more than {t} spectra')
            return parser.get_subset(filtered_spectra)


    parser = _getParser(args.train_data)
    dd = collections.defaultdict(list)
    counter = 0

    baseline_spectra = parser.psms.keys()
    for key in tqdm.tqdm(baseline_spectra):
        data = list(parser.getDataAsTuple(key))
        seq = parser.psms[key].seq
        z = key[1]

        spectra = _getSpectraPair(seq, data)
        if spectra is None:
            continue

        score, yi_1, yw_1, *rest = spectra[0]
        score, yi_2, yw_2, *rest = spectra[1]

        base_j = [jacc(zip(yi_1, yw_1), zip(yi_2, yw_2), threshold=t) for t in thresholds]
        temp = dict()
        temp['seq'] = seq
        temp['charge'] = _bin_z(z)
        temp['peplen'] = _getbin(len(seq))
        temp['model'] = 'baseline'

        for i, t in enumerate(thresholds):
            temp[f'{t:.2f}'] = base_j[i]

        dd[counter] = temp
        counter += 1

    print(f'finished parsing baseline data...')
    return dd


parser = argparse.ArgumentParser(description='Evaluates the difference between model and mock for optimal Jacc similarity')
parser.add_argument('--model', help='path to model to use')
parser.add_argument('--mock', help='path to corresponding mock model')
parser.add_argument('--train_data', help='files containing data used for training, for baseline estimation ')
parser.add_argument('--spectra_threshold', default=100, type=int, help='min nbr of spectra to consider')
parser.add_argument('--test_files', nargs='+', help='files to check corr on ')
parser.add_argument('-n', '--max_spectra', type=int, default=-1, help='number of predictions to run')

if __name__ == '__main__':
    args = parser.parse_args()
    model_pickle = args.model
    mock_pickle = args.mock
    file_gen = common_utils.yield_open(filenames=args.test_files)
    testdata = itertools.chain.from_iterable(file_gen)
    thresholds = np.linspace(0.25, 0.95, num=15)

    with open(model_pickle, 'rb') as picklefile, open(mock_pickle, 'rb') as mockfile:
        model = pickle.load(picklefile)
        mockmodel = pickle.load(mockfile)

        np.seterr(invalid='ignore')

        model = model.finalizeModel(alpha=450)
        mockmodel = mockmodel.finalizeModel(alpha=450)
        nlines = args.max_spectra if args.max_spectra > 0 else None
        dd = baseline()
        counter = len(dd.keys())
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
                    temp[f'{t:.2f}'] = sims[i]

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
                    temp[f'{t:.2f}'] = mocks[i]

                dd[counter] = temp
                counter += 1
            except ValueError as e:
                print("Unexpected number of tokens found on line!")
                e.args += (line,)
                raise

        df = pd.DataFrame.from_dict(dd, orient='index')
        df_long = pd.melt(df, id_vars=['seq', 'charge', 'peplen', 'model'], var_name="threshold")
        # print(df_long.head())

        plt.ioff()
        from numpy import median
        ax = sns.factorplot(x="threshold", y="value", hue="model", data=df_long, size=12, ci=99, legend=False, estimator=median)
        ax.despine(offset=10)
        plt.legend(loc='best')
        ax.savefig('jacc_median_ci95.pdf')