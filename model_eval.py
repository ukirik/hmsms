"""
This module should hold the code for evaluation of the model
which will happen in a number of different ways
1. Model vs Mock
2. Model vs Experimental data: consider x-validation to estimate training data bias
    a. part of in house data, not used for training
    b. external data
3. Comparison with other models
"""
import pickle
import itertools
import argparse
import common_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Evaluate HMM')
parser.add_argument('--model', help='path to model to use')
parser.add_argument('--mock', help='path to corresponding mock model')
parser.add_argument('--test_files', nargs='+', help='files to check corr on ')
parser.add_argument('-o', '--order', help='order, default = 2', default=2, type=int)
parser.add_argument('-i', '--input', help='input_file', nargs='+')
parser.add_argument('-x', '--nfold', help='nbr of slices for X validation', default=5, type=int)
# parser.add_argument('--out', help='name of directory where any resultant files will be stored', default='.')
# parser.add_argument('-n', '--name', help='name of the model')
# parser.add_argument('-p', '--pickle', help='print/pickle location', default='hmm_models')
parser.add_argument('-t', '--nthreads', help='number of threads to use', default=1, type=int)
# parser.add_argument('-m', '--use_model', help='model to use, instead of training a new one')
# parser.add_argument('-f', '--test_files', nargs='+', help='files to check corr on ')
# parser.add_argument('-g', '--graph', default=False, action='store_true')
# parser.add_argument('-d', '--debug', default=False, action='store_true')

args = parser.parse_args()


def vs_mock(model_path, mock_path, test_files):

    model_pickle = model_path
    mock_pickle = mock_path
    file_gen = common_utils.yield_open(filenames=test_files)
    testdata = itertools.chain.from_iterable(file_gen)
    dd = {}

    with open(model_pickle, 'rb') as picklefile, open(mock_pickle, 'rb') as mockfile:

        model = pickle.load(picklefile)
        mockmodel = pickle.load(mockfile)

        np.seterr(invalid='ignore')

        model = model.finalizeModel(alpha=250)
        mockmodel = mockmodel.finalizeModel(alpha=250)

        bin_z = lambda x: x if int(x) < 4 else '4+'
        bin_l = lambda s: (len(s) - 7) // 5 if (len(s) - 7) // 5 < 4 else 4

        def getbin(l):
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

        counter = 0
        nlines = None
        for line in itertools.islice(testdata, nlines):
            try:
                tokens = line.rstrip('\r\n').split('\t')
                z, seq, score, y_ions, y_ints, b_ions, b_ints, y_frac = tokens
                if len(y_ints) == 0:
                    continue
                if int(z) == 1:
                    continue

                y_ints = [float(i) for i in y_ints.split(' ')]
                y_ions = y_ions.split(' ')
                # b_ints = [float(i) for i in b_ints.split(' ')]
                # b_ions = b_ions.split(' ')
                ions, probs = model.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)
                mockions, mockprobs = mockmodel.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)

                d = {'exp': {i: ii for i, ii in zip(y_ions, y_ints)},
                     'model': {ion: p for ion, p in zip(ions, probs)},
                     'mock': {i: p for i, p in zip(mockions, mockprobs)}
                     }

                df = pd.DataFrame.from_dict(d)
                #df.fillna(0, inplace=True) # TODO: this might be misleading
                corrs = df.corr(method='pearson')
                dd[counter] = {'seq': seq,
                               'charge': bin_z(z),
                               'pep length': getbin(len(seq)),
                               'model': 'mock',
                               'correlation': corrs.iat[0, 1]
                               }

                dd[counter + 1] = {'seq': seq,
                                   'charge': bin_z(z),
                                   'pep length': getbin(len(seq)),
                                   'model': 'fwd',
                                   'correlation': corrs.iat[0, 2]
                                   }

                counter += 2

            except ValueError as e:
                print("Unexpected number of tokens found on line!")
                e.args += (line,)
                raise

        df = pd.DataFrame.from_dict(dd, orient='index')
        import seaborn as sns
        sns.set(style="ticks")

        ax = sns.factorplot(x="pep length", y="correlation", hue="model", col="charge", data=df,
                            order=['07-11', '12-16', '17-21', '22-26', '27+'],
                            col_order=['2', '3', '4+'], kind="box", legend=False)

        ax.despine(offset=10, trim=True)
        plt.legend(loc='best')
        ax.savefig('vs_mock.pdf')


def _pool_worker(filename, doshuffle):
    from hmm_model import FragmentHMM
    print(f"Processing file: {filename}")
    with open(filename, 'r') as f:
        partial = FragmentHMM(order=args.order, indata=f, shuffle=doshuffle)
        return partial


def x_validatation(files, nslices, allcombos=False):
    """
    :param files: list of input files
    :param nslices: an integer denoting the number of slices to do x-validation on
    :param allcombos: logical to denote if all possible combinations should be considered (takes long time)
    :return: list of models used to do the predictions.
    """
    assert isinstance(files, list)

    if isinstance(nslices, int):
        assert len(files) > nslices

    from multiprocessing import Pool as ThreadPool
    from functools import partial
    import copy, sys

    def partition_list(l, predicate, randomize=False):
        if randomize:
            np.random.shuffle(l)

        train, test = [], []
        for index, item in enumerate(l):
            (test if predicate(index) else train).append(item)

        return train, test

    # Pandas visual options
    pd.set_option('precision', 4)
    pd.set_option('display.width', 320)

    with ThreadPool(args.nthreads) as pool:
        models = []
        #slices = [files[n: n+nslices] for n in range(0, len(files), nslices)]

        summary_df = None
        n = int(len(files)/nslices)
        combs = itertools.combinations(range(len(files)), n)
        loop_over = combs if allcombos else range(nslices)
        for i, e in enumerate(loop_over):
            model = None

            if allcombos:
                input_files, validation_files = partition_list(files, lambda x: x in e)
            else:
                input_files, validation_files = partition_list(files, lambda x: True if i*n <= x < (i+1)*n else False)

            # print(f"Training on {input_files}, testing on {validation_files}")

            partial_models = pool.imap_unordered(partial(_pool_worker, doshuffle=False), input_files)
            for m in partial_models:
                model = copy.deepcopy(m) if model is None else model + m

            if not allcombos:
                models.append(model)

            df = _validate_model(model, validation_files)
            summary = df.groupby(['charge', 'pep length']).mean(numeric_only=True)
            summary_df = summary if summary_df is None else pd.concat((summary_df, summary), axis=1)
            # print(summary)

    #print(summary_df)
    summary_df.to_csv(sys.stdout)
    return models


def _validate_model(m, testfiles, alpha=200):
    file_gen = common_utils.yield_open(filenames=testfiles)
    testdata = itertools.chain.from_iterable(file_gen)
    dd = {}
    np.seterr(invalid='ignore')
    model = m.finalizeModel(alpha)

    bin_z = lambda x: x if int(x) < 4 else '4+'
    bin_l = lambda s: (len(s) - 7) // 5 if (len(s) - 7) // 5 < 4 else 4

    def getbin(l):
        if 7 <= l < 12:
            return '07-11'
        elif 12 <= l < 17:
            return '12-16'
        elif 17 <= l < 22:
            return '17-21'
        elif 22 <= l < 27:
            return '22-26'
        elif 27 <= l:
            return '27+'

    counter = 0
    nlines = None
    for line in itertools.islice(testdata, nlines):
        try:
            tokens = line.rstrip('\r\n').split('\t')
            z, seq, score, y_ions, y_ints, b_ions, b_ints, y_frac = tokens
            if len(y_ints) == 0:
                continue
            if int(z) == 1:
                continue

            y_ints = [float(i) for i in y_ints.split(' ')]
            y_ions = y_ions.split(' ')
            # b_ints = [float(i) for i in b_ints.split(' ')]
            # b_ions = b_ions.split(' ')
            ions, probs = model.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)

            d = {'exp': {i: ii for i, ii in zip(y_ions, y_ints)},
                 'model': {ion: p for ion, p in zip(ions, probs)},
                 }

            df = pd.DataFrame.from_dict(d)
            #df.fillna(0, inplace=True) # TODO: this might be misleading
            corrs = df.corr(method='pearson')
            dd[counter] = {'seq': seq,
                           'charge': bin_z(z),
                           'pep length': getbin(len(seq)),
                           'correlation': corrs.iat[0, 1]
                           }

            counter += 1

        except ValueError as e:
            print("Unexpected number of tokens found on line!")
            e.args += (line,)
            raise

    df = pd.DataFrame.from_dict(dd, orient='index')

    return df


if __name__ == '__main__':
    #vs_mock(args.model, args.mock, args.test_files)
    xval_models = x_validatation(args.input, args.nfold, False)
