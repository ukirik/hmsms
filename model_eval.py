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
import collections
import re
import common_utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.ioff()

import random
import seaborn as sns

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


def _pool_worker(filename, doshuffle):
    from hmm_model import FragmentHMM
    print(f"Processing file: {filename}")
    with open(filename, 'r') as f:
        partial = FragmentHMM(order=args.order, indata=f, single_precision=True, shuffle=doshuffle)
        return partial


#def vs_mock(model_path, mock_path, test_files):
def vs_mock(args):

    model_pickle = args.model
    mock_pickle = args.mock
    file_gen = common_utils.yield_open(filenames=args.test_files)
    testdata = itertools.chain.from_iterable(file_gen)
    dd = {}

    with open(model_pickle, 'rb') as picklefile, open(mock_pickle, 'rb') as mockfile:

        model = pickle.load(picklefile)
        mockmodel = pickle.load(mockfile)

        np.seterr(invalid='ignore')

        model = model.finalizeModel(alpha=args.alpha)
        mockmodel = mockmodel.finalizeModel(alpha=args.alpha)

        bin_z = lambda x: x if int(x) < 4 else '4+'
        bin_l = lambda s: (len(s) - 7) // 5 if (len(s) - 7) // 5 < 4 else 4

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
                               'pep length': _getbin(len(seq)),
                               'model': 'mock',
                               'correlation': corrs.iat[0, 1]
                               }

                dd[counter + 1] = {'seq': seq,
                                   'charge': bin_z(z),
                                   'pep length': _getbin(len(seq)),
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


#def x_validation(files, nslices, allcombos=False):
def x_validation(args):
    """
    :param files: list of input files
    :param nslices: an integer denoting the number of slices to do x-validation on
    :param allcombos: logical to denote if all possible combinations should be considered (takes long time)
    :return: list of models used to do the predictions.
    """
    files = args.input
    nslices = args.nfold
    allcombos = args.allcombos
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
            logger.info(f'Starting iteration {i + 1}')
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
            logger.info(f'--Validating...')
            df = _validate_model(model, validation_files)
            summary = df.groupby(['charge', 'pep length']).mean(numeric_only=True)
            summary_df = summary if summary_df is None else pd.concat((summary_df, summary), axis=1)
            # print(summary)

    #print(summary_df)
    summary_df.to_csv(sys.stdout)
    return models


def _validate_model(m, testfiles):
    file_gen = common_utils.yield_open(filenames=testfiles)
    testdata = itertools.chain.from_iterable(file_gen)
    dd = {}
    np.seterr(invalid='ignore')
    model = m.finalizeModel(args.alpha)

    bin_z = lambda x: x if int(x) < 4 else '4+'
    bin_l = lambda s: (len(s) - 7) // 5 if (len(s) - 7) // 5 < 4 else 4

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
                           'pep length': _getbin(len(seq)),
                           'correlation': corrs.iat[0, 1]
                           }

            counter += 1

        except ValueError as e:
            print("Unexpected number of tokens found on line!")
            e.args += (line,)
            raise

    df = pd.DataFrame.from_dict(dd, orient='index')

    return df


def baseline(args):

    def _validateSpectra(spectra):
        prev = None
        assert len(spectra) == 2

        for s in spectra:
            try:
                score, yi, yw, bi, bw, yf = s
                if len(yi) < 3:
                    return False
                if prev == s:
                    return False
            except Exception as e:
                print(f"Unexpected number of tokens found on line!")
                e.args += (str(s),)
                raise

            prev = s
        return True

    def _getSpectraPair(seq, data):
        valid_spectra = data
        ntry = 1
        spectra = random.sample(valid_spectra, 2)
        try:
            while not _validateSpectra(spectra):
                ntry += 1
                if ntry > 5:
                    valid_spectra = [d for d in data if len(d[1]) > 3]
                    if len(valid_spectra) < 2:
                        print(f"Not enough spectra for {seq}")
                        return None

                spectra = random.sample(valid_spectra, 2)
        except Exception as e:
            import time
            print(f"Exception occured during processing {seq}")
            time.sleep(1)
            input("Enter to continue...")

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
            return parser

    import tqdm

    corrs = collections.defaultdict(list)
    parser = _getParser(args.data)

    t = args.spectra_threshold
    baseline_spectra = [key for key in parser.getKeys() if len(list(parser.getDataAsTuple(key))) > t]
    print(f'{len(baseline_spectra)} peptides have more than {t} spectra')

    nlines = min(args.max_spectra, len(baseline_spectra)) if args.max_spectra > 0 else None
    for key in tqdm.tqdm(itertools.islice(baseline_spectra, nlines), total=nlines):
        data = list(parser.getDataAsTuple(key))
        n = len(data)

        charge = key[1]
        seq = parser.psms[key].seq

        spectra = _getSpectraPair(seq, data)
        if spectra is None:
            continue

        dd = {}
        for j in range(len(spectra)):
            score, yi, yw, bi, bw, yf = spectra[j]
            dd[j] = {i: ii for i, ii in zip(yi, yw)}

        pepdf = pd.DataFrame.from_dict(dd)
        pepdf.fillna(0, inplace=True)
        corr = pepdf.corr(method='pearson')
        p = corr.iat[0, 1]
        corrs['baseline'].append(p)

    print(f'finished parsing baseline data...')

    model_pickle = args.model
    mock_pickle = args.mock
    file_gen = common_utils.yield_open(filenames=args.test_files)
    testdata = itertools.chain.from_iterable(file_gen)

    with open(model_pickle, 'rb') as picklefile, open(mock_pickle, 'rb') as mockfile:
        model = pickle.load(picklefile)
        mockmodel = pickle.load(mockfile)

        np.seterr(invalid='ignore')

        model = model.finalizeModel(alpha=args.alpha)
        mockmodel = mockmodel.finalizeModel(alpha=args.alpha)
        #nlines = args.max_spectra if args.max_spectra > 0 else None
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

                ions, probs = model.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)
                mockions, mockprobs = mockmodel.calc_fragments(charge=int(z), seq=seq, ion_type='y',
                                                               use_yfrac=False)

                d = {'exp': {int(i[1:]): ii for i, ii in zip(y_ions, y_ints)},
                     'model': {int(i[1:]): p for i, p in zip(ions, probs)},
                     'mock': {int(i[1:]): p for i, p in zip(mockions, mockprobs)}
                     }

                df = pd.DataFrame.from_dict(d)
                # df.reindex(sorted(df.index,key=lambda x: re.sub('[A-z]','',x)))
                # df.fillna(0, inplace=True) # TODO: this might be misleading
                corr = df.corr(method='pearson')
                r_mock = corr.iat[0, 1]
                r_mode = corr.iat[0, 2]

                corrs['mock'].append(r_mock)
                corrs['model'].append(r_mode)

            except ValueError as e:
                print("Unexpected number of tokens found on line!")
                e.args += (line,)
                raise

    df = pd.DataFrame.from_dict(corrs, orient='index')
    df = df.transpose()
    df = df.reindex(columns=['baseline', 'model', 'mock'])
    # cols = sorted(df.columns.tolist())
    # df = df[cols]

    nvals = df.count()
    xticklabs = [f'{z}\n(n={nvals[i]})' for i, z in enumerate(df.columns)]

    ax = sns.boxplot(data=df, linewidth=2)
    ax.set_xticklabels(labels=xticklabs)
    plt.savefig(f'{args.name}.pdf')


parser = argparse.ArgumentParser(description='Scripts for evaluating HMM fragmentation predictor')
parser.add_argument('-t', '--nthreads', help='number of threads to use', default=1, type=int)
parser.add_argument('-a', '--alpha', help="alpha parameter for finalizing models", type=int, default=400)
parser.add_argument('--out', help='name of directory where any resultant files will be stored', default='.')
subparsers = parser.add_subparsers(help='sub-command help')

# create the parser for the "mock_model" command
parser_mock = subparsers.add_parser('mock_model', help='evaluates model performance by comparing to a mock model')
parser_mock.add_argument('--model', help='path to model to use')
parser_mock.add_argument('--mock', help='path to corresponding mock model')
parser_mock.add_argument('--test_files', nargs='+', help='files to check corr on ')
parser_mock.set_defaults(func=vs_mock)

# create the parser for the "x_val" command
parser_xval = subparsers.add_parser('x_val', help='evaluates variability of model performance using N-fold x-validation')
parser_xval.add_argument('-o', '--order', help='order, default = 2', default=2, type=int)
parser_xval.add_argument('-i', '--input', help='input_file', nargs='+')
parser_xval.add_argument('-x', '--nfold', help='nbr of slices for X validation', default=5, type=int)
parser_xval.add_argument('--allcombos', help='whether or not to check all combinations', default=False)
parser_xval.set_defaults(func=x_validation)


# create the parser for the "mock_model" command
parser_base = subparsers.add_parser('baseline', help='evaluates model performance by comparing bsaeline variation in spectra')
parser_base.add_argument('--data', help='entire knowledgebase of spectra')
parser_base.add_argument('--max_spectra', default=-1, type=int, help='max number of spectra to process')
parser_base.add_argument('--spectra_threshold', default=100, type=int, help='min nbr of spectra to consider')
parser_base.add_argument('--model', help='path to model to use')
parser_base.add_argument('--mock', help='path to corresponding mock model')
parser_base.add_argument('--test_files', nargs='+', help='files to check corr on ')
parser_base.add_argument('--name', help='Name for the boxplot to be generated', default='baseline_box')
parser_base.set_defaults(func=baseline)

# parser.add_argument('-n', '--name', help='name of the model')
# parser.add_argument('-p', '--pickle', help='print/pickle location', default='hmm_models')

# parser.add_argument('-m', '--use_model', help='model to use, instead of training a new one')
# parser.add_argument('-f', '--test_files', nargs='+', help='files to check corr on ')
# parser.add_argument('-g', '--graph', default=False, action='store_true')
# parser.add_argument('-d', '--debug', default=False, action='store_true')

if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(process)-5d %(thread)d %(message)s')
    logger = logging.getLogger()
    args = parser.parse_args()
    args.func(args)

