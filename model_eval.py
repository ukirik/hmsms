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
# parser.add_argument('-o', '--order', help='order, default = 2', default=2, type=int)
# parser.add_argument('-i', '--input', help='input_file', nargs='+', default=sys.stdin)
# parser.add_argument('--out', help='name of directory where any resultant files will be stored', default='.')
# parser.add_argument('-n', '--name', help='name of the model')
# parser.add_argument('-p', '--pickle', help='print/pickle location', default='hmm_models')
# parser.add_argument('-t', '--nthreads', help='number of threads to use', default=1, type=int)
# parser.add_argument('-m', '--use_model', help='model to use, instead of training a new one')
# parser.add_argument('-f', '--test_files', nargs='+', help='files to check corr on ')
# parser.add_argument('-g', '--graph', default=False, action='store_true')
# parser.add_argument('-d', '--debug', default=False, action='store_true')

args = parser.parse_args()


def vs_mock():

    model_pickle = args.model
    mock_pickle = args.mock
    file_gen = common_utils.yield_open(filenames=args.test_files)
    testdata = itertools.chain.from_iterable(file_gen)
    dd = {}

    with open(model_pickle, 'rb') as picklefile, open(mock_pickle, 'rb') as mockfile:

        model = pickle.load(picklefile)
        mockmodel = pickle.load(mockfile)

        np.seterr(invalid='ignore')

        model = model.finalizeModel(alpha=250)
        mockmodel = mockmodel.finalizeModel(alpha=250)

        nlines = 1000
        _z = lambda x: x if int(x) < 4 else '4+'
        _l = lambda s: (len(s) - 7) // 5 if (len(s) - 7) // 5 < 4 else 4

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
                               'charge': _z(z),
                               'pep length': getbin(len(seq)),
                               'model': 'mock',
                               'correlation': corrs.iat[0, 1]
                               }

                dd[counter + 1] = {'seq': seq,
                                   'charge': _z(z),
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
                            order=['7-11', '12-16', '17-21', '22-26', '27+'],
                            col_order=['2', '3', '4+'], kind="box", legend=False)

        ax.despine(offset=10, trim=True)
        plt.legend(loc='best')
        ax.savefig('vs_mock.pdf')


if __name__ == '__main__':
    vs_mock()