import argparse
import pickle
import operator
import pprint
import fileinput
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='Check correlations with different alpha')
parser.add_argument('-m', '--model', help='input pickle file for the model', required=True)
parser.add_argument('-t', '--test_files', nargs='+', help='files to check correlation with', required=True)
parser.add_argument('-l', '--linear_search', action='store_true', default=False,
                    help='toggle linear grid search, false by default')
parser.add_argument('-r', '--range', type=int, nargs=2, default=[1, 1000],
                    help='the range of values to search for an optimal alpha')
parser.add_argument('-n', '--nvals', type=int, nargs=1, default=10, help='the nbr of values to evaluate')
parser.add_argument('--nthreads', help='number of threads to use', default=4, type=int)
parser.add_argument('--multi_pass', help='toggle multiple pass search', default=False, action='store_true')

# parser.add_argument('-n', help='name of the model', default='hmm_model')
# parser.add_argument('-p', help='print/pickle location', default='hmm_models')
args = parser.parse_args()


def testAlpha(a, m, testdata):
    model = m.finalizeModel(alpha=a)
    corrs = []
    with fileinput.input(files=testdata) as line_iter:
        for line in line_iter:
            try:
                tokens = line.rstrip('\r\n').split('\t')
                z, seq, score, y_ions, y_ints, b_ions, b_ints, y_frac = tokens
                if len(y_ints) == 0:
                    continue
                if int(z) == 1:
                    continue

            except ValueError as e:
                print(f"Unexpected number of tokens found on line: {line}")
                e.args += (line,)
                continue

            y_ints = [float(i) for i in y_ints.split(' ')]
            y_ions = y_ions.split(' ')
            # b_ints = [float(i) for i in b_ints.split(' ')]
            # b_ions = b_ions.split(' ')
            ions, probs = model.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)

            d = {'exp': {i: ii for i, ii in zip(y_ions, y_ints)},
                 'model': {ion: p for ion, p in zip(ions, probs)}}

            df = pd.DataFrame.from_dict(d)
            p = df.corr(method='pearson').iat[0, 1]
            corrs.append(p)

    corrs = np.asarray(corrs)
    mean_corr = np.nanmean(corrs)
    return a, mean_corr


def estimateForAlphas(alphas, i):
    #testdata = itertools.chain.from_iterable(file_gen)

    from multiprocessing import Pool as ThreadPool
    from functools import partial
    with open(args.model, 'rb') as pickle_f, ThreadPool(args.nthreads) as pool:
        m = pickle.load(pickle_f)
        corrs = pool.imap_unordered(partial(testAlpha, m=m, testdata=args.test_files), alphas)
        sorted_res = sorted(corrs, key=operator.itemgetter(1), reverse=True)
        print(f"Iteration {i}:")
        pprint.pprint(dict(sorted_res))

    return sorted_res


def main(low, high, iter_counter):
    search_space = np.linspace(low, high, args.nvals, dtype=int) if args.linear_search else np.geomspace(low, high, args.nvals, dtype=int)
    iter_counter += 1
    res = estimateForAlphas(search_space, iter_counter)

    if args.multi_pass:
        # Re-run with second and third best, assumption is that best value is in between 2nd and 3rd best.
        best = res[0][0]
        x = res[1][0]
        y = res[2][0]
        assert not ((best < x and best < y) or (best > x and best > y)), f"best val={best}, sec={x}, third={y}"

        if abs(x-y) < 5:
            print(f"Optimal alpha is between {x}-{y}")
        elif x < y:
            main(x, y, iter_counter)
        else:
            main(y, x, iter_counter)
    else:
        print(f"Top alpha value is {res[0][0]} with an average corr {res[0][1]}")
        print(f"For comparison: the second best alpha was {res[1][0]} with an average corr {res[1][1]}")

if __name__ == "__main__":
    low, high = args.range
    main(low, high, 0)


