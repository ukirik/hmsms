from hmm_model import FragmentHMM
import common_utils
import ms_utils
import argparse
import sys
import pickle
import itertools
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Train HMM')
parser.add_argument('-o', '--order', help='order, default = 2', default=2, type=int)
parser.add_argument('-i', '--input', help='input_file', nargs='+', default=sys.stdin)
parser.add_argument('--out', help='name of directory where any resultant files will be stored', default='.')
parser.add_argument('-n', '--name', help='name of the model')
parser.add_argument('-p', '--pickle', help='print/pickle location', default='hmm_models')
parser.add_argument('-t', '--nthreads', help='number of threads to use', default=1, type=int)
parser.add_argument('-m', '--use_model', help='model to use, instead of training a new one')
parser.add_argument('-f', '--test_files', nargs='+', help='files to check corr on ')
parser.add_argument('-g', '--graph', default=False, action='store_true')
parser.add_argument('-d', '--debug', default=False, action='store_true')

args = parser.parse_args()


def annotate(row, ax):
    #ax.annotate(row.name, (row.exp, row.model),
    ax.annotate(row.Index, (row.exp, row.model),
                xytext=(10, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle="-", connectionstyle="arc,angleA=180,armA=10"),
                family='sans-serif', fontsize=8, color='darkslategrey')


def plot2File(df, file, seq, z, score, p, s):
    """ Plot predictions vs experimental """

    score = float(score)
    # plttitle = f"Correlations for {seq}+{z} \n pearson={p} \n spearman={s}"
    ax = df.plot(x='exp', y='model', kind='scatter', s=40)
    plt.figtext(.5, .9, f"Correlations for {seq}+{z} {score:{6}.{5}}", fontsize=10, ha='center')
    plt.figtext(.5, .8, f"pearson={p:{5}.{4}} \n spearman={s:{5}.{4}}", fontsize=8, ha='center')
    df.apply(annotate, ax=ax, axis=1)
    #    for row in df.itertuples():
    #        ax.annotate(row.Index, (row.exp, row.model),
    #                    xytext=(10, 20), textcoords='offset points',
    #                    arrowprops=dict(arrowstyle="-", connectionstyle="arc,angleA=180,armA=10"),
    #                    family='sans-serif', fontsize=8, color='darkslategrey')

    plt.savefig(file, bbox_inches='tight', format='pdf')
    plt.close()


def check_corr(testfiles, m, zeronas=False):
    np.seterr(invalid='ignore')
    model = m.finalizeModel(alpha=256)
    results = []

    _zbin = lambda x: x if int(x) < 4 else '4+'

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

    for somefile in testfiles:
        print(f"Calculating predication correlations for file: {somefile}")

        with open(somefile, 'r') as f:
            nlines = common_utils.getNbrOfLines([somefile])
            # nlines = 10000
            for line in tqdm(itertools.islice(f, nlines), mininterval=1, total=nlines):
                try:
                    tokens = line.rstrip('\r\n').split('\t')
                    z, seq, score, y_ions, y_ints, b_ions, b_ints, y_frac = tokens
                    if len(y_ints) == 0:
                        continue

                    y_ints = [float(i) for i in y_ints.split(' ')]
                    y_ions = y_ions.split(' ')
                    # b_ints = [float(i) for i in b_ints.split(' ')]
                    # b_ions = b_ions.split(' ')
                    ions, probs = model.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)

                    d = {'exp': {i: ii for i, ii in zip(y_ions, y_ints)},
                         'model': {ion: prob for ion, prob in zip(ions, probs)}
                         }

                    df = pd.DataFrame.from_dict(d)
                    if zeronas:
                        df.fillna(0, inplace=True)

                    p = df.corr(method='pearson').iat[0, 1]
                    s = df.corr(method='spearman').iat[0, 1]
                    res = {
                        'seq': seq,
                        'charge': z,
                        'z_bin': _zbin(z),
                        'peplen': len(seq),
                        'l_bin': _lbin(len(seq)),
                        'mpt_class': ms_utils.getMPClass(seq, z),
                        'exp_ints': df['exp'].to_json(),
                        'pred_ints': df['model'].to_json(),
                        'pearsons': p,
                        'spearmans': s
                    }
                    results.append(res)

                except ValueError as e:
                    print("Unexpected number of tokens found on line, skipping this entry!")
                    e.args += (line,)
                    continue

        # print("Correlations:")
        # print("=" * 30)
        # import warnings
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     df_p = pd.DataFrame.from_dict(pearson).applymap(np.nanmean)
        #     df_p.index = ["7-11", "12-16", "17-21", "22-26", "27+"]
        #     df_p.rename(columns={'4': '4+'}, inplace=True)
        #     print(df_p)
        #
        #     df_s = pd.DataFrame.from_dict(spearman).applymap(np.nanmean)
        #     df_s.index = ["7-11", "12-16", "17-21", "22-26", "27+"]
        #     df_s.rename(columns={'4': '4+'}, inplace=True)
        #     print(df_s)
        #     print()

    d = pd.DataFrame(results)
    d = d[['seq', 'charge', 'z_bin', 'peplen', 'l_bin', 'mpt_class', 'pearsons', 'spearmans', 'exp_ints', 'pred_ints']]
    return d


def pool_worker(filename, doshuffle):
    print(f"Processing file: {filename}")
    with open(filename, 'r') as f:
        partial = FragmentHMM(order=args.order, indata=f, shuffle=doshuffle)
        return partial


def generateModel(is_mock=False, save=True):
    model = None
    with ThreadPool(args.nthreads) as pool:
        from functools import partial
        print("Instantiating an order {} model, "
              "intermediate models will be created and used as needed...".format(args.order))

        # for f in tqdm(files, desc='Reading training data', mininterval=10):
        # partial_counts = pool.map(lambda f: FragmentHMM(order=args.order, indata=f), files)
        partial_models = pool.imap_unordered(partial(pool_worker, doshuffle=is_mock), args.input)
        # model = FragmentHMM.from_partials(partial_models)
        i = 0
        for m in partial_models:
            logger.info(f'Starting to merge model {i}')
            if model is None:
                import copy
                model = copy.deepcopy(m)
            else:
                model += m
            logger.info(f'Done merging...')
            i += 1

    if save:
        prefix = "hmm".format(model.order) if args.name is None else args.name
        suffix = "mock" if is_mock else ""
        picklefile = f"{prefix}_{model.order}_{suffix}.pickle"
        model.serialize(args.pickle, picklefile)
        print("Saving model {} to disk at {}".format(picklefile, args.pickle))

    return model


if __name__ == '__main__':

    file_gen = common_utils.yield_open(filenames=args.input)
    files = itertools.chain.from_iterable(file_gen)

    from multiprocessing import Pool as ThreadPool
    import logging

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(process)-5d %(thread)d %(message)s')
    logger = logging.getLogger()

    if args.use_model is not None:
        picklefile = open(args.use_model, 'rb')
        model = pickle.load(picklefile)
    else:
        model = generateModel()
        mock = generateModel(is_mock=True)

    if args.test_files:
        print(f'testing files: {args.test_files}')
        if args.out is not '.':
            import os
            os.makedirs(args.out, exist_ok=True)

        df = check_corr(args.test_files, model)
        df.to_csv(path_or_buf=os.path.join(args.out, "pred_results.csv"))
        print(f'Average Pearsons correlation: {df["pearsons"].mean()}')
        print(f'Average Spearmans correlation: {df["spearmans"].mean()}')

