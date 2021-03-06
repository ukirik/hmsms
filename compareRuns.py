import argparse
import os
import multiprocessing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Consolidate and evaluate results from multiple runs')
parser.add_argument('-i', '--input', help='input_files', nargs='+', required=True)
parser.add_argument('--out', help='name of directory where any resultant files will be stored', default='.')
parser.add_argument('-t', '--nthreads', help='number of threads to use', default=1, type=int)
parser.add_argument('-g', '--graph', default=False, action='store_true')
parser.add_argument('-n', '--name', default='merged_out.csv', help='name of the output file')
args = parser.parse_args()


def getdf(filepath):
    logger.info(f'Processing {filepath}')
    dirpath, filename = os.path.split(filepath)
    df = pd.read_csv(filepath, index_col=0)
    df['run'] = dirpath
    return df


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(process)-5d %(thread)d %(message)s')
    logger = logging.getLogger()

    logger.info(f'Following files will be processed and merged:\n{args.input}')
    with multiprocessing.Pool(args.nthreads) as pool:
        frames = pool.imap_unordered(getdf, args.input)

        logger.info(f'Merging all dataframes...')
        df = pd.concat(frames)
        logger.info(f'Merging complete! Printing to file {args.name}')
        df['z_bin'] = df['z_bin'].astype('category')
        df['l_bin'] = df['l_bin'].astype('category')
        df['mpt_class'] = df['mpt_class'].astype('category')
        df['run'] = df['run'].astype('category')
        print(df.info())
        df.to_csv(args.name)

