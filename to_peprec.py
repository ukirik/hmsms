import argparse, os, pathlib

parser = argparse.ArgumentParser(description='Convert training data to PEPREC')
parser.add_argument('-f', '--files', nargs='+', help='files contaning peptides')
parser.add_argument('-s', '--suffix', default='peprec', help='suffix for the output file names')
parser.add_argument('-o', '--output', default='.', help='where to save the output files')

if __name__ == '__main__':
    args = parser.parse_args()
    for infile in args.files:
        dirname, basename = os.path.split(infile)
        fname, ext = os.path.splitext(basename)
        outfname = f'{fname}_{args.suffix}{ext}'
        outdir = args.output if os.path.isabs(args.output) else os.path.abspath(args.output)
        pathlib.Path(outdir).mkdir(parents=True, exist_ok=True)
        outfile = os.path.join(outdir, outfname)
        print(f'Printing PEPREC to {outfile}')

        with open(infile, 'r') as inf, open(outfile, 'w') as outf:
            pepid = 0
            outf.write('spec_id modifications peptide charge\n')
            for line in inf:
                try:
                    tokens = line.rstrip('\r\n').split('\t')
                    charge, seq, score, y_ions, y_ints, b_ions, b_ints, y_frac = tokens
                    pepid = '_'.join([seq, charge])
                    outf.write(f'{pepid} - {seq} {charge}\n')
                except ValueError as e:
                    print("Unexpected number of tokens found on line!")
                    e.args += (line,)
                    raise