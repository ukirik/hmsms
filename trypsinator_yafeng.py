"""
This is a slightly modified version of Yafeng Zhu's trypsin digestion program (https://github.com/yafeng/trypsin)
Input: a fasta file which contains protein sequence to be digested,
Output: a txt file which contains all trypsin digested peptides and corresponding protein accessions.
Modificaitons from the original script:
1. Porting from py2.7 -> py3.6
2. CLI is changed to argparse
3. Min-length option to filter output
4. Added progress meter tqdm

"""
import sys
import argparse
import tqdm
from Bio import SeqIO

parser = argparse.ArgumentParser(description='Train HMM')
parser.add_argument('-i', '--input', help='input_file', default=sys.stdin)
parser.add_argument('-o', '--out', help='output file name', default=None)
parser.add_argument('-n', '--missed-cleavage', help='number of misscleavage allowed', default=1)
parser.add_argument('-l', '--min-length', help='minimum length to be considered', default=7)
args = parser.parse_args()


def TRYPSIN(proseq, miss_cleavage):
    peptides = []
    cut_sites = [0]
    appendPep = lambda x: peptides.append(x) if len(x) > args.min_length else None
    for i in range(0, len(proseq) - 1):
        if proseq[i] == 'K' and proseq[i + 1] != 'P':
            cut_sites.append(i + 1)
        elif proseq[i] == 'R' and proseq[i + 1] != 'P':
            cut_sites.append(i + 1)

    if cut_sites[-1] != len(proseq):
        cut_sites.append(len(proseq))

    if len(cut_sites) > 2:
        if miss_cleavage == 0:
            for j in range(0, len(cut_sites) - 1):
                appendPep(proseq[cut_sites[j]:cut_sites[j + 1]])

        elif miss_cleavage == 1:
            for j in range(0, len(cut_sites) - 2):
                appendPep(proseq[cut_sites[j]:cut_sites[j + 1]])
                appendPep(proseq[cut_sites[j]:cut_sites[j + 2]])

                appendPep(proseq[cut_sites[-2]:cut_sites[-1]])

        elif miss_cleavage == 2:
            for j in range(0, len(cut_sites) - 3):
                appendPep(proseq[cut_sites[j]:cut_sites[j + 1]])
                appendPep(proseq[cut_sites[j]:cut_sites[j + 2]])
                appendPep(proseq[cut_sites[j]:cut_sites[j + 3]])

            appendPep(proseq[cut_sites[-3]:cut_sites[-2]])
            appendPep(proseq[cut_sites[-3]:cut_sites[-1]])
            appendPep(proseq[cut_sites[-2]:cut_sites[-1]])
    else:  # there is no trypsin site in the protein sequence
        appendPep(proseq)
    return peptides


if __name__ == '__main__':
    handle = SeqIO.parse(args.input, 'fasta')
    outf = args.out if args.out is not None else args.input+'_trypsinated.out'
    output = open(outf, 'w')

    for record in tqdm.tqdm(handle):
        proseq = str(record.seq)
        peptide_list = TRYPSIN(proseq, args.missed_cleavage)
        for peptide in peptide_list:
            output.write("%s\t%s\n" % (record.id, peptide))

    handle.close()
