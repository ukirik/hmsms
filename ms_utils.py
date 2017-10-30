import collections
from enum import Enum

aadict = {
    'A'	: 71.037114,
    'R'  : 156.101111,
    'N'	 : 114.042927,
    'D'	 : 115.026943,
    'C'	 : 103.009185,
    'E'	 : 129.042593,
    'Q'	 : 128.058578,
    'G'	 : 57.021464,
    'H'	 : 137.058912,
    'I'	 : 113.084064,
    'L'	 : 113.084064,
    'K'	 : 128.094963,
    'M'	 : 131.040485,
    'F'	 : 147.068414,
    'P'	 : 97.052764,
    'S'	 : 87.032028,
    'T'	 : 101.047679,
    'U'	 : 150.95363,
    'W'	 : 186.079313,
    'Y'	 : 163.06332,
    'V'	 : 99.068414
}

immdict = {
    'H' : [110],
    'I' : [86],
    'L' : [86],
    'F' : [120],
    'P' : [70],
    'Q' : [101],
    'W' : [159, 130, 170, 171],
    'Y' : [136]
}

misc = {
    'H'  : 1.007825032,
    'p'  : 1.00727647,
    'OH' : 17.00274,
    'H2O': 18.010565,
    'NH3': 17.026549
}

def getMass(seq, iontype=''):
    h   = 1.00794
    oh  = 17.00274
    h2o = 18.01057
    nh3 = 17.02655

    m = h if iontype == 'b' else h2o
    for ch in seq:
        m += aadict[ch]

    return m

def getAACounts(seq):
    c = collections.Counter(seq)
    return {aa: c[aa] for aa in seq}

class MPT(Enum):
    MOBILE = 1
    PARTIAL = 2
    NONMOBILE = 3

def getMPClass(seq, z):
    counts = collections.defaultdict(int, getAACounts(seq))
    if z > counts['R'] + counts['K'] + counts['H']:
        return MPT.MOBILE
    elif counts['K'] + counts['H'] > z - counts['R'] > 0:
        return MPT.PARTIAL
    elif counts['R'] > z:
        return MPT.NONMOBILE
    else:
        AssertionError('Should not happen!')