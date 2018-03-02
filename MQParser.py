import sys, random, re, logging, collections, os, io


class PSM(object):
    """
    This class represents the data pertained on a particular (peptideID, charge state) tuple
    """

    __slots__ = ['id', 'seq', 'weights', 'ions', 'scores', 'y_fractions']

    def __init__(self, key, seq):
        self.id = key
        self.seq = seq
        self.weights = {'b': list(), 'y': list()}
        self.ions = {'b': list(), 'y': list()}
        self.scores = list()
        self.y_fractions = list()

    def addSpectra(self, ions_n_ints, score, y_frac):
        self.scores.append(score)
        self.y_fractions.append(y_frac)
        for ion_type, val_tuple in ions_n_ints.items():
            self.ions[ion_type].append(val_tuple[1])
            self.weights[ion_type].append(val_tuple[2])

    def addSpectra2(self, ions_n_ints, score, y_frac):
        """
        Stores the ion series as a list of ints rather than a list of strings
        this is useful for decreasing memory consumption
        """
        self.scores.append(score)
        self.y_fractions.append(y_frac)
        for ion_type, val_tuple in ions_n_ints.items():
            key, ion_series, ion_ints = val_tuple
            ion_series = [int(x[1:]) for x in ion_series]
            self.ions[ion_type].append(ion_series)
            self.weights[ion_type].append(ion_ints)

    def getSize(self):
        from common_utils import get_size
        total = sys.getsizeof(self.id)
        total += sys.getsizeof(self.seq)
        total += get_size(self.weights)
        total += get_size(self.ions)
        total += get_size(self.scores)
        total += get_size(self.y_fractions)
        return total

    def listSize(self):
        from common_utils import get_size
        print(f"Size listing of PSMs for {(self.seq, self.id[1])}, # of spectra={len(self.scores)}")
        print("-"*60)
        print(f"{'Id/key':8s}:{sys.getsizeof(self.id):>6d}")
        print(f"{'Sequence':8s}:{sys.getsizeof(self.seq):>6d}")
        print(f"{'Weights':8s}:{get_size(self.weights):>6d}")
        print(f"{'Ions':8s}:{get_size(self.ions):>6d}")
        print(f"{'Scores':8s}:{get_size(self.scores):>6d}")
        print(f"{'y_fracs':8s}:{get_size(self.y_fractions):>6d}")
        print("=" * 60)

    def listIons(self):
        from itertools import islice
        for i in islice(range(len(self.ions['y'])), 7):
            y = self.ions['y'][i]
            b = self.ions['b'][i]
            print(', '.join(y))
            print(', '.join(b))
            print()


class MQParser(object):
    __slots__ = ['psms', 'pos', 'neg', 'fdr', 'fdr_threshold', 'ion_threshold', 'mc', 'debug']

    def __init__(self, fdr_threshold=0.01, ion_threshold=0.001, mc=1, dodebug=False):
        self.psms = {}
        self.pos = 0.0
        self.neg = 0.0
        self.fdr = 0.0
        self.fdr_threshold = fdr_threshold
        self.ion_threshold = ion_threshold
        self.mc = mc
        self.debug = dodebug

    def newPSM(self, key, seq):
        psm = PSM(key, seq)
        self.psms[key] = psm

    def updateInfo(self, key, ions_ints, s, yfrac):
        self.psms[key].addSpectra2(ions_ints, s, yfrac)

    def getDataAsTuple(self, key):
        if key not in self.psms:
            raise KeyError('Specified key not found')

        hit = self.psms[key]

        # Assertions:
        if len(hit.scores) != len(hit.y_fractions):
            raise AssertionError("Values of uneven length for {}".format(key))
        if len(hit.ions['b']) != len(hit.weights['b']):
            raise AssertionError("Values of uneven length for {}".format(key))
        if len(hit.ions['y']) != len(hit.weights['y']):
            raise AssertionError("Values of uneven length for {}".format(key))

        _getionseries = lambda itype, iseries: [[itype+str(ion) for ion in s] for s in iseries]

        return zip(hit.scores,
                   _getionseries('y', hit.ions['y']), hit.weights['y'],
                   _getionseries('b', hit.ions['b']), hit.weights['b'],
                   hit.y_fractions)

    def getKeys(self):
        return list(self.psms.keys())

    def get_subset(self, keyset):
        import copy
        parser = copy.copy(self)
        parser.psms = {k: v for k, v in self.psms.items() if k in keyset}
        return parser

    def incrementNeg(self):
        self.neg += 1
        self._updateFDR()

    def incrementPos(self):
        self.pos += 1

    def _updateFDR(self):
        self.fdr = self.neg / (self.neg + self.pos)
        return self.fdr

    def __contains__(self, key):
        return self.psms.__contains__(key)

    def processChunk(self, df, skip_sort):
        if self.debug:
            print("{0:10}: {1:>12,}".format("Data Frame", sys.getsizeof(df)))
            print("{0:10}: {1:>12,}".format("Parser", sys.getsizeof(self)))
            input("Enter to continue...")

        if not skip_sort:
            df = df.sort_values(by='PEP')

        '''
        Initiate variables, TODO: Consider placing these in a container
        '''
        for row in df.itertuples():
            '''
            Filtering rows based on reverse hits, modified peptides or missing intensities
            '''
            if row.Reverse == '+' or row.Proteins == '':
                self.incrementNeg()
                continue
            if row.Modifications != 'Unmodified':
                continue
            if row.Intensities == '':
                continue
            if row.Number_of_Matches == 0:
                continue
            if not self._checkMisscleavage(row.Sequence):
                continue

            '''
            Use (Peptide ID, Charge) tuple as a key. OBS: The consistency of peptide IDs is delegated to MQ
            '''
            key = (int(row.Peptide_ID), int(row.Charge))
            intensities = [float(x) for x in row.Intensities.split(';')]
            ion_series = re.findall('[abcxyz]\d+', row.Matches)
            ii = list(zip(intensities, ion_series))
            total_int = {}
            temp = {}

            '''
            Normalize each ion in b/y series to the sum of respective series
            '''
            if key not in self:
                self.newPSM(key, row.Sequence)

            for ion_type in 'by':
                total_int[ion_type] = sum([i for (i, s) in ii if s[0] == ion_type])
                ion_int = collections.defaultdict(float)
                for _int, _ion in ii:
                    if _ion[0] == ion_type:
                        ion_int[_ion] += _int / total_int[ion_type]

                if len(ion_int) != 0:
                    _ions, _weights = zip(*(x for x in ion_int.items() if x[1] > self.ion_threshold))
                    temp[ion_type] = (key, _ions, _weights)
                else:
                    # If a spectra has only y- or b- ions an empty list is appended
                    # TODO: Is there a better way to handle this? Are the empty lists used ever?
                    temp[ion_type] = (key, [], [])
            try:
                yfrac = total_int['y'] / (total_int['y'] + total_int['b'])
                self.updateInfo(key, temp, float(row.Score), yfrac)

            except ZeroDivisionError:
                '''
                This happens when there are no y- or b- ions, for ex. when one spectra 
                only has an a2 ion, and no other ions! Such spectra are useless and thus deleted
                '''

                logging.debug("Empty/Useless spectra found:")
                logging.debug(row)

            '''
            Update FDR counter and check limit criterion 
            '''
            self.incrementPos()
            if self.fdr > self.fdr_threshold:
                print("Breaking loop on FDR limit: pos={}, neg={}".format(self.pos, self.neg))
                return False

        return True

    def outputResults(self, outf, spectra_t, nslices):
        all_keys = self.getKeys()
        l = len(all_keys)
        random.shuffle(all_keys)
        nfiles = nslices
        key_slices = [all_keys[i * l // nfiles:(i + 1) * l // nfiles] for i in range(nfiles)]

        filepath = os.path.join(outf, "training_set_{}_{}.tsv")
        _openf = lambda i: open(filepath.format(i, spectra), 'w')

        for i, keys in enumerate(key_slices):
            with open(filepath.format(i, spectra_t), 'w') as outfile:
                for key in keys:
                    data = list(self.getDataAsTuple(key))
                    # data = sorted(data, key=itemgetter(0), reverse=True) # TODO: Unnecessary, data is already sorted
                    if not data:    # check for empty spectra
                        continue

                    charge = key[1]
                    seq = self.psms[key].seq
                    spectra = self._getSpectra(data, spectra_t)
                    for s in spectra:
                        self.write_results(outfile, charge, seq, *s)

    def _checkMisscleavage(self, seq):
        matches = re.findall('[KR]', seq)
        if seq.endswith('K') or seq.endswith('R'):
            return len(matches) <= self.mc + 1
        else:
            return len(matches) <= self.mc

    def _getSpectra(self, data, spectra):
        # Pick the spectra to use, depending on runtime param
        if spectra == 'best' or spectra == 'top1':
            return [data[0]]

        elif spectra == 'top3':
            return [data[psm] for psm in range(min(len(data), 3))]

        elif spectra == 'med':
            return [data[len(data) // 2]]

        elif spectra == 'all':
            return data

        elif spectra == 'ave':
            _ysum_weight = collections.defaultdict(float)
            _bsum_weight = collections.defaultdict(float)
            yf = 0
            score = 0
            nhits = len(data)
            for _s, _yi, _yw, _bi, _bw, _yf in data:
                for i, w in zip(_yi, _yw):
                    _ysum_weight[i] += w / nhits

                for i, w in zip(_bi, _bw):
                    _bsum_weight[i] += w / nhits

                yf += _yf
                score += _s

            yi, yw, bi, bw = [], [], [], []
            if len(_ysum_weight) != 0:
                yi, yw = zip(*_ysum_weight.items())
            if len(_bsum_weight) != 0:
                bi, bw = zip(*_bsum_weight.items())

            yf /= nhits
            score /= nhits

            return [(score, yi, yw, bi, bw, yf)]

    def _write(self, f, charge, seq, score, y_ions, y_weights, b_ions, b_weights, y_frag):
        y_ions = ' '.join(map(str, y_ions))
        b_ions = ' '.join(map(str, b_ions))
        y_weights = ' '.join(map(str, y_weights))
        b_weights = ' '.join(map(str, b_weights))

        f.write("\t".join(map(str, (charge, seq, score, y_ions, y_weights, b_ions, b_weights, y_frag))))
        f.write('\n')

    def write_results(self, f, charge, seq, score, y_ions, y_weights, b_ions, b_weights, y_frag, i=-1):
        if isinstance(f, io.IOBase):
            self._write(f, charge, seq, score, y_ions, y_weights, b_ions, b_weights, y_frag)
        else:
            # TODO: is this necessary, when is it ever run?
            if i % 5 == 0:
                # write to benchmark file every 1/5 time
                self._write(f[-1], charge, seq, score, y_ions, y_weights, b_ions, b_weights, y_frag)
            elif i % 5 == 1:
                # write to test or training file every 1/5 time
                self._write(f[-2], charge, seq, score, y_ions, y_weights, b_ions, b_weights, y_frag)
            else:
                # write to training the rest of the time
                self._write(f[0], charge, seq, score, y_ions, y_weights, b_ions, b_weights, y_frag)