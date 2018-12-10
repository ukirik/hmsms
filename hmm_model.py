from hmm_utils import *
import numpy as np
import copy
import os
import pickle
import math
import random


class FragmentHMM(object):
    def __init__(self,
                 indata=None,
                 order=2,
                 window=(5, 5),
                 length=(7, 27),
                 charge=(2, 4),
                 single_precision=False,
                 alphabet='ACDEFGHIKLMNPQRSTUVWY',
                 # Adding U as a valid aa, since some human prots contain it (e.g. GPX)
                 blank_aa=' ',
                 shuffle=False):

        # Initiate states
        n_states, c_states = window
        state_labels = ['nt', 'nx'] + ['n' + str(x) for x in range(n_states - 2, 0, -1)] + \
                       ['c' + str(x) for x in range(1, c_states - 1)] + ['cx', 'ct', 'c*']

        self.states = {i: j for i, j in zip(state_labels, range(len(state_labels)))}

        self.blank_aa = blank_aa
        self.emissions_str = alphabet + blank_aa
        self.order = order
        self.min_z, self.max_z = charge
        self.min_len, self.max_len = length

        self.T = {}  # Transitions matrix, previously named A
        self.E = {}  # Emission matrix, previously named B
        self.pi = {}  # Initial probabilities
        self.y_frac = np.zeros((self.max_z + 1, self.max_len + 1), np.float32)
        self.counts = np.zeros((self.max_z + 1, self.max_len + 1), np.uint16)
        self.T_raw = None
        self.E_raw = None
        self.nlines = 0
        self.finalized = False
        self.shuffle = shuffle

        # Init matrices
        n_states = len(self.states)
        self.pcount = 0.5
        _normalize_rows = lambda x: x / x.sum(axis=1)[:, np.newaxis]

        if indata is not None:
            self.t_init = np.ones((n_states - 1, n_states)) * self.pcount
            self.t_init = _normalize_rows(self.t_init)  # TODO is this necessary?

            for ion in 'yb':
                self.pi[ion] = IProbs(self.min_len, self.max_len, self.min_z, self.max_z, n_states, self.pcount)
                self.E[ion] = Emissions(n_states, order=self.order, charge=charge, single_precision=single_precision, emissions=self.emissions_str)
                self.T[ion] = OffsetList(self.min_z)
                for i in range(self.min_z, self.max_z + 1):
                    self.T[ion].append(copy.deepcopy(self.t_init))

            self.getCounts(indata)

    @classmethod
    def from_partials(cls, partial_models):
        for index, model in enumerate(partial_models):
            if index == 0:
                if not isinstance(model, FragmentHMM):
                    raise ValueError(f"Instance of FragmentHMM expected, {model.__class__} found")

                obj = copy.deepcopy(model)
            else:
                obj += model

        return obj

    def comparable(self, other):
        if not isinstance(other, FragmentHMM):
            return False
        if self.order != other.order:
            return False
        if self.min_z != other.min_z or self.max_z != other.max_z:
            return False
        if self.min_len != other.min_len or self.max_len != other.max_len:
            return False
        if self.states != other.states:
            return False
        if self.emissions_str != other.emissions_str:
            return False
        return True

    def getCounts(self, training_data):

        print("Training model...".format(self.order))
        for line in training_data:
            self.nlines += 1
            try:
                tokens = line.rstrip('\r\n').split('\t')
                charge, seq, score, y_ions, y_ints, b_ions, b_ints, y_frac = tokens
            except ValueError as e:
                print("Unexpected number of tokens found on line!")
                e.args += (line,)
                raise

            charge = int(charge)
            l = len(seq)

            # # Collect all peptides longer than max_len into a bin
            if l < self.min_len:
                continue
            elif l > self.max_len:
                l = self.max_len

            # # Collect all charge states higher than max_z into a bin
            if charge < self.min_z:
                continue
            elif charge > self.max_z:
                charge = self.max_z

            try:
                loop_over = zip('yb', ((y_ions, y_ints), (b_ions, b_ints)))
                for ion_type, (ions, intensities) in loop_over:
                    if len(ions) == 0 or len(intensities) == 0:
                        continue

                    intensities = [float(x) for x in intensities.split(' ')]
                    sum_int = sum(intensities)
                    weights = [x / sum_int for x in intensities]
                    ions = [int(x[1:]) for x in ions.split(' ')]

                    # l cannot be used for this sanity check, since we are pooling peptides longer than max_len
                    if max(ions) >= len(seq):
                        raise AssertionError(f"Unexpected number of ions:\n ions: {ions}, sequence: {seq}")

                    # Check if there are missing ions, if so, add them with intensity 0.0
                    if len(ions) < len(seq) - 1:
                        ions, weights = self.pad_missing_fragments(seq, ions, weights)

                    self.y_frac[charge, l] += float(y_frac)
                    self.counts[charge, l] += 1
                    self.update_parameters(seq, ions, weights, charge, ion_type)
            except KeyError:
                # This might happen if an U or or O is encountered in the peptide seq
                import sys, traceback
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("Warning: unexpected value found")
                traceback.print_exception(exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout)
                continue

        with np.errstate(divide='ignore', invalid='ignore'):
            self.y_frac /= self.counts

    def pad_missing_fragments(self, seq, ions, weights):
        """
        Pads the ions and weights arrays with missing fragments with weight 0.0
        """
        # TODO check that this works as intended and not superslowly
        temp = np.zeros(len(seq) - 1)
        for i, w in zip(ions, weights):
            temp[i - 1] = w

        ions = list(range(1, len(seq)))
        weights = temp.tolist()
        assert len(ions) == len(weights)

        return ions, weights

    def update_parameters(self, seq, ions, weights, charge, ion_type):
        pi = self.pi[ion_type][charge]
        A = self.T[ion_type][charge]
        B = self.E[ion_type]
        l = len(seq)

        """
        Creates a mock model by shuffling the ion labels in place. This mock model can then
        be used for establishing a basis for evaluation of predictions
        """
        if self.shuffle:
            random.shuffle(ions)

        for ion, weight in zip(ions, weights):
            # avoid a bunch of zero additions
            if np.isclose(0.0, weight, atol=1e-9):
                continue

            path = self.get_path(ion, l, ion_type)

            for o in range(0, self.order + 1):
                emission = self.seq_to_em_w_prior(seq, o)
                temp = B[charge, o]
                for state, em in zip(path, emission):
                    prior, aa = em
                    temp[state, prior, aa] += weight

            for _from, _to in zip(path, path[1:]):
                A[_from, _to] += weight

            # Second index of the pi should be path[0] not ion TODO: change if this breaks something
            if l > self.max_len:
                pi[self.max_len][path[0]] += weight
            else:
                pi[l][path[0]] += weight

    """
    Path related methods: for a given ion, find and/or translate path
    """

    def find_state(self, x):
        if x > self.states['cx']:
            return self.states['cx']

        if x < self.states['nx']:
            return self.states['nx']

        return x

    def translate_path(self, path):
        return [next(key for key, val in self.states.items() if val == el) for el in path]

    def get_path(self, ion, length, ion_type):
        if ion_type == 'y':
            return self.get_path(length - ion, length, 'b')

        start = self.states['n1'] - ion + 1
        path = [self.find_state(i) for i in range(start, start + length)]

        if path[0] != self.states['n1']:
            path[0] = self.states['nt']

        if path[-1] == self.states['c1']:
            path[-1] = self.states['c*']
        else:
            path[-1] = self.states['ct']

        return path

    def seq_to_emission(self, seq, order=None):
        if order is None:
            order = self.order

        seq = self.blank_aa * order + seq
        return [seq[n:n + order + 1] for n in range(0, len(seq) - order)]

    def seq_to_em_w_prior(self, seq, order):
        ems = self.seq_to_emission(seq, order)
        return [(e[:-1], e[-1]) for e in ems]

    """ GENERATOR METHODS
    These methods are used for inferring ion probabilities
    """

    def _ion_prob(self, seq, charge, ion, ion_type):
        """
        Calculates the probability of an ion to be observed given the model
        """

        A = self.T[ion_type]
        B = self.E[ion_type]
        pi = self.pi[ion_type]

        seqlen = len(seq)
        path = self.get_path(ion, seqlen, ion_type)
        emission = self.seq_to_emission(seq)

        """ Initial probability """
        # last index of pi should be path[0] not ion TODO: change if this messes up something
        # prob = pi[charge][seqlen][ion] if seqlen < self.max_len else pi[charge][self.max_len][ion]
        prob = pi[charge][seqlen][[path[0]]] if seqlen < self.max_len else pi[charge][self.max_len][path[0]]

        for s, (prior, e) in zip(path, emission):
            prob *= B[charge, self.order, s, prior, e]

        for a, b in zip(path[:-1], path[1:]):
            prob *= A[charge][a, b]

        return prob

    def _ion_prob_l(self, seq, charge, ion, ion_type):
        A = self.T[ion_type]
        B = self.E[ion_type]
        pi = self.pi[ion_type]

        seqlen = len(seq)
        path = self.get_path(ion, seqlen, ion_type)
        # emission = self.seq_to_emission(seq)
        emission = self.seq_to_em_w_prior(seq, self.order)

        """ Initial log probability """
        # last index of pi should be path[0] not ion TODO: change if this messes up something
        # prob = pi[charge][seqlen][ion] if seqlen < self.max_len else pi[charge][self.max_len][ion]
        prob = pi[charge][seqlen][[path[0]]] if seqlen < self.max_len else pi[charge][self.max_len][path[0]]
        prob = math.log(prob)

        for s, (prior, e) in zip(path, emission):
            p = B[charge, self.order, s, prior, e]
            prob += math.log(p)

        for a, b in zip(path[:-1], path[1:]):
            prob += math.log(A[charge][a, b])

        return math.exp(prob)

    def _get_ions_and_ps(self, seq, charge, ion_type, raw=False, use_yfrac=True):
        # contains what both calc_spectra and calc_fragments needs
        l = len(seq)
        ions = list(range(1, l))

        # if too long, sample from the highest bin, if too short predict uniform spectra
        # with random normal noise, to avoid NaNs in correlation calculation
        if charge < self.min_z or l < self.min_len:
            ions = ['%s%i' % (ion_type, ion) for ion in ions]
            constant = np.ones(l - 1, dtype=np.float) / (l - 1)
            # constant += np.random.normal(loc=1, scale=0.25, size=len(constant))
            peaks = abs(constant) / sum(abs(constant))
            return ions, peaks

        if charge > self.max_z:
            charge = self.max_z

        if l > self.max_len:
            l = self.max_len

        if ion_type == 'y':
            ions.reverse()
            ion_frac = self.y_frac[charge, l]
        else:
            ion_frac = 1 - self.y_frac[charge, l]

        # ps = np.array([self._ion_prob(seq, charge, ion, ion_type) for ion in ions])
        ps = np.array([self._ion_prob_l(seq, charge, ion, ion_type) for ion in ions])

        ion_frac = ion_frac if use_yfrac else 1
        if raw:
            ps = ion_frac * ps
        else:
            ps = ion_frac * ps / ps.sum()
            assert (np.isclose(1.0, np.sum(ps), atol=1e-9))

        ions = ['%s%i' % (ion_type, ion) for ion in ions]
        return ions, ps

    def calc_fragments(self, seq, charge, ion_type='all', use_yfrac=True):
        if not self.finalized:
            raise AssertionError('The model is not finalized, i.e. pseudocounts are not added yet! Call finalizeModel() with an appropriate alpha value')

        # y_ions, y_ps, b_ions, b_ps = [], [], [], []
        if ion_type == 'y':
            return self._get_ions_and_ps(seq, charge, 'y', use_yfrac=use_yfrac)
        elif ion_type == 'b':
            return self._get_ions_and_ps(seq, charge, 'b', use_yfrac=use_yfrac)
        else:
            y_ions, y_ps = self._get_ions_and_ps(seq, charge, 'y')
            b_ions, b_ps = self._get_ions_and_ps(seq, charge, 'b')
            return tuple(y_ions + b_ions), tuple(list(y_ps) + list(b_ps))

    def serialize(self, location, name):
        if not os.path.exists(location):
            os.mkdir(location)

        outfile = os.path.join(location, name)
        with open(outfile, 'wb') as f:
            pickle.dump(self, f)

    def finalizeModel(self, alpha):
        if self.finalized:
            AssertionError("Model already finalized!")

        model = copy.deepcopy(self)
        model.E_raw = self.E
        model.T_raw = self.T
        for ion_type in 'yb':
            model.E[ion_type].addPseudoCounts(alpha)

            for z in range(self.min_z, self.max_z + 1):
                for transitions in range(len(self.states) - 1):
                    model.T[ion_type][z][transitions] /= self.T[ion_type][z][transitions].sum()

            model.pi[ion_type].norm()

        model.finalized = True
        return model

    def __add__(self, other):
        """
        This method merges the counts in this model with the other model. It will attempt to add
        the counts found in IProbs, E and T matrices, if and only if two models are of the same
        order. Delegates the actual adding to __iadd__
        :param other: another model of the same order
        :return: a copy of this model, where the Iprobs/E/T are added summed
        """
        if not self.comparable(other):
            raise ValueError("Objects cannot be added!")

        temp = copy.deepcopy(self)
        temp += other
        return temp

    def __iadd__(self, other):
        """
        Adds two models in-place, such that self is modified after the call
        :param other:
        :return:
        """
        if not self.comparable(other):
            raise ValueError("Objects cannot be added!")

        with np.errstate(divide='ignore', invalid='ignore'):
            self.y_frac = self.y_frac * self.counts + other.y_frac * other.counts
            self.nlines += other.nlines
            self.counts += other.counts
            self.y_frac /= self.counts

        """
        Important to note how the pseudo counts are handled when merging partial models
        a naive sum will propagate defaults to become larger and larger
        self.pi (IProbs) and self.E (Emissions) objects handle the addition themselves
        self.T has to be handled explicitly
        """
        for ion_type in 'yb':
            self.pi[ion_type] += other.pi[ion_type]
            self.E[ion_type] += other.E[ion_type]
            for z in range(self.min_z, self.max_z + 1):
                self.T[ion_type][z] += other.T[ion_type][z] - self.t_init
                # for transitions in range(len(self.states) - 1):
                #     self.T[ion_type][z][transitions] += (other.T[ion_type][z][transitions] -

        return self




