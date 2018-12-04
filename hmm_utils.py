from common_utils import OffsetList
import numpy as np
import copy


class IProbs(OffsetList):
    """
    This is a utility class that provides convenient access to
    Pi-matrix, giving the initial probabilities for states.
    TODO: Are the limitations (min/max length or charge) necessary?
    """
    __slots__ = ["min_l", 'max_l', 'min_z', 'max_z', 'n_states', 'pcount']

    def __init__(self,
                 min_length,
                 max_length,
                 min_charge,
                 max_charge,
                 n_states,
                 prior=0.5):

        self.min_l = min_length
        self.max_l = max_length
        self.min_z = min_charge
        self.max_z = max_charge
        self.n_states = n_states
        self.pcount = prior / n_states

        OffsetList.__init__(self, self.min_z)

        for z in range(self.min_z, self.max_z + 1):
            self.append(OffsetList(self.min_l))
            for l in range(self.min_l, self.max_l + 1):
                self[z].append(OffsetList(1))

                if isinstance(prior, float):
                    self[z][l] = np.ones(n_states, dtype=np.float64) * self.pcount

                elif isinstance(prior, IProbs):
                    # TODO is this ever used? or is it unusued case from development times
                    self[z][l] = prior[z][l]

    def norm(self):
        for z in range(self.min_z, self.max_z + 1):
            for l in range(self.min_l, self.max_l + 1):
                temp = self[z][l]
                _sum = sum(temp)
                if _sum == 0:
                    continue
                else:
                    temp /= _sum

    def comparable(self, other):
        if not isinstance(other, IProbs):
            return False
        if self.max_z != other.max_z or self.min_z != other.min_z:
            return False
        if self.max_l != other.max_l or self.min_l != other.min_l:
            return False
        if self.n_states != other.n_states:
            return False
        return True

    def __eq__(self, other):
        if not self.comparable(other):
            return False

        for z in range(self.min_z, self.max_z + 1):
            for l in range(self.min_l, self.max_l + 1):
                if not np.allclose(self[z][l], other[z][l]):
                    return False
                # for i in range(1, len(self[z][l])):
                #     if not math.isclose(self[z][l][i], other[z][l][i]):
                #         return False
        return True

    def __add__(self, other):
        if not self.comparable(other):
            raise ValueError("Objects cannot be added")

        temp = copy.deepcopy(self)
        temp += other
        return temp

    def __iadd__(self, other):
        if not self.comparable(other):
            raise ValueError("Objects cannot be added")

        for z in range(self.min_z, self.max_z + 1):
            for l in range(self.min_l, self.max_l + 1):
                # Subtract the pseudo counts to avoid doubling it (since they already have the pseudo count)
                self[z][l] += (other[z][l] - np.ones(self.n_states) * self.pcount)

        return self


class Econtainer(object):
    """
    This class represents the three dimensional 
    """

    __slots__ = ["n_states", "emissions", "size", "nvar", "values", "pcount"]

    def __init__(self,
                 n_states=-1, # TODO why is this the default value??
                 order=0,
                 pcount=0,
                 emissions='ACDEFGHIKLMNPQRSTVWY '):

        self.n_states = n_states
        self.emissions = {aa: index for (index, aa) in enumerate(emissions)}

        # Nbr of possible variables (emission combinations)
        self.nvar = len(emissions) ** order

        # Total size of the matrix, depends on the length
        self.size = (n_states, self.nvar, len(emissions))

        self.pcount = 1 / len(emissions) if order == 0 else pcount

        # temporary (hopefully!) ugly hack to keep memory usage low
        dt = np.float64 if order < 4 else np.float32
        self.values = np.ones(self.size, dtype=dt) * self.pcount

    def norm(self):
        """
        Row-wise normalize the emissions
        :return: normalized copy of this matrix
        """
        def inner_norm(x):
            x /= x.sum(axis=1)[:, np.newaxis]

        for state in range(self.n_states):
            inner_norm(self.values[state])

    def comparable(self, other):
        if not isinstance(other, Econtainer):
            return False
        if self.emissions != other.emissions:
            return False
        if self.size != other.size:
            return False
        if self.nvar != other.nvar:
            return False
        if self.n_states != other.n_states:
            return False
        return True

    # Lookup related methods
    def _lookupAAs(self, chars):
        """
        Iterate the prior AAs such that deeper priors with are aligned:
        # e.g. 'AA', 'CA', 'DA' ... rather than 'AA', 'AC', 'AD' ...
        :param chars: 
        :return: 
        """
        indx = 0
        for i, c in enumerate(chars):
            indx += self.emissions[c] * (len(self.emissions) ** i)
        return int(indx)

    def _reverseLookup(self, indx, order):
        if order == 0:
            return ''

        nvar = len(self.emissions) ** (order - 1)
        (q, r) = divmod(indx, nvar)
        c = next(key for key, value in self.emissions.items() if value == q)
        return self._reverseLookup(r, order - 1) + c

    def _inner_lookup(self, x):
        if isinstance(x, int):
            return x

        if isinstance(x, str):
            return self.emissions[x] if len(x) == 1 else self._lookupAAs(x)

        raise ValueError("x needs to be int or str, {} found".format(type(x)))

    def _lookup(self, x):
        if isinstance(x, int):
            return x

        if isinstance(x, tuple):
            return tuple([self._inner_lookup(i) for i in x])

        return None

    def __getitem__(self, y):
        return self.values[self._lookup(y)]

    def __setitem__(self, i, y):
        self.values[self._lookup(i)] = y

    def __add__(self, other):
        if not self.comparable(other):
            raise ValueError("Objects cannot be added!")

        temp = copy.deepcopy(self)
        temp.values += other.values
        return temp

    def __iadd__(self, other):
        if not self.comparable(other):
            raise ValueError("Objects cannot be added!")
        self.values += (other.values - self.pcount)
        return self

    def __eq__(self, other):
        if not self.comparable(other):
            raise ValueError("Objects cannot be compared to each other!")

        return np.allclose(self.values, other.values)


class Emissions(object):
    """
    External container object that represents the generalized emissions matrices for a model
    For each charge state and order, it contains a Econtainer object which is then used for
    keeping track of counts/probabilities.
    """

    __slots__ = ["n_states", "order", "min_z", "max_z", "emissions", "np_arrays"]

    def __init__(self,
                 n_states=-1,
                 order=0,
                 charge=(2,6),
                 emissions='ACDEFGHIKLMNPQRSTVWY '):

        self.n_states = n_states
        self.order = order
        self.min_z, self.max_z = charge
        self.emissions = {aa: index for (index, aa) in enumerate(emissions)}
        self.np_arrays = OffsetList(self.min_z)

        for z in range(self.min_z, self.max_z + 1):
            self.np_arrays.append(list())
            for o in range(self.order+1):
                self.np_arrays[z].append(Econtainer(n_states, order=o, emissions=emissions))

    def norm(self, order=None):
        """
        Normalizes the Emission matrix based on the last index, in other words emission probabilities. 
        @:param order: controls whether or not to normalize all orders (useful)
        """
        for z in range(self.min_z, self.max_z + 1):
            orders = range(self.order+1) if order is None else range(order, order+1)
            for o in orders:
                self.np_arrays[z][o].norm()

    def addPseudoCounts(self, alpha):
        """
        This method is used to weigh the prior knowledge. 
        Once this method is called the counts are translated to probabilities
        """

        def innerFold(lower, higher):
            for state in range(self.n_states):
                that = lower[state]
                this = higher[state]
                (conditionals, emissions) = np.shape(this)
                for i in range(conditionals):
                    for j in range(emissions):
                        this[i][j] += alpha * that[i // len(self.emissions)][j]

        for z in range(self.min_z, self.max_z + 1):
            for order in range(self.order):
                lower_order = self.np_arrays[z][order]
                lower_order.norm()
                innerFold(lower_order, self.np_arrays[z][order + 1])

        self.norm()

    def comparable(self, other):
        if not isinstance(other, Emissions):
            return False
        if self.order != other.order:
            return False
        if self.max_z != other.max_z or self.min_z != other.min_z:
            return False
        if self.n_states != other.n_states:
            return False
        if self.emissions != other.emissions:
            return False
        # if self.np_arrays != other.np_arrays:
        #     return False
        return True

    def __getitem__(self, y):
        if isinstance(y, int):
            return self.np_arrays[y]
        elif isinstance(y, tuple):
            if len(y) == 2:
                return self.np_arrays[y[0]][y[1]]
            else:
                return self.np_arrays[y[0]][y[1]][y[2:]]
        else:
            raise ValueError("Expected int or tuple, found {}".format(y))

    def __setitem__(self, i, y):
        self.np_arrays[i[0]][i[1]][i[2:]] = y

    def __add__(self, other):
        if not self.comparable(other):
            raise ValueError("Objects cannot be added")

        temp = copy.deepcopy(self)
        for z in range(self.min_z, self.max_z + 1):
            for o in range(self.order + 1):
                temp.np_arrays[z][o] += other.np_arrays[z][o]

        return temp

    def __iadd__(self, other):
        if not self.comparable(other):
            raise ValueError("Objects cannot be added")

        for z in range(self.min_z, self.max_z + 1):
            for o in range(self.order + 1):
                self.np_arrays[z][o] += other.np_arrays[z][o]

        return self

    def __eq__ (self, other):
        if not self.comparable(other):
            raise ValueError("Objects cannot be compared")

        for z in range(self.min_z, self.max_z + 1):
            for o in range(self.order + 1):
                if not self.np_arrays[z][o] == other.np_arrays[z][o]:
                    return False

        return True
