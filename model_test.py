import unittest
import itertools
import hmm_model
import common_utils
import numpy as np
import numpy.testing as nptest


class MyTestCase(unittest.TestCase):
    def setUp(self):

        self.minifiles = [
            "../new_data/training/short0.tsv",
            "../new_data/training/short1.tsv",
            "../new_data/training/short2.tsv",
            "../new_data/training/short3.tsv"
        ]

        self.longerfiles = [
            "../new_data/training/training_set_0_top3.tsv",
            "../new_data/training/training_set_1_top3.tsv",
            "../new_data/training/training_set_2_top3.tsv",
            "../new_data/training/training_set_3_top3.tsv"
        ]

    """
    Checks to make sure training data is read completely, by comparing the lines read
    with the number of lines in the files
    """

    def test_model_traindata(self):
        file_gen = common_utils.yield_open(filenames=self.longerfiles)
        file_it = itertools.chain.from_iterable(file_gen)
        model = hmm_model.FragmentHMM(order=1, indata=file_it)

        import subprocess
        cmd = "cat ../new_data/training/training_set_{0..3}_top3.tsv | wc -l"
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode('utf-8')
        result = int(result.strip())

        self.assertEqual(model.nlines, result)

    """
    Checks for a case where different charge states had exactly the same counts
    while its possible, its VERY unlikely that different charge states would have the EXACT same values
    """

    def test_modelMatrices(self):
        file_gen = common_utils.yield_open(filenames=self.longerfiles)
        file_it = itertools.chain.from_iterable(file_gen)
        model = hmm_model.FragmentHMM(order=1, indata=file_it)

        for ion_type in 'yb':
            ematrices = model.E[ion_type].np_arrays
            tmatrix = model.T[ion_type]
            for z in range(model.min_z, model.max_z):
                self.assertNotEqual(ematrices[z], ematrices[z + 1])
                x = tmatrix[z]
                y = tmatrix[z + 1]
                self.assertFalse(np.allclose(x, y))

    """
    Checks if initializing the model with partial models gives the same matrices
    using mini data
    """

    def test_partialConstructor_mini(self):
        file_gen = common_utils.yield_open(filenames=self.minifiles)
        file_it = itertools.chain.from_iterable(file_gen)
        model = hmm_model.FragmentHMM(order=1, indata=file_it)

        # Running partial model tests serially, since multiprocessing fails due to some pickle issue
        partial_models = []
        for filename in self.minifiles:
            with open(filename, 'r') as f:
                partial = hmm_model.FragmentHMM(order=1, indata=f)
                partial_models.append(partial)
        final_model = hmm_model.FragmentHMM.from_partials(partial_models)

        self.assertTrue(model.comparable(final_model))
        self.assertEqual(model.nlines, final_model.nlines)
        self.assertEqual(model.pi, final_model.pi)

        for ion_type in 'yb':
            self.assertEqual(model.E[ion_type], final_model.E[ion_type])
            for z in range(model.min_z, model.max_z + 1):
                nptest.assert_allclose(model.T[ion_type][z], final_model.T[ion_type][z])

    """
    Checks if initializing the model with partial models gives the same matrices
    using larger training data
    """

    def test_partialConstructor_larger(self):

        file_gen = common_utils.yield_open(filenames=self.longerfiles)
        file_it = itertools.chain.from_iterable(file_gen)
        model = hmm_model.FragmentHMM(order=1, indata=file_it)

        # Running partial model tests serially, since multiprocessing fails due to some pickle issue
        partial_models = []
        for filename in self.longerfiles:
            with open(filename, 'r') as f:
                partial = hmm_model.FragmentHMM(order=1, indata=f)
                partial_models.append(partial)
        final_model = hmm_model.FragmentHMM.from_partials(partial_models)

        self.assertTrue(model.comparable(final_model))
        self.assertEqual(model.nlines, final_model.nlines)
        self.assertEqual(model.pi, final_model.pi)
        for ion_type in 'yb':
            self.assertEqual(model.E[ion_type], final_model.E[ion_type])
            for z in range(model.min_z, model.max_z + 1):
                nptest.assert_allclose(model.T[ion_type][z], final_model.T[ion_type][z])

    """
    Checks if initializing the model with partial models gives the same predictions
    as reading the entire training data into a single model. If the matrices give equality (tested separately)
    the predictions SHOULD be same as well. Equality is tested by means of correlation, but np.allclose could be
    an idea as well.
    """

    def test_predictionEquality(self):
        file_gen = common_utils.yield_open(filenames=self.longerfiles)
        file_it = itertools.chain.from_iterable(file_gen)
        model = hmm_model.FragmentHMM(order=1, indata=file_it)

        # Running partial model tests serially, since multiprocessing fails due to some pickle issue
        partial_models = []
        for filename in self.longerfiles:
            with open(filename, 'r') as f:
                partial = hmm_model.FragmentHMM(order=1, indata=f)
                partial_models.append(partial)
        final_model = hmm_model.FragmentHMM.from_partials(partial_models)

        import scipy.stats
        with open("../new_data/training/training_set_4_top3.tsv", 'r') as testfile:
            m1 = model.finalizeModel(200)
            m2 = final_model.finalizeModel(200)
            for line in testfile:
                try:
                    tokens = line.rstrip('\r\n').split('\t')
                    z, seq, y_ions, y_ints, b_ions, b_ints, y_frac = tokens
                    if y_ints == '' or int(z) < model.min_z:
                        continue
                    y_ints = [float(i) * float(y_frac) for i in y_ints.split(' ')]
                    y_ions = y_ions.split(' ')

                    spectra = {i: ii for i, ii in zip(y_ions, y_ints)}
                    ions1, probs1 = m1.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)
                    ions2, probs2 = m2.calc_fragments(charge=int(z), seq=seq, ion_type='y', use_yfrac=False)

                    r = scipy.stats.pearsonr(probs1, probs2)[0]
                    corr = r > 0.95

                    self.assertAlmostEqual(sum(probs1), 1,
                                           msg="predictions from model1 dont add up to 1: {}".format(sum(probs1)))
                    self.assertAlmostEqual(sum(probs2), 1,
                                           msg="predictions from model2 dont add up to 1: {}".format(sum(probs2)))
                    self.assertTrue(corr, msg="Predictions are not correlated for {}+{}, "
                                              "\n M1:{} \n M2:{}".format(seq, z, probs1, probs2))
                    self.assertTrue(np.allclose(probs1, probs2), msg="Predictions are not identical for {}+{}, "
                                                                     "\n M1:{} \n M2:{}".format(seq, z, probs1, probs2))

                except ValueError:
                    print(tokens)


if __name__ == '__main__':
    unittest.main()
