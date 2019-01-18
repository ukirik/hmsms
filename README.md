# HMSMS: HMM-based peptide fragmentation predictor
This document gives a basic introduction to the project and the ideas behind it.

## HMM Basics
Markov models are probabilistic network models where the model is in a state $S$ at each time point $t$, and transitions from one state to the another (or possibly the same) iteratively in a stochastic manner. Specifically for Markov models, the probabilities for the next step $t_{i+1}$ is dependent on the last $n$ states, where $n$ denotes the _order_ of the system. So for a first order Markov model, $P(S_{i+1})$ is dependent only on $S_i$.

In [Hidden Markov Models (HMM)](https://en.wikipedia.org/wiki/Hidden_Markov_model) the states in which the model resides at a specific time point are hidden, or unavailable, but the _observations_ from each time point are available. As an example and for an in-depth introduction, see [this article](http://www.cs.sjsu.edu/~stamp/RUA/HMM.pdf).

## HMM applied to peptide fragmentation
Consider the fragmentation of a peptide in MS/MS setting, here's [an overview](http://www.matrixscience.com/help/fragmentation_help.html). If we model the fragmentation process as an HMM, then each fragment ion can be seen as a path in the system, such that the process goes from N-terminal of the peptide transitioning to the C-terminal, breaking on the backbone at a particular point.

If we define the states of the process such that:
 - each path starts at N-terminal (state **nt**)
 - break site between states **n1** and **c1**
 - all paths end at C-terminal (state **ct**)

In this setup the obvious corner cases are $b_1$ and $y_1$ ions, where **nt** = **n1** and **ct = c1** respectively.

The observations are the amino acids that are _emitted_ from each state. So for an example peptide sequence `HAPPIER` and fragmentation between the two prolines (i.e. $b_3$ and $y_4$ ions) the following emissions would have occurred:

- **nt** -> `H`
- **n2** -> `A`
- **n1** -> `P`
- **c1** -> `P`
- **c2** -> `I`
- **c3** -> `E`
- **ct** -> `R`

Each HMM is characterized by three matrices:
1. Transition matrix
2. Outcome/Emission matrix
3. Initial state probabilities

These matrices are described in detail below.

### 1. The transition matrix

This defines the transition probabilities from one state to the next. In this context, since we have a strictly linear flow of progress from N-terminal to the C-terminal, most of the values should be zeroes.


|       | nt | nx | n3 | n2 | n1 | c* | c1 | c2 | c3 | cx | ct | end |
|-------|----|----|----|----|----|----|----|----|----|----|----|-----|
| start | .  | 0  | .  | .  | .  | 0  | 0  | 0  | 0  | 0  | 0  | 0   |
| nt    | 0  | .  | .  | .  | .  | 0  | 0  | 0  | 0  | 0  | 0  | 0   |
| nx    | 0  | .  | .  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0   |
| n3    | 0  | 0  | 0  | .  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0   |
| n2    | 0  | 0  | 0  | 0  | .  | 0  | 0  | 0  | 0  | 0  | 0  | 0   |
| n1    | 0  | 0  | 0  | 0  | 0  | .  | .  | 0  | 0  | 0  | 0  | 0   |
| c*    | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | .   |
| c1    | 0  | 0  | 0  | 0  | 0  | 0  | 0  | .  | 0  | 0  | .  | 0   |
| c2    | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | .  | 0  | .  | 0   |
| c3    | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | .  | .  | 0   |
| cx    | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | .  | .  | 0   |
| ct    | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | 0  | .   |

**c*** is the special state where y-terminal fragment has only one amino acid, which is important since for tryptic peptides there will be an overrepresentation of K/R. Also worth noting that **start** is an initial distribution and **end** is an absorbing, or final, state thus there are no transitions from this state.


### 2. The emission matrix
The emission matrix defines the probabilities of each state emitting a particular outcome, as shown below:

|   | nt | nx | n3 | n2 | n1 | c* | c1 | c2 | c3 | cx | ct |
|---|----|----|----|----|----|----|----|----|----|----|----|
| A | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| C | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| D | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| E | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| F | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| G | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| H | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| I | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| K | .  | .  | .  | .  | .  | *  | .  | .  | .  | .  | .  |
| L | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| M | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| N | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| P | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| Q | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| R | .  | .  | .  | .  | .  | *  | .  | .  | .  | .  | .  |
| S | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| T | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| U | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| V | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| W | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| Y | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |
| **Σ** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** | **1.0** |

The emission matrix is more difficult to visualize since for several reasons:

- firstly and primarily, there are no strict limitations for emission probabilities, besides the fact that the sum of all emission probabilities has to add up to 1.0
- secondly, the emission matrix is also conditional on previous emissions for first order models and above. In the manuscript, we considered models of order 3 and 4, which expands the probability space (and memory consumption) exponentially.

### 3. How to use the code
The code provided essentially of proof-of-principle, in the sense that it's not entirely clean or optimized. That being said, they are functional and relatively easy to use. Below is the general workflow required to train a model:

1. Get some data to train the model on, and parse the fragment intensities. Currently we only provide a parser for working with Andromeda results, which should be familiar to MQ users. What you need is the msms.txt file from the MQ results. Consider how many output files you want to generate, this is useful for multithreading, as well as doing train-test-validate schemes:

       python hmsms/parseMQ.py -f msms.txt -v -n <nbr_of_output_files>

    If the msms.txt file is really large, you might want to filter and pre-sort the file to speed up parsing, and in that case use `--sorted` flag to indicate that the file is pre-sorted (according to andromeda score)

       python ../hmsms/parseMQ.py -f msms.txt -v -n <nbr_of_files> [--sorted]

    You can also specify which spectra to be extracted for each (peptide, charge state) pair, in case there are more than one available (majority of the peptides have more than one spectra) by using the `-s` flag: valid options are best scoring spectrum, median scoring spectrum, randomly chosen spectrum, or a composite average calculated from all available spectra for that (peptide, charge state)

        python ../hmsms/parseMQ.py -f msms.txt -v -n <nbr_of_files> [-s best|med|rand|ave]


2. Train a model using the files you generated at step 1.

       python ../hmsms/main.py -o <model_order> -i <training_data_files> -t <nbr_of_child_processes> -n <model_name>

3. From a python notebook or any other python script, load the model you generated at step 2. You can now get predictions for any (sequence, charge state) pair by calling the function

       model.calc_fragments(charge=int(z), seq=peptide_seq)


### 4. Potential improvements

There are several shortcomings of the current implementation which are listed below, in no particular order:

- Parallel processing is implemented using `multiprocessing` module, in order to bypass the limitations of GIL. In the current implementation, each child process takes a portion of the training data and creates a _partial_ model of the same order. These are then collapsed onto a final model by adding the counts (prior to normalising probabilities). Thus it is a trade-off between CPU time and memory consumption.

  Parallelisation could be done in a smarter way, using for ex. shared memory. This is left for a future release currently.

- For real-life use of this module, a rewrite in C or Fortran is likely to give a considerable performance boost, even though the numeric aspects of the code are all in Numpy.

- There are multiple python files for different types of comparisons, while functional as they are, there is significant overlap between these modules. A cleanup would be a nice improvement in due time. 
