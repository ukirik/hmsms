# HMSMS: HMM-based peptide fragmentation predictor
This document gives a basic introduction to the project and the ideas behind it. [TODO: add more info]

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

+---+----+----+----+----+----+----+----+----+----+----+----+-------+
|   | nt | nx | n3 | n2 | n1 | c* | c1 | c2 | c3 | cx | ct | total |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| A | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  |    | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| C | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| D | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| E | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| F | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| G | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| H | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| I | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| K | .  | .  | .  | .  | .  | *  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| L | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| M | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| N | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| P | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| Q | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| R | .  | .  | .  | .  | .  | *  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| S | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| T | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| U | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| V | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| W | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| Y | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+
| - | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | .  | 1.0   |
+---+----+----+----+----+----+----+----+----+----+----+----+-------+

The emission matrix is more difficult to visualize since in our case it would be fairly large as we have 20+ possible emissions from each state.
