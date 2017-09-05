# Inevitable Probability
This repository contains the code and data used to obtain the results reported in the following paper:

Orhan AE, Ma WJ (2017) [Efficient probabilistic inference in generic neural networks trained with non-probabilistic feedback.] (https://www.nature.com/articles/s41467-017-00181-8)*Nature Communications*, 8, 138.

The bulk of the code is written in [Theano](http://www.deeplearning.net/software/theano/) (0.8.2) + [Lasagne](http://lasagne.readthedocs.io/en/latest/) (0.2.dev1). Each folder contains code and data pertaining to a particular model type or experiment:

+`alt_objectives:` training with alternative objectives
+`ffwd:` experiments with feedforward nets
+`nin_nhu:` experiments measuring the efficiency of generic nets
+`random_ffwd:` experiments with random feedforward nets
+`recurrent_ei:` experiments with recurrent excitatory-inhibitory (EI) nets

The code for generating the results reported in Figure 9 (`qamar2013`) is written in Matlab and uses some routines from the Matlab Statistics and Optimization Toolboxes. Some of the files are meant to be run on a local computer cluster. You may need to modify them according to your needs.
