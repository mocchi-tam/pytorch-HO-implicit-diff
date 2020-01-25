## Hyperparameter Optimization by implicit-diff in Pytorch ##

This is a hyperparameter optimization technique, and I personally practiced implementation in pytorch.
Hyper-gradients can be computed efficiently using the implicit function theorem and Neumann series expansion. See the author's paper for details.

See the paper: https://arxiv.org/abs/1911.02590

----------

For an Iris dataset classification task in a three-layer neural network, I followed an optimization method using adaptive weight-decays as hyperparameters.
