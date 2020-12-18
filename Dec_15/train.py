# TODO: Create a random forest on the trainining data.
#
# For determinism, create a generator
#   generator = np.random.RandomState(args.seed)
# at the beginning and then use this instance for all random number generation.
#
# Use a simplified decision tree from the `decision_tree` assignment:
# - use `entropy` as the criterion
# - use `max_depth` constraint, so split a node only if:
#   - its depth is less than `args.max_depth`
#   - the criterion is not 0 (the corresponding instance targetsare not the same)
# When splitting nodes, proceed in the depth-first order, splitting all nodes
# in left subtrees before nodes in right subtrees.
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# Additionally, implement:
# - feature subsampling: when searching for the best split, try only
#   a subset of features. When splitting a node, start by generating
#   a feature mask using
#     generator.uniform(size=number_of_features) <= feature_subsampling
#   which gives a boolean value for every feature, with `True` meaning the
#   feature is used during best split search, and `False` it is not.
#   (When feature_subsampling == 1, all features are used, but the mask
#   should still be generated.)
#
# - train a random forest consisting of `args.trees` decision trees
#
# - if `args.bootstrapping` is set, right before training a decision tree,
#   create a bootstrap sample of the training data using the following indices
#     indices = generator.choice(len(train_data), size=len(train_data))
#   and if `args.bootstrapping` is not set, use the original training data.
#
# During prediction, use voting to find the most frequent class for a given
# input, choosing the one with smallest class index in case of a tie.
import itertools

import numpy as np
