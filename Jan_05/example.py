import argparse

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--l2", default=1., type=float, help="L2 regularization factor")
parser.add_argument("--learning_rate", default=0.1, type=float, help="Learning rate")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=57, type=int, help="Random seed")
parser.add_argument("--test_size", default=42, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.


# TODO: Create a gradient boosted trees on the classification training data.
#
# Notably, train for `args.trees` iteration. During iteration `t`:
# - the goal is to train `classes` regression trees, each predicting
#   raw weight for the corresponding class.
# - compute the current predictions `y_t(x_i)` for every training example `i` as
#     y_t(x_i)_c = \sum_{i=1}^t args.learning_rate * tree_{iter=i,class=c}.predict(x_i)
#     (note that y_0 is zero)
# - loss in iteration `t` is
#     L = (\sum_i NLL(target_i, softmax(y_{t-1}(x_i) + trees_to_train_in_iter_t.predict(x_i)))) +
#         1/2 * args.l2 * (sum of all node values in trees_to_train_in_iter_t)
# - for every class `c`:
#   - start by computing `g_i` and `h_i` for every training example `i`;
#     the `g_i` is the first derivative of NLL(target_i_c, softmax(y_{t-1}(x_i))_c)
#     with respect to y_{t-1}(x_i)_c, and the `h_i` is the second derivative of the same.
#   - then, create a decision tree minimizing the above loss L. According to the slides,
#     the optimum prediction for a given node T with training examples I_T is
#       w_T = - (\sum_{i \in I_T} g_i) / (args.l2 + sum_{i \in I_T} h_i)
#     and the value of the loss with the above prediction is
#       c_GB = - 1/2 (\sum_{i \in I_T} g_i)^2 / (args.l2 + sum_{i \in I_T} h_i)
#     which you should use as a splitting criterion.
#
# During tree construction, we split a node if:
# - its depth is less than `args.max_depth`
# - there is more than 1 example corresponding to it (this was covered by
#     a non-zero criterion value in the previous assignments)