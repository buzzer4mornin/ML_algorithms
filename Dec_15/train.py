# TODO: Create a decision tree on the trainining data.
#
# - For each node, predict the most frequent class (and the one with
#   smallest index if there are several such classes).
#
# - When splitting a node, consider the features in sequential order, then
#   for each feature consider all possible split points ordered in ascending
#   value, and perform the first encountered split descreasing the criterion
#   the most. Each split point is an average of two nearest unique feature values
#   of the instances corresponding to the given node (i.e., for four instances
#   with values 1, 7, 3, 3 the split points are 2 and 5).
#
# - Allow splitting a node only if:
#   - when `args.max_depth` is not None, its depth must be less than `args.max_depth`;
#     depth of the root node is zero;
#   - there are at least `args.min_to_split` corresponding instances;
#   - the criterion value is not zero.
#
# - When `args.max_leaves` is None, use recursive (left descendants first, then
#   right descendants) approach, splitting every node if the constraints are valid.
#   Otherwise (when `args.max_leaves` is not None), always split a node where the
#   constraints are valid and the overall criterion value (c_left + c_right - c_node)
#   decreases the most. If there are several such nodes, choose the one
#   which was created sooner (a left child is considered to be created
#   before a right child).