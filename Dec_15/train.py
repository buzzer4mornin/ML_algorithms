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
class Node:
    def __init__(self,node_score, time, gini_or_entropy, num_samples, num_samples_per_class, predicted_class , name):
        self.node_score = node_score
        self.time=time
        self.gini_or_entropy = gini_or_entropy
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None
        self.name=name


    def __lt__(self, other):
        return self.time < other.time if self.node_score == other.node_score else self.node_score > other.node_score


mynode = Node(95, 10, 5, 100, 6, 0,  "so")
mynode_2 = Node(66, 3, 10, 200, 4, 1, "now")
mynode_3 = Node(95, 4, 15, 300, 2, 2, "then")


import heapq as hq
h = []
'''hq.heappush(h, (5, 6))
hq.heappush(h, (7, 1))
hq.heappush(h, (1, 3))
hq.heappush(h, (3, 5))
y = hq.heappop(h)
print(y)
'''

hq.heappush(h, mynode)
hq.heappush(h, mynode_2)
hq.heappush(h, mynode_3)

#y = hq.heappop(h)
print(h,)
