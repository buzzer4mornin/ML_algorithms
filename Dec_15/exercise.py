import numpy as np

X = np.array([1, 7, 3, 3, 3, 5, 6, 7]).reshape(-1, 1)
y = np.array([0, 1, 2, 2, 2, 2, 3, 1])


def best_split(X, y):
    """Find the best split for a node.
    "Best" means that the average impurity of the two children, weighted by their
    population, is the smallest possible. Additionally it must be less than the
    impurity of the current node.
    To find the best split, we loop through all the features, and consider all the
    midpoints between adjacent training samples as possible thresholds. We compute
    the Gini impurity of the split generated by that particular feature/threshold
    pair, and return the pair with smallest impurity.
    Returns:
        best_idx: Index of the feature for best split, or None if no split is found.
        best_thr: Threshold to use for the split, or None if no split is found.
    """
    # Need at least two elements to split a node.
    if len(y) < 2:  # +++
        return None, None

    # Count of each class in the current node.
    num_parent = [np.sum(y == c) for c in range(4)]
    #print(num_parent)

    # Gini of current node.
    best_gini = 1.0 - sum((n / len(y)) ** 2 for n in num_parent)
    #best_gini = len(y) * sum((n / len(y)) * (1 - (n / len(y))) for n in num_parent)
    print("initial gini:", best_gini, "\n=====================")
    best_idx, best_thr = None, None

    # Loop through all features.
    for idx in range(1):
        # Sort data along selected feature.
        #thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
        thresholds, classes = zip(*sorted(zip(X[:, idx], y)))
        print(thresholds, classes)
        # We could actually split the node according to each feature/threshold pair
        # and count the resulting population for each class in the children, but
        # instead we compute them in an iterative fashion, making this for loop
        # linear rather than quadratic.
        num_left = [0] * 4
        num_right = num_parent.copy()
        for i in range(1, len(y)):  # possible split positions
            c = classes[i - 1]
            num_left[c] += 1
            num_right[c] -= 1
            gini_left = 1.0 - sum(
                (num_left[x] / i) ** 2 for x in range(4)
            )
            gini_right = 1.0 - sum(
                (num_right[x] / (len(y) - i)) ** 2 for x in range(4)
            )

            # The Gini impurity of a split is the weighted average of the Gini
            # impurity of the children.
            gini = (i * gini_left + (len(y) - i) * gini_right) / len(y)

            # The following condition is to make sure we don't try to split two
            # points with identical values for that feature, as it is impossible
            # (both have to end up on the same side of a split).
            if thresholds[i] == thresholds[i - 1]:
                continue

            if gini < best_gini:
                best_gini = gini
                best_idx = idx
                best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

    return best_idx, best_thr


a, b = best_split(X, y)

print("=====================\n", "split index vs point", a, b)