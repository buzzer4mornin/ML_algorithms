from collections import Counter
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils.extmath import weighted_mode



def stablesoftmax(x):
    """Compute the softmax of vector x in a numerically stable way."""
    expZ = np.exp(x - np.max(x))
    return expZ / expZ.sum(axis=0, keepdims=True)


def minkowski_distance(a, b, p):
    distance = 0
    for d in range(len(a)-1):
        distance += abs(a[d] - b[d]) ** p
    distance = distance ** (1 / p)
    return distance


# Locate the most similar neighbors
def get_neighbors(train_data, train_target, test_row, k, p, weight_mode):

    d = list()
    for x in train_data:
        d.append(minkowski_distance(test_row, x, p))

    #print(len(d))
    #print(len(train_target))
    '''distances = list()
    for x, y in zip(train_data,train_target):
        dist = minkowski_distance(test_row, x, p)
        distances.append((y, dist))

    distances.sort(key=lambda tup: tup[1])'''

    neighbors = list()
    neighbors_w = list()
    for i in range(k):
        index = np.argmin(d)
        a = train_target.pop(index)
        b = d.pop(index)
        neighbors.append(a)
        neighbors_w.append(b)
        #neighbors.append(distances[i][0])
        #neighbors_w.append(distances[i][1])
    #print("\n")
    print("original:",neighbors_w)
    if weight_mode == "uniform":
        neighbors_w = [1 for _ in range(len(neighbors_w))]
    elif weight_mode == "inverse":
        neighbors_w = [1/k for k in neighbors_w]
    elif weight_mode == "softmax":
        neighbors_w = stablesoftmax(neighbors_w)

    print("weighted:", np.array(neighbors_w))
    return np.array(neighbors), np.array(neighbors_w)


# Test distance function
dataset = [[1.7810836, 4.550537003],
           [2.465489372, 3.362125076],
           [3.326561688, 4.200293529],
           [1.28807019, 1.550220317],
           [3.16407232, 3.405305973],
           [3.627531214, 2.259262235],
           [4.332441248, 1.088626775],
           [6.122596716, 1.17106367],
           [8.175418651, -0.222068655],
           [7.173756466, 3.108563011]]
# Test distance function
train_data = [[2.7810836, 2.550537003],
              [1.465489372, 2.362125076],
              [3.396561688, 4.400293529],
              [1.38807019, 1.850220317],
              [3.06407232, 3.005305973],
              [7.627531214, 2.759262235],
              [5.332441248, 2.088626775],
              [6.922596716, 1.77106367],
              [8.675418651, -0.242068655],
              [7.673756466, 3.508563011],
              [3.332441248, 3.088626775],
              [2.922596716, 7.77106367],
              [3.675418651, -5.242068655],
              ]
train_target = [0,
                2,
                0,
                3,
                3,
                1,
                0,
                2,
                1,
                1,
                2,
                2,
                2]



def predict_classification(train_data, train_target, test_row, num_neighbors, p, weight_mode):
    neighbors,neighbors_w = get_neighbors(train_data, train_target, test_row, num_neighbors, p , weight_mode)
    result = weighted_mode(neighbors, neighbors_w)
    prediction = int(result[0])
    return prediction






prediction = predict_classification(train_data,train_target,dataset[2], 4, 3, "inverse")
print("decision class: {}".format(prediction))
#print("\nprediction:", prediction)
#print('Expected %d, Got %d.' % (dataset[0][-1], prediction))