#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.utils.extmath import weighted_mode

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-4, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.


def kernel(args, x, y):
    return [(args.kernel_gamma * (np.dot(x, y)) + 1) ** args.kernel_degree if args.kernel == "poly"
            else np.exp(-1 * args.kernel_gamma * ((x - y) @ (x - y)))]

def smo(args, train_data, train_target, test_data, test_target,vote1,vote2, x_votes):
    # Create initial weights
    #print("Training classes {} and {}".format(vote1, vote2))

    def predict_train(row):
        sums = 0
        for i in range(len(train_data)):
            sums += a[i] * train_target[i] * train_kernels[row, i]
        return sums + b

    def predict_test(row):
        sums = 0
        for i in range(len(train_data)):
            sums += a[i] * train_target[i] * test_kernels[row, i]
        return sums + b

    def predict_votes(row):
        sums = 0
        for i in range(len(train_data)):
            sums += a[i] * train_target[i] * test_kernels_votes[row, i]
        return sums + b

    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    train_kernels = np.empty((len(train_data), len(train_data)), dtype=float)
    test_kernels = np.empty((len(test_data), len(train_data)), dtype=float)
    test_kernels_votes = np.empty((len(x_votes), len(train_data)), dtype=float)

    for x1, x1_ in enumerate(train_data):
        for x2, x2_ in enumerate(train_data):
            train_kernels[x1][x2] = kernel(args, x1_, x2_)[0]

    for x1, x1_ in enumerate(test_data):
        for x2, x2_ in enumerate(train_data):
            test_kernels[x1][x2] = kernel(args, x1_, x2_)[0]

    for x1, x1_ in enumerate(x_votes):
        for x2, x2_ in enumerate(train_data):
            test_kernels_votes[x1][x2] = kernel(args, x1_, x2_)[0]

    passes_without_as_changing = 0
    train_accs, test_accs = [], []

    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            j = j + (j >= i)

            E_i = predict_train(i) - train_target[i]
            E_j = predict_train(j) - train_target[j]

            if ((a[i] < (args.C - args.tolerance)) and (train_target[i] * E_i < -1 * args.tolerance)) or (a[i] > args.tolerance and (train_target[i] * E_i > args.tolerance)):
                d = 2 * train_kernels[i, j] - train_kernels[i, i] - train_kernels[j, j]
                # check if derivative > tol
                if d > -1 * args.tolerance:
                    continue
                else:
                    aj_ = a[j] - (train_target[j] * (E_i - E_j)) / d

                    # TODO: define range for [L,H] (both ti,tj equal or not)
                    if train_target[i] == train_target[j]:
                        L = max(0, a[i] + a[j] - args.C)
                        H = min(args.C, a[j] + a[i])
                    else:
                        L = max(0, a[j] - a[i])
                        H = min(args.C, args.C + a[j] - a[i])
                    # clip new aj
                    aj_ = np.clip(aj_, L, H)
                    if abs(aj_ - a[j]) < args.tolerance:
                        continue
                    else:
                        # ai, bias updates regarding KKT
                        ai_ = a[i] - train_target[i] * train_target[j] * (aj_ - a[j])
                        bi_ = b - E_i - train_target[i] * (ai_ - a[i]) * train_kernels[i, i] - train_target[j] * (aj_ - a[j]) * train_kernels[j, i]
                        bj_ = b - E_j - train_target[i] * (ai_ - a[i]) * train_kernels[i, j] - train_target[j] * (aj_ - a[j]) * train_kernels[j, j]

                        # new bias update
                        if args.tolerance < ai_ < (args.C - args.tolerance):
                            new_b = bi_
                        elif args.tolerance < aj_ < (args.C - args.tolerance):
                            new_b = bj_
                        else:
                            new_b = (bi_ + bj_) / 2
                        # updating ai,aj,b
                        a[i] = ai_
                        a[j] = aj_
                        b = new_b
                        as_changed += 1

        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.
        o = []
        for i in range(len(train_data)):
            my_predict = predict_train(i)
            if my_predict >= 0:
                o.append(1)
            else:
                o.append(-1)
        my = sklearn.metrics.accuracy_score(o, train_target)

        t = []
        for i in range(len(test_data)):
            my_predict_t = predict_test(i)
            if my_predict_t >= 0:
                t.append(1)
            else:
                t.append(-1)
        my_t = sklearn.metrics.accuracy_score(t, test_target)

        # TODO: VOTES VOTES VOTES VOTES VOTES VOTES
        votes = []
        for i in range(len(x_votes)):
            my_predict_votes = predict_votes(i)
            if my_predict_votes >= 0:
                votes.append(vote1)
            else:
                votes.append(vote2)


        train_accs.append(my)
        test_accs.append(my_t)

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        # PRINTS
        '''if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    print("Training finished after iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))'''

    return np.array(votes).reshape(-1,1)

def main(args):
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)


    #TODO: Fill in Array
    c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = [], [], [], [], [], [], [], [], [], []
    c0_t, c1_t, c2_t, c3_t, c4_t, c5_t, c6_t, c7_t, c8_t, c9_t = [], [], [], [], [], [], [], [], [], []

    for i in range(len(train_data)):
        if train_target[i] == 0:
            c0.append(train_data[i])
        elif train_target[i] == 1:
            c1.append(train_data[i])
        elif train_target[i] == 2:
            c2.append(train_data[i])
        elif train_target[i] == 3:
            c3.append(train_data[i])
        elif train_target[i] == 4:
            c4.append(train_data[i])
        elif train_target[i] == 5:
            c5.append(train_data[i])
        elif train_target[i] == 6:
            c6.append(train_data[i])
        elif train_target[i] == 7:
            c7.append(train_data[i])
        elif train_target[i] == 8:
            c8.append(train_data[i])
        else:
            c9.append(train_data[i])
    for i in range(len(test_data)):
        if test_target[i] == 0:
            c0_t.append(test_data[i])
        elif test_target[i] == 1:
            c1_t.append(test_data[i])
        elif test_target[i] == 2:
            c2_t.append(test_data[i])
        elif test_target[i] == 3:
            c3_t.append(test_data[i])
        elif test_target[i] == 4:
            c4_t.append(test_data[i])
        elif test_target[i] == 5:
            c5_t.append(test_data[i])
        elif test_target[i] == 6:
            c6_t.append(test_data[i])
        elif test_target[i] == 7:
            c7_t.append(test_data[i])
        elif test_target[i] == 8:
            c8_t.append(test_data[i])
        else:
            c9_t.append(test_data[i])
    '''c0, c1, c2, c3, c4, c5, c6, c7, c8, c9 = np.array(c0), np.array(c1), np.array(c2), np.array(c3), np.array(c4), np.array(c5), np.array(c6), np.array(c7), np.array(c8), np.array(c9)
    c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_, c9_ = np.full(len(c0), 0), np.full(len(c1), 1), np.full(len(c2), 2), np.full(len(c3), 3), \
                        np.full(len(c4), 4), np.full(len(c5), 5), np.full(len(c6), 6), np.full(len(c7), 7), \
                        np.full(len(c8), 8),np.full(len(c9), 9)'''
    '''i = 2
    j = 5
    res_x = np.vstack([xs[i], xs[j]])
    res_y = np.concatenate([ys[i], ys[j]])
    res_y = [1 if res_y[m]==2 else -1 for m in range(len(res_y))]'''
    xs = [np.array(c0), np.array(c1), np.array(c2), np.array(c3), np.array(c4), np.array(c5), np.array(c6), np.array(c7), np.array(c8), np.array(c9)]
    ys = [np.full(len(c0), 0), np.full(len(c1), 1), np.full(len(c2), 2), np.full(len(c3), 3), np.full(len(c4), 4), np.full(len(c5), 5), np.full(len(c6), 6), np.full(len(c7), 7), np.full(len(c8), 8),np.full(len(c9), 9)]

    xs_t = [np.array(c0_t), np.array(c1_t), np.array(c2_t), np.array(c3_t), np.array(c4_t), np.array(c5_t), np.array(c6_t), np.array(c7_t), np.array(c8_t), np.array(c9_t)]
    ys_t = [np.full(len(c0_t), 0), np.full(len(c1_t), 1), np.full(len(c2_t), 2), np.full(len(c3_t), 3), np.full(len(c4_t), 4), np.full(len(c5_t), 5), np.full(len(c6_t), 6), np.full(len(c7_t), 7), np.full(len(c8_t), 8), np.full(len(c9_t), 9)]


    empty = np.empty((len(test_target),1))
    result = np.empty((len(test_target),1))
    d = dict()
    for i in range(args.classes):
        d[i] = dict()
        for j in range(args.classes):
            if j <= i:
                continue
            d[i][j] = dict()

            res_x = np.vstack([xs[i], xs[j]])
            res_y = np.concatenate([ys[i], ys[j]])
            res_y = [1 if res_y[m] == i else -1 for m in range(len(res_y))]

            res_x_t = np.vstack([xs_t[i], xs_t[j]])
            res_y_t = np.concatenate([ys_t[i], ys_t[j]])
            res_y_t = [1 if res_y_t[m] == i else -1 for m in range(len(res_y_t))]

            arr = []
            for u in range(args.classes):
                arr.append(xs_t[u])

            # TODO: wrong x_votes, it should be unordered
            x_votes = np.vstack(arr)
            x_votes = test_data     #TODO: SO WE CHANGE IT
            votes = smo(args, res_x, res_y, res_x_t, res_y_t, i, j, x_votes)
            #print("----------------{}---------------".format(len(votes)))
            #print(votes)
            if i==0 and j==1:
                result = np.c_[empty, votes]
            else:
                result = np.c_[result, votes]
            #print(result.shape)
            #print(result)


    my_test = []
    for row in result:
        row = np.delete(row, 0)
        #print(row)
        select = weighted_mode(row, np.full(len(row), 1))
        my_test.append(int(select[0]))

    my_test = np.array(my_test)


    '''import collections
    k = collections.Counter(my_test)
    print(k)
    k = collections.Counter(test_target)
    print(k)'''

    test_accuracy = sklearn.metrics.accuracy_score(my_test, test_target)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
