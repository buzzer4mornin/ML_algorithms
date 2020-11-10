#!/usr/bin/env python3
import argparse
import sys

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--hidden_layer", default=50, type=int, help="Hidden layer size")
parser.add_argument("--iterations", default=10, type=int, help="Number of iterations over the data")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.



def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, stratify=target, test_size=args.test_size, random_state=args.seed)

    # One Hot Encoding
    xx = np.array(train_target)
    train_target = np.zeros((xx.size, xx.max() + 1))
    train_target[np.arange(xx.size), xx] = 1
    xx = np.array(test_target)
    test_target = np.zeros((xx.size, xx.max() + 1))
    test_target[np.arange(xx.size), xx] = 1

    # Generate initial model weights
    weights = [generator.uniform(size=[train_data.shape[1], args.hidden_layer], low=-0.1, high=0.1),
               generator.uniform(size=[args.hidden_layer, args.classes], low=-0.1, high=0.1)]
    biases = [np.zeros(args.hidden_layer), np.zeros(args.classes)]


    def dRelu(x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def relu(x):
        return max(0.0, x)

    def stablesoftmax(x):
        """Compute the softmax of vector x in a numerically stable way."""
        expZ = np.exp(x - np.max(x))
        return expZ / expZ.sum(axis=0, keepdims=True)


    '''def compute_multiclass_loss(Y, Y_hat):  # Y -> actual, Y_hat -> predicted
        L_sum = np.sum(np.multiply(Y, np.log(Y_hat)))
        m = Y.shape[1]
        L = -(1 / m) * L_sum
        L = np.squeeze(L)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17). #https://stackoverflow.com/questions/50337332/backpropagation-for-sigmoid-activation-and-softmax-output
        assert (L.shape == ())
        return L'''



    def forward(inputs):
        # TODO: Implement forward propagation, returning *both* the value of the hidden
        # layer and the value of the output layer.

        layers = np.array([inputs, np.zeros(weights[1].shape[1]), np.zeros(weights[0].shape[1])])

        # We assume a neural network with a single hidden layer of size `args.hidden_layer`
        # and ReLU activation, where ReLU(x) = max(x, 0), and an output layer with softmax
        # activation.
        #
        # The value of the hidden layer is computed as ReLU(inputs @ weights[0] + biases[0]).
        # The value of the output layer is computed as softmax(hidden_layer @ weights[1] + biases[1]).
        layers[1] = np.dot(inputs, weights[0]) + biases[0]              # FeedForward on Hidden Layer
        hidden_in = layers[1]
        layers[1] = np.array([relu(x) for x in layers[1]])              # Activation() on Hidden Layer

        layers[2] = np.dot(layers[1], weights[1]) + biases[1]           # FeedForward on Output Layer + Activation()
        output_in = layers[2]
        layers[2] = stablesoftmax(layers[2])
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you can use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.
        return layers[1], layers[2], hidden_in, output_in



    for iteration in range(args.iterations):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`.
        gradient_components = 0

        f_dcost_wh = np.zeros((weights[0].shape[0], weights[0].shape[1]))
        f_dcost_bh = np.zeros((biases[0].shape[0], ))
        f_dcost_wo = np.zeros((weights[1].shape[0], weights[1].shape[1]))
        f_dcost_bo = np.zeros((biases[1].shape[0],))

        for i in permutation:
            hidden_out, output_out, hidden_in, output_in = forward(train_data[i])

            """Propogate OUTPUT layer weights"""
            dcost_dzo = output_out - train_target[i]
            dzo_dwo = hidden_out
            # Reshape
            dcost_dzo = dcost_dzo.reshape(dcost_dzo.shape[0], 1)
            dzo_dwo = dzo_dwo.reshape(dzo_dwo.shape[0], 1)
            # Gradients <-----
            dcost_wo = np.dot(dzo_dwo, dcost_dzo.T)
            dcost_bo = dcost_dzo
            dcost_bo = dcost_bo.reshape(dcost_bo.shape[0], )
            # towards avg_update  <=====================================
            f_dcost_wo += dcost_wo
            f_dcost_bo += dcost_bo

            """Propogate HIDDEN layer weights"""
            #dcost_dw = dcost_dah * dah_dzh * dzh_dwh
            dzo_dah = weights[1]
            dcost_dah = np.dot(dzo_dah, dcost_dzo)
            dah_dzh = dRelu(np.array(hidden_in))
            dzh_dwh = train_data[i]
            # Reshape
            dah_dzh = dah_dzh.reshape(dah_dzh.shape[0], 1)
            dzh_dwh = dzh_dwh.reshape(dzh_dwh.shape[0], 1)
            # Gradients <-----
            dcost_wh = np.dot(dzh_dwh, (dah_dzh * dcost_dah).T)
            dcost_bh = dcost_dah * dah_dzh
            dcost_bh = dcost_bh.reshape(dcost_bh.shape[0], )
            # towards avg_update  <=====================================
            f_dcost_wh += dcost_wh
            f_dcost_bh += dcost_bh

            gradient_components += 1
            if gradient_components == args.batch_size:
                """UPDATE weights"""
                # Update Weights <========
                weights[0] = weights[0] - args.learning_rate * f_dcost_wh / gradient_components
                biases[0] = biases[0] - args.learning_rate * f_dcost_bh / gradient_components

                weights[1] = weights[1] - args.learning_rate * f_dcost_wo / gradient_components
                biases[1] = biases[1] - args.learning_rate * f_dcost_bo / gradient_components

                f_dcost_wh = np.zeros((weights[0].shape[0], weights[0].shape[1]))
                f_dcost_bh = np.zeros((biases[0].shape[0],))
                f_dcost_wo = np.zeros((weights[1].shape[0], weights[1].shape[1]))
                f_dcost_bo = np.zeros((biases[1].shape[0],))
                gradient_components = 0
        assert gradient_components == 0

        # TODO: After the SGD iteration, measure the accuracy for both the
        # train test and the test set and print it in percentages.
        train_accuracy, test_accuracy = 0, 0


        for i in range(train_data.shape[0]):
            train_accuracy += np.argmax(forward(train_data[i])[1]) == np.argmax(train_target[i])
        train_accuracy /= train_data.shape[0]


        for i in range(test_data.shape[0]):
            test_accuracy += np.argmax(forward(test_data[i])[1]) == np.argmax(test_target[i])
        test_accuracy /= test_data.shape[0]


        print("After iteration {}: train acc {:.1f}%, test acc {:.1f}%".format(
            iteration + 1, 100 * train_accuracy, 100 * test_accuracy))

    return tuple(weights + biases)

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    parameters = main(args)
    print("Learned parameters:", *(" ".join([" "] + ["{:.2f}".format(w) for w in ws.ravel()[:20]] + ["..."]) for ws in parameters), sep="\n")