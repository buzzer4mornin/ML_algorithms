#!/usr/bin/env python3
import argparse

import numpy as np

import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--clusters", default=3, type=int, help="Number of clusters")
parser.add_argument("--examples", default=200, type=int, help="Number of examples")
parser.add_argument("--init", default="random", type=str, help="Initialization (random/kmeans++)")
parser.add_argument("--iterations", default=20, type=int, help="Number of kmeans iterations to perfom")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

def plot(args, iteration, data, centers, clusters):
    import matplotlib.pyplot as plt

    if args.plot is not True:
        if not plt.gcf().get_axes(): plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Initialization {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
    if args.plot is True: plt.show()
    else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

def main(args):
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate artificial data
    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    # TODO: Initialize `centers` to be
    # - if args.init == "random", K random data points, using the indices
    #   returned by
    #     generator.choice(len(data), size=args.clusters, replace=False)
    # - if args.init == "kmeans++", generate the first cluster by
    #     generator.randint(len(data))
    #   and then iteratively sample the rest of the clusters proportionally to
    #   the square of their distances to their closest cluster using
    #     generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))
    #   Use the `np.linalg.norm` to measure the distances.
    if args.init == "random":
        indices = generator.choice(len(data), size=args.clusters, replace=False)
    if args.init == "kmeans++":
        indices = [generator.randint(len(data))]
        unselected = np.arange(len(data))
        for i in range(args.clusters - 1):
            unselected = unselected[unselected != indices[-1]]
            square_distances = np.min(np.linalg.norm(np.expand_dims(data[unselected], 1) - data[indices], axis=-1) ** 2, axis=-1)
            indices.append(generator.choice(unselected, p=square_distances / np.sum(square_distances)))
    centers = data[indices].copy()

    if args.plot:
        plot(args, 0, data, centers, clusters=None)

    # Run `args.iterations` of the K-Means algorithm.
    for iteration in range(args.iterations):
        # TODO: Perform a single iteration of the K-Means algorithm, storing
        # zero-based cluster assignment to `clusters`.

        # Find the nearest centers
        distances = np.linalg.norm(np.expand_dims(data, 1) - centers, axis=-1)
        clusters = np.argmin(distances, axis=-1)
        assert all(np.sum(distances + 1e-6 >= np.max(distances, keepdims=True, axis=-1), axis=-1) == 1)

        # Update cluster centers
        for c in range(args.clusters):
                centers[c] = np.mean(data[clusters == c], axis=0)

        if args.plot:
            plot(args, 1 + iteration, data, centers, clusters)

    return clusters

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    centers = main(args)
    print("Cluster assignments:", centers, sep="\n")