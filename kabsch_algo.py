#!/usr/bin/env python3
import click
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from svd import svd


def calculate_rmsd(X, Y):
    """Calculate Root-mean-square deviation from two sets of vectors X and Y.

    Args:
        X (np.ndarray): NxD matrix, where N is points and D is dimension.
        Y (np.ndarray): NxD matrix, where N is points and D is dimension.

    Returns:
        float: Root-mean-square deviation between the two structures.
    """

    result = np.sum((X - Y)**2, axis=1)
    result = np.mean(result)
    return np.sqrt(result)


def kabsch(X, Y):
    """Using the Kabsch algorithm with two sets of paired point X and Y.

    Args:
        X (np.ndarray): NxD matrix, where N is points and D is dimension.
        Y (np.ndarray): NxD matrix, where N is points and D is dimension.

    Returns:
        np.ndarray: Rotation matrix (D,D)
    """

    # Computation of the covariance matrix
    C = np.dot(np.transpose(X), Y)

    # Computation of the optimal rotation matrix see:
    # http://en.wikipedia.org/wiki/Kabsch_algorithm
    V, S, W = svd(C)
    d = (np.linalg.det(V) * np.linalg.det(W)) < 0.0

    if d:
        V[:, -1] = -V[:, -1]

    # Create Rotation matrix U
    U = np.dot(V, W)

    return U


@click.command()
@click.argument('first_arr', type=click.Path(exists=True))
@click.argument('second_arr', type=click.Path(exists=True))
def cli(first_arr, second_arr):
    """Calculate RMSD between structures and plots them.

    Args:
        first_arr (path): Path to first Numpy array .csv.
        second_arr (path): Path to second Numpy array .csv.
    """

    # Load arrays
    X = np.genfromtxt(first_arr, delimiter=',')
    Y = np.genfromtxt(second_arr, delimiter=',')

    # Translate X and Y to center
    X_ = X - X.mean(axis=0)
    Y_ = Y - Y.mean(axis=0)

    # Rotate X to match Y
    U = kabsch(X_, Y_)
    X_ = np.dot(X_, U)

    # Print and plot results
    print()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X_[:, 0], X_[:, 1], X_[:, 2], color='g', marker='o', linestyle='--', alpha=.5)
    ax.plot(Y_[:, 0], Y_[:, 1], Y_[:, 2], color='b', marker='^', linestyle=':', alpha=.5)
    ax.set_title("RMSD: {}".format(calculate_rmsd(X_, Y_)))
    plt.show()


if __name__ == "__main__":
    cli()
