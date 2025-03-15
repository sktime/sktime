"""Functions to sample sktime datasets.

Used in experiments to get deterministic resamples.
"""

import random

import numpy as np
import pandas as pd
import sklearn.utils


def stratified_resample(X_train, y_train, X_test, y_test, random_state):
    """Stratified resample data without replacement using a random state.

    Reproducible resampling. Combines train and test, resamples to get the same class
    distribution, then returns new train and test.

    Parameters
    ----------
    X_train : pd.DataFrame
        train data attributes in sktime pandas format.
    y_train : np.array
        train data class labels.
    X_test : pd.DataFrame
        test data attributes in sktime pandas format.
    y_test : np.array
        test data class labels as np array.
    random_state : int
        seed to enable reproducible resamples

    Returns
    -------
    new train and test attributes and class labels.
    """
    all_labels = np.concatenate((y_train, y_test), axis=None)
    all_data = pd.concat([X_train, X_test])
    random_state = sklearn.utils.check_random_state(random_state)
    # count class occurrences
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    assert list(unique_train) == list(
        unique_test
    )  # haven't built functionality to deal with classes that exist in
    # test but not in train
    # prepare outputs
    X_train = pd.DataFrame()
    y_train = np.array([])
    X_test = pd.DataFrame()
    y_test = np.array([])
    # for each class
    for label_index in range(0, len(unique_train)):
        # derive how many instances of this class from the counts
        num_instances = counts_train[label_index]
        # get the indices of all instances with this class label
        label = unique_train[label_index]
        indices = np.where(all_labels == label)[0]
        # shuffle them
        random_state.shuffle(indices)
        # take the first lot of instances for train, remainder for test
        train_indices = indices[0:num_instances]
        test_indices = indices[num_instances:]
        del indices  # just to make sure it's not used!
        # extract data from corresponding indices
        train_instances = all_data.iloc[train_indices, :]
        test_instances = all_data.iloc[test_indices, :]
        train_labels = all_labels[train_indices]
        test_labels = all_labels[test_indices]
        # concat onto current data from previous loop iterations
        X_train = pd.concat([X_train, train_instances])
        X_test = pd.concat([X_test, test_instances])
        y_train = np.concatenate([y_train, train_labels], axis=None)
        y_test = np.concatenate([y_test, test_labels], axis=None)
    # reset indexes to conform to sktime format.
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    return X_train, y_train, X_test, y_test


def random_partition(n, k=2, seed=42):
    """Construct a uniformly random partition, iloc reference.

    Parameters
    ----------
    n : int
        size of set to partition
    k : int, optional, default=2
        number of sets to partition into
    seed : int
        random seed, used in random.shuffle

    Returns
    -------
    parts : list of list of int
        elements of `parts` are lists of iloc int indices between 0 and n-1
        elements of `parts` are of length floor(n / k) or ceil(n / k)
        elements of `parts`, as sets, are disjoint partition of [0, ..., n-1]
        elements of elements of `parts` are in no particular order
        `parts` is sampled uniformly at random, subject to the above properties
    """
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)

    parts = []
    for i in range(k):
        d = round(len(idx) / (k - i))
        parts += [idx[:d]]
        idx = idx[d:]

    return parts
