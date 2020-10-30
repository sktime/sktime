#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["PresplitFilesCV", "SingleSplit"]

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class PresplitFilesCV:
    """
    Cross-validation iterator over split predefined in files.

    This class is useful in orchestration where the train and test set
    is provided in separate files.
    """

    def __init__(self, cv=None):
        self.cv = cv

    def split(self, data, y=None, groups=None):
        """
        Split the data according to the train/test index.

        Parameters
        ----------
        data : pandas.DataFrame

        Yields
        ------
        train : ndarray
            Train indicies
        test : ndarray
            Test indices
        """
        # check input
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"Data must be pandas DataFrame, but found {type(data)}")
        if not np.all(data.index.unique().isin(["train", "test"])):
            raise ValueError(
                "Train-test split not properly defined in "
                "index of passed pandas DataFrame"
            )

        # this is a bit of a hack, PresplitFilesCV would need to talk to the
        # data loader during orchestration,
        # but since this is not possible with sklearn's interface, we load
        # both files separately, combined them
        # keep the training and test split in the index of the combined
        # dataframe, and split them here again
        n_instances = data.shape[0]
        idx = np.arange(n_instances)
        train = idx[data.index == "train"]
        test = idx[data.index == "test"]
        yield train, test

        # if additionally a cv iterator is provided, yield the predefined
        # split first, then reshuffle and apply cv,
        # note that test sets may overlap with the presplit file test set
        if self.cv is not None:
            for train, test in self.cv.split(idx, y=y):
                yield train, test

    def get_n_splits(self):
        n_splits = 1 if self.cv is None else 1 + self.cv.get_n_splits()
        return n_splits


class SingleSplit:
    """
    Helper class for orchestration that uses a single split for training and
    testing. Wrapper for sklearn.model_selection.train_test_split

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
        Allowed inputs are lists, numpy arrays, scipy-sparse
        matrices or pandas dataframes.
    test_size : float, int or None, optional (default=0.25)
        If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the
        absolute number of test samples. If None, the value is set to the
        complement of the train size. By default, the value is set to 0.25.
        The default will change in version 0.21. It will remain 0.25 only
        if ``train_size`` is unspecified, otherwise it will complement
        the specified ``train_size``.
    train_size : float, int, or None, (default=None)
        If float, should be between 0.0 and 1.0 and represent the
        proportion of the dataset to include in the train split. If
        int, represents the absolute number of train samples. If None,
        the value is automatically set to the complement of the test size.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    shuffle : boolean, optional (default=True)
        Whether or not to shuffle the data before splitting. If shuffle=False
        then stratify must be None.
    stratify : array-like or None (default=None)
        If not None, data is split in a stratified fashion, using this as
        the class labels.
    """

    def __init__(
        self,
        test_size=0.25,
        train_size=None,
        random_state=None,
        shuffle=True,
        stratify=None,
    ):
        self._test_size = test_size
        self._train_size = train_size
        self._random_state = random_state
        self._shuffle = shuffle
        self._stratify = stratify

    def split(self, data, y=None, groups=None):
        """
        Parameters
        ---------
        data : pandas dataframe
            data used for cross validation

        Returns
        -------
        tuple
            (train, test) indexes
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be provided as a pandas DataFrame")
        n_instances = data.shape[0]
        idx = np.arange(n_instances)

        yield train_test_split(
            idx,
            test_size=self._test_size,
            train_size=self._train_size,
            random_state=self._random_state,
            shuffle=self._shuffle,
            stratify=self._stratify,
        )

    @staticmethod
    def get_n_splits():
        return 1
