# -*- coding: utf-8 -*-
"""Data storage for benchmarking."""

__all__ = ["UEADataset", "RAMDataset", "make_datasets"]
__author__ = ["viktorkaz", "mloning"]

import os

import pandas as pd

from sktime.benchmarking.base import BaseDataset, HDDBaseDataset
from sktime.datasets import load_from_tsfile_to_dataframe


class UEADataset(HDDBaseDataset):
    """Represent a dataset in UEA/UCR format on the hard-drive."""

    def __init__(
        self,
        path,
        name,
        suffix_train="_TRAIN",
        suffix_test="_TEST",
        fmt=".ts",
        target_name="target",
    ):
        super(UEADataset, self).__init__(path, name)
        # create all the neccesary attributes for UAEDataset object
        # store a dataset

        self._target_name = target_name
        self._suffix_train = suffix_train
        self._suffix_test = suffix_test
        self._fmt = fmt

        # generate and validate file paths
        filename = os.path.join(self.path, self.name, self.name)
        self._train_path = filename + self._suffix_train + self._fmt
        self._validate_path(self._train_path)

        self._test_path = filename + self._suffix_test + self._fmt
        self._validate_path(self._test_path)

    def load(self):
        """Load dataset."""
        # load training and test set from separate files

        X_train, y_train = load_from_tsfile_to_dataframe(
            self._train_path, return_separate_X_and_y=True
        )
        X_test, y_test = load_from_tsfile_to_dataframe(
            self._test_path, return_separate_X_and_y=True
        )

        # combine into single dataframe
        data_train = pd.concat([X_train, pd.Series(y_train)], axis=1)
        data_test = pd.concat([X_test, pd.Series(y_test)], axis=1)

        # rename target variable
        data_train.rename(
            columns={data_train.columns[-1]: self._target_name}, inplace=True
        )
        data_test.rename(
            columns={data_test.columns[-1]: self._target_name}, inplace=True
        )

        # concatenate the two dataframes, keeping training and test split in
        # index, necessary for later optional CV
        data = pd.concat([data_train, data_test], axis=0, keys=["train", "test"])

        return data


class RAMDataset(BaseDataset):
    """Represent a dataset in RAM."""

    def __init__(self, dataset, name):
        """Container for storing a dataset in memory."""
        if not isinstance(dataset, pd.DataFrame):
            raise ValueError(
                f"Dataset must be pandas DataFrame, but found: " f"{type(dataset)}"
            )
        self._dataset = dataset
        super(RAMDataset, self).__init__(name=name)

    def load(self):
        """Load dataset."""
        return self._dataset


def make_datasets(path, dataset_cls, names=None, **kwargs):
    """Make datasets."""
    # check dataset class
    # if not isinstance(dataset_cls, BaseDataset):
    #     raise ValueError(f"dataset must inherit from BaseDataset, but found:"
    #                      f"{type(dataset_cls)}")

    # check dataset names
    if names is not None:
        if not isinstance(names, list):
            raise ValueError(f"names must be a list, but found: {type(names)}")
    else:
        names = os.listdir(path)  # get names if names is not specified

    # generate datasets
    datasets = [dataset_cls(path=path, name=name, **kwargs) for name in names]
    return datasets
