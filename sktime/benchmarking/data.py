__all__ = ["UEADataset", "RAMDataset", "make_datasets"]
__author__ = ["Viktor Kazakov", "Markus Löning"]

import os

import pandas as pd

from sktime.benchmarking.base import BaseDataset, HDDBaseDataset
from sktime.utils.load_data import load_from_tsfile_to_dataframe


class UEADataset(HDDBaseDataset):

    def __init__(self, path, name, suffix_train="_TRAIN",
                 suffix_test="_TEST", fmt=".ts", target="target"):
        self.target = target
        self._suffix_train = suffix_train
        self._suffix_test = suffix_test
        self._fmt = fmt

        super(UEADataset, self).__init__(path, name)

    def load(self):
        """Load dataset"""

        # load training and test set from separate files
        filename = os.path.join(self.path, self.name, self.name)
        train_path = filename + self._suffix_train + self._fmt
        test_path = filename + self._suffix_test + self._fmt
        X_train, y_train = load_from_tsfile_to_dataframe(train_path, return_separate_X_and_y=True)
        X_test, y_test = load_from_tsfile_to_dataframe(test_path, return_separate_X_and_y=True)

        # combine into single dataframe
        data_train = pd.concat([X_train, pd.Series(y_train)], axis=1)
        data_test = pd.concat([X_test, pd.Series(y_test)], axis=1)

        # rename target variable
        data_train.rename(columns={data_train.columns[-1]: self.target}, inplace=True)
        data_test.rename(columns={data_test.columns[-1]: self.target}, inplace=True)

        # concatenate the two dataframes, keeping training and test split in index, necessary for later optional CV
        data = pd.concat([data_train, data_test], axis=0, keys=["train", "test"]).reset_index(level=1, drop=True)

        return data


class RAMDataset(BaseDataset):

    def __init__(self, dataset, name):
        """
        Container for storing a dataset in memory

        Paramteters
        -----------
        dataset : pandas DataFrame
            The actual dataset
        dataset_name : str
            Name of the dataset
        """

        self._dataset = dataset
        super(RAMDataset, self).__init__(name=name)

    def load(self):
        """
        Returns
        -------
        pandas DataFrame
            dataset in pandas DataFrame format
        """
        return self._dataset


def make_datasets(path, dataset_cls, names=None, **kwargs):
    """Helper function to make datasets"""
    # check input format
    # if not isinstance(dataset_cls, BaseDataset):
    #     raise ValueError(f"dataset must inherit from BaseDataset, but found:"
    #                      f"{type(dataset_cls)}")
    if names is not None:
        if not isinstance(names, list):
            raise ValueError(f"names must be a list, but found: {type(names)}")

    # get names if names is not specified
    names = os.listdir(path) if names is None else names
    return [dataset_cls(path=path, name=name, **kwargs) for name in names]
