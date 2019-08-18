__all__ = ["DatasetUEA", "DatasetCollectionUEA", "DatasetRAM", "DatasetHDD"]
__author__ = ["Viktor Kazakov", "Markus Löning"]

import os
import pandas as pd
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.benchmarking.base import BaseDataset


class DatasetUEA(BaseDataset):

    def __init__(self, path, name, suffix_train="_TRAIN.ts",
                 suffix_test="_TEST.ts", target="target"):
        self.target = target
        self._suffix_train = suffix_train
        self._suffix_test = suffix_test

        super(DatasetUEA, self).__init__(path, name)

    def load(self):
        """Load dataset"""

        # load training and test set from separate files
        train_path = os.path.join(self.path, self.name + self._suffix_train)
        test_path = os.path.join(self.path, self.name + self._suffix_test)
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


class DatasetRAM(BaseDataset):
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
        self._dataset_name = name

    @property
    def dataset_name(self):
        """
        Returns
        -------
        str
            Name of the dataset
        """
        return self._dataset_name

    def load(self):
        """
        Returns
        -------
        pandas DataFrame
            dataset in pandas DataFrame format
        """
        return self._dataset


class DatasetHDD(BaseDataset):
    pass


class DatasetCollectionUEA:

    def __init__(self, path, dataset_names=None):
        self.path = path
        self.dataset_names = os.listdir(self.path) if dataset_names is None else dataset_names

    def generate_dataset_hooks(self):
        """Generate dataset hooks"""
        datasets = []
        for dataset_name in self.dataset_names:
            path = os.path.join(self.path, dataset_name)
            dataset = DatasetUEA(path=path, name=dataset_name)
            datasets.append(dataset)
        return datasets

