import glob
import threading
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from sktime.datasets import load_from_tsfile


class BaseExperiment(ABC):

    def __init__(self, *, experiment_name: str, dataset_path: str, result_path: str):
        self.experiment_name = experiment_name
        self.dataset_path = dataset_path
        self.result_path = result_path

    def run_experiment(self):

        threads = []
        ran_once = False
        for dataset in glob.iglob(self.dataset_path + '**/**/', recursive=True):

            if ran_once is False:
                ran_once = True
                continue
            test_dataset = None
            train_dataset = None
            for datafile in glob.iglob(dataset + '/*.ts', recursive=True):
                if datafile.find('TEST') != -1:
                    test_dataset = datafile
                elif datafile.find('TRAIN') != -1:
                    train_dataset = datafile

            if test_dataset is None or train_dataset is None:
                print(f'Skipping dataset {dataset} as cant fine both TRAIN and TEST')
            else:
                X_train, y_train = load_from_tsfile(train_dataset)
                X_test, y_test = load_from_tsfile(test_dataset)
                if '\\' in dataset:
                    dataset_name = dataset.split('\\')
                else:
                    dataset_name = dataset.split('/')
                dataset_name = dataset_name[-2]
                print("Running for dataset:", dataset_name)
                thread = threading.Thread(name=dataset_name, target=self._run_experiment_for_dataset, args=[X_train, y_train, X_test, y_test, dataset_name])
                thread.start()
                threads.append(thread)
                # self._run_experiment_for_dataset()

        for thread in threads:
            thread.join()

    @abstractmethod
    def _run_experiment_for_dataset(
            self,
            X_train: pd.DataFrame,
            y_train: np.ndarray,
            X_test: pd.DataFrame,
            y_test: np.ndarray,
            dataset_name: str
    ):
        ...

    def run_test_experiment(self):
        print("Starting test")
        walk_dir = self.dataset_path

        ran_once = False
        for dataset in glob.iglob(self.dataset_path + '**/**/', recursive=True):
            if ran_once is False:
                ran_once = True
                continue
            test_dataset = None
            train_dataset = None
            for datafile in glob.iglob(dataset + '/*.ts', recursive=True):
                if datafile.find('TEST') != -1:
                    test_dataset = datafile
                elif datafile.find('TRAIN') != -1:
                    train_dataset = datafile
                else:
                    joe = ''
            joe = ''

            if test_dataset is None or train_dataset is None:
                print(f'Skipping dataset {dataset} as cant fine both TRAIN and TEST')
            else:
                X_train, y_train = load_from_tsfile(train_dataset)
                X_test, y_test = load_from_tsfile(test_dataset)
                if '\\' in dataset:
                    dataset_name = dataset.split('\\')
                else:
                    dataset_name = dataset.split('/')
                dataset_name = dataset_name[-2]
                self._run_experiment_for_dataset(X_train, y_train, X_test, y_test, dataset_name)
                break
        print("Experiment end")



    def _read_whole_file(self, file_path: str):
        """Method that returns the file in an array line by line

        Parameters
        ----------
        file_path: str
            Path to file to read

        Returns
        -------
        list
            Each item is a line from the file.
        """
        file_data = []
        with open(file_path) as f:
            for line in f:
                file_data.append(line)
        return file_data
