# -*- coding: utf-8 -*-
import glob
import math
from abc import ABC
from threading import Thread, Lock

from sktime.datasets import load_from_tsfile

class BaseExperiment(ABC):
    def __init__(
            self,
            experiment_name: str,
            dataset_path: str,
            result_path: str,
            n_threads: int = 1
    ):
        self.experiment_name = experiment_name
        self.dataset_path = dataset_path
        self.result_path = result_path
        self.n_threads = n_threads

    def run_experiment(self):
        global lock
        datasets = self._load_datasets()

        num_datasets = len(datasets)

        split = num_datasets / self.n_threads

        if (num_datasets % self.n_threads) != 0:
            split = math.floor(split)

        print(f"Beginning running experiment {self.experiment_name}")
        print(f"Result directory will be {self.result_path}")
        print(f"Loaded {num_datasets} datasets for experiment {self.experiment_name}")
        print(f"Using {self.n_threads} threads")

        threads = []

        for i in range(0, num_datasets, split):
            start = i
            end = i + split
            if end > num_datasets:
                end = num_datasets
            threads.append(Thread(
                target=threaded_experiment,
                args=[datasets[start:end], self]
            ))

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()
        return

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

    def _load_datasets(self):
        datasets = []
        ran_once = False
        for dataset in glob.iglob(self.dataset_path + "**/**/", recursive=True):
            if ran_once is False:
                ran_once = True
                continue
            test_dataset = None
            train_dataset = None
            for datafile in glob.iglob(dataset + "/*.ts", recursive=True):
                if datafile.find("TEST") != -1:
                    test_dataset = datafile
                elif datafile.find("TRAIN") != -1:
                    train_dataset = datafile

            if test_dataset is None or train_dataset is None:
                print(f"Skipping dataset {dataset} as cant fine both TRAIN and TEST")
            else:
                datasets.append([train_dataset, test_dataset, dataset])

        return datasets


def threaded_experiment(dataset_paths, experiment: BaseExperiment):

    result_path = experiment.result_path
    experiment_name = experiment.experiment_name


    for i in range(0, len(dataset_paths)):
        dataset = dataset_paths[i]
        train_path = dataset[0]
        test_path = dataset[1]
        dataset_path = dataset[2]

        X_train, y_train = load_from_tsfile(train_path)
        X_test, y_test = load_from_tsfile(test_path)
        if "\\" in dataset_path:
            dataset_name = dataset_path.split("\\")
        else:
            dataset_name = dataset_path.split("/")
        dataset_name = dataset_name[-2]
        experiment._run_experiment_for_dataset(X_train, y_train, X_test, y_test, dataset_name, result_path, experiment_name)
        print(f"finished running {dataset_name}")
