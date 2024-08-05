import numpy as np
from sklearn.preprocessing import StandardScaler

from momentfm.utils.data import load_from_tsfile


class ClassificationDataset:
    def __init__(self, data_split="train"):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', 'val' or 'test'.
        """

        self.seq_len = 512
        self.train_file_path_and_name = "../data/ECG5000_TRAIN.ts"
        self.test_file_path_and_name = "../data/ECG5000_TEST.ts"
        self.data_split = data_split  # 'train' or 'test'

        # Read data
        self._read_data()

    def _transform_labels(self, train_labels: np.ndarray, test_labels: np.ndarray):
        labels = np.unique(train_labels)  # Move the labels to {0, ..., L-1}
        transform = {}
        for i, l in enumerate(labels):
            transform[l] = i

        train_labels = np.vectorize(transform.get)(train_labels)
        test_labels = np.vectorize(transform.get)(test_labels)

        return train_labels, test_labels

    def __len__(self):
        return self.num_timeseries

    def _read_data(self):
        self.scaler = StandardScaler()

        self.train_data, self.train_labels = load_from_tsfile(
            self.train_file_path_and_name
        )
        self.test_data, self.test_labels = load_from_tsfile(
            self.test_file_path_and_name
        )

        self.train_labels, self.test_labels = self._transform_labels(
            self.train_labels, self.test_labels
        )

        if self.data_split == "train":
            self.data = self.train_data
            self.labels = self.train_labels
        else:
            self.data = self.test_data
            self.labels = self.test_labels

        self.num_timeseries = self.data.shape[0]
        self.len_timeseries = self.data.shape[2]

        self.data = self.data.reshape(-1, self.len_timeseries)
        self.scaler.fit(self.data)
        self.data = self.scaler.transform(self.data)
        self.data = self.data.reshape(self.num_timeseries, self.len_timeseries)

        self.data = self.data.T

    def __getitem__(self, index):
        assert index < self.__len__()

        timeseries = self.data[:, index]
        timeseries_len = len(timeseries)
        labels = self.labels[index,].astype(int)
        input_mask = np.ones(self.seq_len)
        input_mask[: self.seq_len - timeseries_len] = 0

        timeseries = np.pad(timeseries, (self.seq_len - timeseries_len, 0))

        return np.expand_dims(timeseries, axis=0), input_mask, labels
