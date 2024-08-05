import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class AnomalyDetectionDataset:
    def __init__(
        self,
        data_split: str = "train",
        data_stride_len: int = 512,
        random_seed: int = 42,
    ):
        """
        Parameters
        ----------
        data_split : str
            Split of the dataset, 'train', or 'test'
        data_stride_len : int
            Stride length for the data.
        random_seed : int
            Random seed for reproducibility.
        """

        self.full_file_path_and_name = (
            "../data/198_UCR_Anomaly_tiltAPB2_50000_124159_124985.out"
        )
        self.series = "198_UCR_Anomaly_tiltAPB2_50000_124159_124985"
        self.data_split = data_split
        self.data_stride_len = data_stride_len
        self.random_seed = random_seed
        self.seq_len = 512

        # Downsampling for experiments. Refer
        # https://github.com/mononitogoswami/tsad-model-selection for more details
        self.downsampling_factor = 10
        self.min_length = (
            2560  # Minimum length of time-series after downsampling for experiments
        )

        # Read data
        self._read_data()

    def _get_borders(self):
        details = self.series.split("_")
        n_train = int(details[4])
        train_end = n_train
        test_start = train_end

        return slice(0, train_end), slice(test_start, None)

    def _read_data(self):
        self.scaler = StandardScaler()
        df = pd.read_csv(self.full_file_path_and_name)
        df.interpolate(inplace=True, method="cubic")

        self.length_timeseries = len(df)
        self.n_channels = 1
        labels = df.iloc[:, -1].values
        timeseries = df.iloc[:, 0].values.reshape(-1, 1)

        data_splits = self._get_borders()

        self.scaler.fit(timeseries[data_splits[0]])
        timeseries = self.scaler.transform(timeseries)
        timeseries = timeseries.squeeze()

        if self.data_split == "train":
            self.data, self.labels = timeseries[data_splits[0]], labels[data_splits[0]]
        elif self.data_split == "test":
            self.data, self.labels = timeseries[data_splits[1]], labels[data_splits[1]]

        self.length_timeseries = self.data.shape[0]

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if seq_end > self.length_timeseries:
            seq_start = self.length_timeseries - self.seq_len
            seq_end = None

        timeseries = self.data[seq_start:seq_end].reshape(
            (self.n_channels, self.seq_len)
        )
        labels = (
            self.labels[seq_start:seq_end]
            .astype(int)
            .reshape((self.n_channels, self.seq_len))
        )

        return timeseries, input_mask, labels

    def __len__(self):
        return (self.length_timeseries // self.data_stride_len) + 1
