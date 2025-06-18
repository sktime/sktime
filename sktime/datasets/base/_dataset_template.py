import pandas as pd

from sktime.datasets.base import BaseDataset, _DatasetFromLoaderMixin


def _my_loader(split=None):
    # Fake split
    X = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    y = pd.Series([0, 1, 0])

    if split == "TRAIN":
        return X.iloc[:2], y.iloc[:2]
    elif split == "TEST":
        return X.iloc[2:], y.iloc[2:]
    else:
        return X, y


class MyDataset(_DatasetFromLoaderMixin, BaseDataset):
    """Example dataset for demonstrating BaseDataset subclassing."""

    # Static reference to loader
    loader_func = staticmethod(_my_loader)

    def __init__(self):
        self.some_param = "unused"  # can add params here if loader needs
        super().__init__()
        self.set_tags(name="my_dataset", n_splits=1)
