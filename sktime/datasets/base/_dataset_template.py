import pandas as pd
from sktime.datasets.base import BaseDataset, _DatasetFromLoaderMixin


def _my_loader(split=None):
    X = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
    y = pd.Series([0, 1, 0], name="target")

    if split == "TRAIN":
        return X.iloc[:2], y.iloc[:2]
    elif split == "TEST":
        return X.iloc[2:], y.iloc[2:]
    elif split is None:
        return X, y
    else:
        raise ValueError(f"Invalid split '{split}'. Must be 'TRAIN', 'TEST', or None.")


class MyDataset(_DatasetFromLoaderMixin, BaseDataset):
    """Example dataset for demonstrating BaseDataset subclassing."""

    loader_func = staticmethod(_my_loader)

    def __init__(self):
        super().__init__()
        self.set_tags(name="my_dataset", n_splits=1)
        self.metadata = {
            "task": "classification",
            "n_instances": 3,
            "n_classes": 2,
            "univariate": False,
            "equal_length": True,
        }
