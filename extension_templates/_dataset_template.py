import pandas as pd
from sktime.datasets.base import BaseDataset, _DatasetFromLoaderMixin

def _my_loader(split=None):
    """
    A simple dataset loader function returning a small tabular dataset.

    Parameters
    ----------
    split : {"TRAIN", "TEST", None}, optional (default=None)
        Subset of data to return. If "TRAIN", returns the training portion.
        If "TEST", returns the testing portion. If None, returns the full dataset.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target labels.
    """
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
    """
    A sample dataset class demonstrating use of _DatasetFromLoaderMixin with BaseDataset.

    This class wraps the `_my_loader` function and adds dataset metadata,
    conforming to the sktime extension pattern for datasets.

    Attributes
    ----------
    loader_func : function
        Static reference to the dataset loading function.
    metadata : dict
        Dictionary containing information about the dataset such as task type,
        number of instances, number of classes, and data characteristics.
    """

    loader_func = staticmethod(_my_loader)

    def __init__(self):
        """Initializes the dataset metadata and sets dataset tags."""
        super().__init__()
        self.set_tags(name="my_dataset", n_splits=1)
        self.metadata = {
            "task": "classification",
            "n_instances": 3,
            "n_classes": 2,
            "univariate": False,
            "equal_length": True,
        }
