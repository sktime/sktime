import pandas as pd
from sktime.datasets.base import BaseDataset

class MyDataset(BaseDataset):
    """
    Create a sample dataset class directly subclassing BaseDataset.

    Demonstrate dataset creation for classification tasks without relying on internal mixins.
    """

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

    def load(self, split=None, *args, **kwargs):
        """
        Return a small synthetic dataset.

        Parameters
        ----------
        split : {"TRAIN", "TEST", None}, optional (default=None)
            Subset of data to return.

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
