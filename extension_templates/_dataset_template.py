import pandas as pd
from sktime.datasets.classification._base import BaseClassificationDataset

class MyDataset(BaseClassificationDataset):
    """
    Create a sample dataset class directly subclassing BaseClassificationDataset.

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

    def _load(self, *args):
        """
        Return a small synthetic dataset, supporting sktime dataset keys.

        Parameters
        ----------
        *args: tuple of strings that specify what to load
            Valid values: "X", "y", "X_train", "y_train", "X_test", "y_test", "cv"

        Returns
        -------
        dataset, if args is empty or length one
            data container corresponding to string in args
        tuple, of same length as args, if args is length 2 or longer
            data containers corresponding to strings in args, in same order
        """
        X = pd.DataFrame({"feature_1": [1, 2, 3], "feature_2": [4, 5, 6]})
        y = pd.Series([0, 1, 0], name="target")

        # train/test split
        X_train, y_train = X.iloc[:2], y.iloc[:2]
        X_test, y_test = X.iloc[2:], y.iloc[2:]

        key_map = {
            "X": X,
            "y": y,
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
        }

        if not args:
            return X, y
        result = []
        for key in args:
            if key in key_map:
                result.append(key_map[key])
            elif key == "cv":
                # Example: return a single split as a tuple
                result.append(((X_train, y_train), (X_test, y_test)))
            else:
                raise ValueError(f"Invalid key '{key}'. Must be one of {list(key_map.keys())} or 'cv'.")
        if len(result) == 1:
            return result[0]
        return tuple(result)
