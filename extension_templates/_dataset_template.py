from sktime.datasets.classification._base import _ClassificationDatasetFromLoader

class MyDataset(_ClassificationDatasetFromLoader):
    """
    Fill-in template for a classification dataset extension.

    Replace class name, _tags, and loader_func with your dataset details.
    """

    _tags = {
        "name": "my_dataset",
        "n_splits": 1,
        "is_univariate": True,
        "n_instances": 3,
        "n_instances_train": 2,
        "n_instances_test": 1,
        "n_classes": 2,
    }

    loader_func = lambda: (X, y)
    def split_func():
        return (X_train, y_train), (X_test, y_test)
    
