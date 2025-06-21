from sktime.datasets.base._dataset_templates import MyDataset


def test_my_dataset():
    ds = MyDataset()

    # Test basic full load
    X, y = ds.load()
    assert len(X) == 3 and len(y) == 3, "Full dataset should have 3 instances"
    assert X.shape[1] == 2, "Dataset should have 2 features"
    assert y.name == "target", "Target column should be named 'target'"

    # Test split load
    X_train, y_train, X_test, y_test = ds.load(
        split=["X_train", "y_train", "X_test", "y_test"]
    )

    assert len(X_train) == 2, "Train split should have 2 instances"
    assert len(X_test) == 1, "Test split should have 1 instance"
    assert len(y_train) == 2 and len(y_test) == 1, "Target splits must match inputs"

    # Check types
    assert isinstance(X_train, type(X)), "X_train should match the type of X"
    assert isinstance(y_train, type(y)), "y_train should match the type of y"
