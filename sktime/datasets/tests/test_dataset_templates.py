from sktime.datasets._dataset_templates import MyDataset


def test_my_dataset():
    ds = MyDataset()

    # Test basic load
    X, y = ds.load()
    assert len(X) == 3 and len(y) == 3

    # Test split load
    X_train, y_train, X_test, y_test = ds.load("X_train", "y_train", "X_test", "y_test")
    assert len(X_train) == 2
    assert len(X_test) == 1
