# -*- coding: utf-8 -*-
from sktime.clustering_redo.partitioning._lloyds import _Lloyds
from sktime.datasets import load_UCR_UEA_dataset
from sktime.distances.tests._utils import create_test_distance_numpy

dataset_name = "Beef"


class _test_class(_Lloyds):
    def __init__(self):
        super(_test_class, self).__init__(random_state=1)
        pass


def test_lloyds():
    """Test implemention of Lloyds."""
    X_train, y_train = load_UCR_UEA_dataset(
        dataset_name, split="train", return_X_y=True
    )
    X_test, y_test = load_UCR_UEA_dataset(dataset_name, split="test", return_X_y=True)

    X_train = create_test_distance_numpy(20, 10, 10)

    lloyd = _test_class()
    lloyd.fit(X_train)


if __name__ == "__main__":
    test_lloyds()
