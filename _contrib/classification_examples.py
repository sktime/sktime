"""Classifier Examples: some use case examples for building and assessing classifiers.

This will become a notebook once complete.
"""
__author__ = ["TonyBagnall"]

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.kernel_based import RocketClassifier
from sktime.classification.hybrid import HIVECOTEV2
from sktime.datasets import load_unit_test


def make_toy_problem(n_dims=1, n_instances_per_class=20, n_timepoints=100):
    """Make a toy binary classification problem out of numpy arrays.

    Parameters
    ----------
    n_dims : int
        Number of dimensions/channels. 1 returns a 2D array (n_instances,
        n_timepoints); >1 returns a 3D array (n_instances, n_dims,
        n_timepoints).
    n_instances_per_class : int
        Number of instances to generate for each of the two classes.
    n_timepoints : int
        Series length.

    Returns
    -------
    X_train, y_train, X_test, y_test
    """

    def _make_split():
        shape1 = (
            (n_instances_per_class, n_timepoints)
            if n_dims == 1
            else (n_instances_per_class, n_dims, n_timepoints)
        )
        X_class1 = np.random.uniform(-1, 1, size=shape1)
        y_class1 = np.zeros(n_instances_per_class)
        X_class2 = np.random.uniform(-0.9, 1.1, size=shape1)
        y_class2 = np.ones(n_instances_per_class)
        X = np.concatenate((X_class1, X_class2), axis=0)
        y = np.concatenate((y_class1, y_class2))
        return X, y

    X_train, y_train = _make_split()
    X_test, y_test = _make_split()
    return X_train, y_train, X_test, y_test


def build_classifiers():
    """Examples of building a classifier.

    1. Directly from 2D numpy arrays.
    2. Directly from 3D numpy arrays.
    3. From a nested pandas.
    4. From a baked in dataset.
    5. From any UCR/UEA dataset downloaded from timeseriesclassification.com.
    """
    X_train, y_train, X_test, y_test = make_toy_problem(n_dims=1)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    # Random forest on 2D data (sklearn expects (n_instances, n_features))
    randf = RandomForestClassifier()
    randf.fit(X_train, y_train)
    print("Random Forest acc = ", randf.score(X_test, y_test))

    # ContractableBOSS, an sktime time series classifier
    cboss = ContractableBOSS(time_limit_in_minutes=1)
    cboss.fit(X_train, y_train)
    print("CBOSS acc = ", cboss.score(X_test, y_test))

    # RocketClassifier, works on 2D or 3D input
    rocket = RocketClassifier(num_kernels=500)
    rocket.fit(X_train, y_train)
    print("Rocket acc = ", rocket.score(X_test, y_test))

    # 3D (multivariate) toy problem, needed for HIVE-COTE style classifiers
    X_train_3d, y_train_3d, X_test_3d, y_test_3d = make_toy_problem(n_dims=3)
    y_train_3d = pd.Series(y_train_3d)
    y_test_3d = pd.Series(y_test_3d)

    hc2 = HIVECOTEV2(time_limit_in_minutes=1)
    hc2.fit(X_train_3d, y_train_3d)
    print("HC2 acc = ", hc2.score(X_test_3d, y_test_3d))


def compare_classifiers(datasets=None):
    """Build pipeline classifiers, compare accuracies and draw a CD diagram.

    Parameters
    ----------
    datasets : list of str, optional
        Names of datasets to load via sktime's dataset loaders. Defaults to
        just the small "UnitTest" dataset bundled with sktime, so this runs
        quickly out of the box. Swap in real UCR/UEA dataset names to run a
        proper comparison, e.g. ["Chinatown", "ItalyPowerDemand", "Coffee"].
    """
    if datasets is None:
        datasets = ["UnitTest"]

    classifiers = {
        "CBOSS": ContractableBOSS(time_limit_in_minutes=1),
        "Rocket": RocketClassifier(num_kernels=500),
    }

    results = pd.DataFrame(index=datasets, columns=list(classifiers.keys()), dtype=float)

    for dataset_name in datasets:
        # load_unit_test is a stand-in loader for the bundled toy dataset;
        # swap for sktime.datasets.load_UCR_UEA_dataset(name=dataset_name)
        # to pull real datasets from timeseriesclassification.com
        if dataset_name == "UnitTest":
            X_train, y_train = load_unit_test(split="train", return_X_y=True)
            X_test, y_test = load_unit_test(split="test", return_X_y=True)
        else:
            from sktime.datasets import load_UCR_UEA_dataset

            X_train, y_train = load_UCR_UEA_dataset(
                name=dataset_name, split="train", return_X_y=True
            )
            X_test, y_test = load_UCR_UEA_dataset(
                name=dataset_name, split="test", return_X_y=True
            )

        for clf_name, clf in classifiers.items():
            clf.fit(X_train, y_train)
            acc = clf.score(X_test, y_test)
            results.loc[dataset_name, clf_name] = acc
            print(f"{dataset_name} / {clf_name} acc = {acc:.4f}")

    print("\nResults summary:")
    print(results)

    _draw_cd_diagram(results)
    return results


def _draw_cd_diagram(results):
    """Draw a critical difference diagram comparing classifiers.

    Requires at least 3 datasets to be meaningful; with only one dataset
    (the default quick-run case) this just prints a note instead.
    """
    if results.shape[0] < 3:
        print(
            "\nSkipping CD diagram: need results on >= 3 datasets for a "
            "meaningful critical difference comparison. Pass more dataset "
            "names to compare_classifiers()."
        )
        return

    try:
        from sktime.benchmarking.evaluation import Evaluator
        from sktime.benchmarking.results import RAMResults

        # sktime's benchmarking API expects results logged per-dataset via a
        # Results object; wiring this up fully requires running fit/predict
        # through sktime.benchmarking.tasks/strategies rather than the plain
        # sklearn-style calls above. Left as a pointer for a full pipeline.
        print(
            "\nTo draw a proper CD diagram, log per-fold results through "
            "sktime.benchmarking (RAMResults + Evaluator) or use the "
            "critical-difference plotting utility in "
            "sktime.benchmarking.evaluation."
        )
    except ImportError:
        print("\nsktime.benchmarking not available; skipping CD diagram.")


if __name__ == "__main__":
    print("=== build_classifiers ===")
    build_classifiers()

    print("\n=== compare_classifiers ===")
    compare_classifiers()
