#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Set up the datasets included in sktime."""

# The file is adapted from:
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/datasets/setup.py


___author__ = ["mloning"]


def configuration(parent_package="", top_path=None):
    """Configure the imports."""
    from numpy.distutils.misc_util import Configuration

    config = Configuration("datasets", parent_package, top_path)

    # add all datasets in sub-folders
    included_datasets = (
        "ArrowHead",
        "BasicMotions",
        "GunPoint",
        "OSULeaf",
        "ItalyPowerDemand",
        "JapaneseVowels",
        "Longley",
        "Lynx",
        "PLAID",
        "ShampooSales",
        "Airline",
        "ACSF1",
        "UnitTest",
        "Uschange",
        "PBS_dataset",
        "segmentation",
        "Covid3Month",
    )
    for dataset in included_datasets:
        config.add_data_dir(f"data/{dataset}")

    return config


if __name__ == "__main__":
    from numpy.distutils.core import setup

    setup(**configuration(top_path="").todict())
