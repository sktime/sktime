# -*- coding: utf-8 -*-
"""Test functions for data tssb data loading."""

__author__ = ["ermshaua"]
__all__ = []

import pytest

from sktime.datasets._tssb_data_io import load_tssb_dataset


@pytest.mark.parametrize("dataset", ["ArrowHead", "InlineSkate", "Plane", None])
def test_tssb_data_loader(dataset):
    if dataset is not None:
        dataset = [dataset]

    tssb = load_tssb_dataset(dataset)

    if dataset is not None:
        assert tssb.shape[0] == 1
        assert tssb.iloc[0].ts_name == dataset[0]
    else:
        assert tssb.shape[0] > 1
