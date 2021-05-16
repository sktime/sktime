# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sktime.annotation.base._base import BaseAnnotator


class TestAnnotator(BaseAnnotator):
    def __init__(self):
        pass

    def fit(self, Z, X=None):
        return self

    def transform(self, Z, X=None):
        Zt = Z.copy()
        Zt.iloc[:] = False
        Zt.iloc[1] = True
        return Zt


test_annotator = TestAnnotator()


def test_output_is_series():

    data = pd.Series(range(5))

    test_annotator.fit(data)
    annotated_series = test_annotator.transform(data)

    assert isinstance(annotated_series, pd.Series)


def test_output_type():

    data = pd.Series(range(5))

    test_annotator.fit(data)
    annotated_series = test_annotator.transform(data)

    assert annotated_series.dtype == np.object
