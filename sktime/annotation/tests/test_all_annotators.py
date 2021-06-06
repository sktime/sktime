# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from sktime.annotation.base._mock import MockAnnotator
from sktime.utils._testing.annotation import make_annotation_problem


def _construct_instance():
    return MockAnnotator()


def test_output_is_series():

    data = make_annotation_problem()
    test_annotator = _construct_instance()
    test_annotator.fit(data)
    annotated_series = test_annotator.transform(data)

    assert isinstance(annotated_series, pd.Series)


def test_output_type():

    data = make_annotation_problem()
    test_annotator = _construct_instance()
    test_annotator.fit(data)
    annotated_series = test_annotator.transform(data)

    assert (
        (annotated_series.dtype == np.object)
        or (annotated_series.dtype == np.bool)
        or (annotated_series.dtype == np.int)
    )
