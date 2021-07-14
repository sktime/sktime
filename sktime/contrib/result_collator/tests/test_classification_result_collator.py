# -*- coding: utf-8 -*-
from sktime.contrib.result_collator import ClassificationResultCollator, ResultCollator
from sktime.contrib.result_collator.classification_results_collator import (
    get_enum_from_url,
)
import socket

test_urls = [
    "https://timeseriesclassification.com/"
    "results/ResultsByClassifier/TSF_TESTFOLDS.csv"
]


# You probably want to disable these tests if this is added to the repo as you
# dont want to hit the server every time someone rebuilds sktime


def test_classification_result_collator():
    result_collator = ClassificationResultCollator(
        urls=test_urls,
        classifiers="*",
        problem_list="*",
        metric="accuracy",
        resamples=5,
        toolkit="sktime",
    )
    results = result_collator.get_results()

    assert results
