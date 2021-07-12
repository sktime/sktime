# -*- coding: utf-8 -*-
from sktime.contrib.result_collator import ClassificationResultCollator

test_urls = [
    "https://timeseriesclassification.com/"
    "results/ResultsByClassifier/TSF_TESTFOLDS.csv"
]


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
