# -*- coding: utf-8 -*-
"""Tests for result collator."""
# Uncomment all this when running locally we don't want this to hit a external sever
# everytime someone prs into sktime
# from sktime.contrib.result_collator import ClassificationResultCollator, ResultCollator
# from sktime.contrib.result_collator.classification_results_collator import (
#     get_enum_from_url,
# )
# import socket
#
# test_urls = [
#     "https://timeseriesclassification.com/"
#     "results/ResultsByClassifier/TSF_TESTFOLDS.csv"
# ]
#
#
# # You probably want to disable these tests if this is added to the repo as you
# # dont want to hit the server every time someone rebuilds sktime
#
#
# def test_classification_result_collator() -> None:
#     """
#     Test method for result collator
#     """
#     return True
#     # Please uncomment this locally we don't want this running everytime sktime is
#     # recompiled
#     # result_collator = ClassificationResultCollator(
#     #     urls=test_urls,
#     #     classifiers="*",
#     #     problem_list="*",
#     #     metric="accuracy",
#     #     resamples=5,
#     #     toolkit="sktime",
#     # )
#     # results = result_collator.get_results()
#     #
#     # assert results
