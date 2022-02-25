# -*- coding: utf-8 -*-
"""Check that sktime classifiers all work with sckit learn model selection.

These are not formal tests, it would just take too long. Check on 3 data formats,
"""


def check_cross_validate(classifier):
    """Check classifier can be evaluated with scikit learn cross validation."""
    trainX, trainy
