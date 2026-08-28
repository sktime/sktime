"""Tests for the sklearn tag adapter in utils.sklearn."""

__author__ = ["chala2001"]


import pytest
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.tests.test_switch import run_test_for_class
from sktime.utils.sklearn._tag_adapter import get_sklearn_tag

# sklearn tag names understood by get_sklearn_tag, and the sktime tag
# carrying the same information
TAG_PAIRS = [
    ("capability:multioutput", "capability:multioutput"),
    ("capability:categorical", "capability:categorical_in_X"),
    ("fit_is_empty", "fit_is_empty"),
]


@pytest.mark.skipif(
    not run_test_for_class(get_sklearn_tag),
    reason="Run if tag adapter has changed.",
)
@pytest.mark.parametrize("tagname, sktime_tagname", TAG_PAIRS)
def test_get_sklearn_tag_on_sktime_estimator(tagname, sktime_tagname):
    """Test that get_sklearn_tag reads sktime tags of sktime estimators.

    sktime objects do not implement ``__sklearn_tags__``. From scikit-learn 1.9 on,
    ``sklearn.utils.get_tags`` raises ``AttributeError`` on such objects instead of
    falling back to defaults, so the adapter must read the sktime tag instead.
    Failure case of bug #10725.
    """
    estimator = KNeighborsTimeSeriesRegressor()

    tag_value = get_sklearn_tag(estimator, tagname)

    expected = estimator.get_tag(sktime_tagname, False, False)
    assert tag_value == expected


@pytest.mark.skipif(
    not run_test_for_class(get_sklearn_tag),
    reason="Run if tag adapter has changed.",
)
@pytest.mark.parametrize("tagname, sktime_tagname", TAG_PAIRS)
def test_get_sklearn_tag_on_sklearn_estimator(tagname, sktime_tagname):
    """Test that get_sklearn_tag still reads tags of genuine sklearn estimators."""
    for estimator in [KNeighborsRegressor(), StandardScaler()]:
        assert isinstance(get_sklearn_tag(estimator, tagname), bool)
