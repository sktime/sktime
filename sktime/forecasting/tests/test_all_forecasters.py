# -*- coding: utf-8 -*-
"""Tests for BaseForecaster API points.

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""

__author__ = ["mloning", "kejsitake", "fkiraly"]

import numpy as np
import pandas as pd
import pytest

from copy import deepcopy

from sktime.exceptions import NotFittedError
from sktime.forecasting.model_selection import (
    SlidingWindowSplitter,
    temporal_train_test_split,
)
from sktime.forecasting.tests._config import (
    TEST_ALPHAS,
    TEST_FHS,
    TEST_OOS_FHS,
    TEST_STEP_LENGTHS_INT,
    TEST_WINDOW_LENGTHS,
    VALID_INDEX_FH_COMBINATIONS,
)
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.registry import all_estimators
from sktime.tests.test_all_estimators import BaseFixtureGenerator, QuickTester
from sktime.utils._testing.forecasting import (
    _assert_correct_pred_time_index,
    _get_expected_index_for_update_predict,
    _get_n_columns,
    _make_fh,
    make_forecasting_problem,
)
from sktime.utils._testing.series import _make_series
from sktime.utils.validation.forecasting import check_fh

# get all forecasters
FORECASTERS = all_estimators(estimator_types="forecaster", return_names=False)
FH0 = 1
INVALID_X_INPUT_TYPES = [list(), tuple()]
INVALID_y_INPUT_TYPES = [list(), tuple()]

# testing data
y = make_forecasting_problem()
y_train, y_test = temporal_train_test_split(y, train_size=0.75)


class ForecasterFixtureGenerator(BaseFixtureGenerator):
    """Fixture generator for forecasting tests."""

    # note: this should be separate from TestAllForecasters
    #   additional fixtures, parameters, etc should be added here
    #   TestAllForecasters should contain the tests only

    estimator_type_filter = "forecaster"

    # todo: the n_columns business should be another conditional fixture


class TestAllForecasters(ForecasterFixtureGenerator, QuickTester):
    """Module level tests for all sktime forecasters."""

    def test_get_fitted_params(self, estimator_instance, scenario):
        """Test get_fitted_params."""
        scenario.run(estimator_instance, method_sequence=["fit"])
        try:
            params = estimator_instance.get_fitted_params()
            assert isinstance(params, dict)

        except NotImplementedError:
            pass

    # todo: should these not be checked in test_all_estimators?
    def test_raises_not_fitted_error(self, estimator_instance):
        """Test that calling post-fit methods before fit raises error."""
        # We here check extra method of the forecaster API: update and update_predict.
        with pytest.raises(NotFittedError):
            estimator_instance.update(y_test, update_params=False)

        with pytest.raises(NotFittedError):
            cv = SlidingWindowSplitter(fh=1, window_length=1, start_with_window=False)
            estimator_instance.update_predict(y_test, cv=cv)

        try:
            with pytest.raises(NotFittedError):
                estimator_instance.get_fitted_params()
        except NotImplementedError:
            pass

    def test_y_multivariate_raises_error(self, estimator_instance):
        """Test that wrong y scitype raises error (uni/multivariate not supported)."""
        if estimator_instance.get_tag("scitype:y") == "univariate":
            y = _make_series(n_columns=2)
            with pytest.raises(ValueError, match=r"univariate"):
                estimator_instance.fit(y, fh=FH0)

        if estimator_instance.get_tag("scitype:y") == "multivariate":
            y = _make_series(n_columns=1)
            with pytest.raises(ValueError, match=r"2 or more variables"):
                estimator_instance.fit(y, fh=FH0)

        if estimator_instance.get_tag("scitype:y") == "both":
            pass

    # todo: should these not be "negative scenarios", tested in test_all_estimators?
    @pytest.mark.parametrize("y", INVALID_y_INPUT_TYPES)
    def test_y_invalid_type_raises_error(self, estimator_instance, y):
        """Test that invalid y input types raise error."""
        with pytest.raises(TypeError, match=r"type"):
            estimator_instance.fit(y, fh=FH0)

    # todo: should these not be "negative scenarios", tested in test_all_estimators?
    @pytest.mark.parametrize("X", INVALID_X_INPUT_TYPES)
    def test_X_invalid_type_raises_error(self, estimator_instance, X):
        """Test that invalid X input types raise error."""
        n_columns_list = _get_n_columns(estimator_instance.get_tag("scitype:y"))

        for n_columns in n_columns_list:
            f = deepcopy(estimator_instance)
            y_train = _make_series(n_columns=n_columns)
            try:
                with pytest.raises(TypeError, match=r"type"):
                    f.fit(y_train, X, fh=FH0)
            except NotImplementedError as e:
                msg = str(e).lower()
                assert "exogenous" in msg

    # todo: refactor with scenarios. Need to override fh and scenario args for this.
    @pytest.mark.parametrize(
        "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
    )
    @pytest.mark.parametrize("steps", TEST_FHS)  # fh steps
    def test_predict_time_index(
        self, estimator_instance, index_type, fh_type, is_relative, steps
    ):
        """Check that predicted time index matches forecasting horizon."""
        n_columns_list = _get_n_columns(estimator_instance.get_tag("scitype:y"))

        for n_columns in n_columns_list:
            f = deepcopy(estimator_instance)
            y_train = _make_series(
                n_columns=n_columns, index_type=index_type, n_timepoints=50
            )
            cutoff = y_train.index[-1]
            fh = _make_fh(cutoff, steps, fh_type, is_relative)

            try:
                f.fit(y_train, fh=fh)
                y_pred = f.predict()
                _assert_correct_pred_time_index(y_pred.index, y_train.index[-1], fh=fh)
            except NotImplementedError:
                pass

    @pytest.mark.parametrize(
        "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
    )
    @pytest.mark.parametrize("steps", TEST_FHS)  # fh steps
    def test_predict_residuals(
        self, estimator_instance, index_type, fh_type, is_relative, steps
    ):
        """Check that predict_residuals method works as expected."""
        n_columns_list = _get_n_columns(estimator_instance.get_tag("scitype:y"))

        for n_columns in n_columns_list:
            f = deepcopy(estimator_instance)
            y_train = _make_series(
                n_columns=n_columns, index_type=index_type, n_timepoints=50
            )
            cutoff = y_train.index[-1]
            fh = _make_fh(cutoff, steps, fh_type, is_relative)
            try:
                f.fit(y_train, fh=fh)
                y_pred = f.predict()

                y_test = _make_series(
                    n_columns=n_columns, index_type=index_type, n_timepoints=len(y_pred)
                )
                y_test.index = y_pred.index
                y_res = f.predict_residuals(y_test)
                _assert_correct_pred_time_index(y_res.index, y_train.index[-1], fh=fh)
            except NotImplementedError:
                pass

    @pytest.mark.parametrize(
        "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
    )
    @pytest.mark.parametrize("steps", TEST_OOS_FHS)  # fh steps
    def test_predict_time_index_with_X(
        self, estimator_instance, index_type, fh_type, is_relative, steps
    ):
        """Check that predicted time index matches forecasting horizon."""
        n_columns_list = _get_n_columns(estimator_instance.get_tag("scitype:y"))

        z, X = make_forecasting_problem(index_type=index_type, make_X=True)

        # Some estimators may not support all time index types and fh types, hence we
        # need to catch NotImplementedErrors.
        for n_columns in n_columns_list:
            f = deepcopy(estimator_instance)
            y = _make_series(n_columns=n_columns, index_type=index_type)
            cutoff = y.index[len(y) // 2]
            fh = _make_fh(cutoff, steps, fh_type, is_relative)

            y_train, y_test, X_train, X_test = temporal_train_test_split(y, X, fh=fh)

            try:
                f.fit(y_train, X_train, fh=fh)
                y_pred = f.predict(X=X_test)
                _assert_correct_pred_time_index(y_pred.index, y_train.index[-1], fh)
            except NotImplementedError:
                pass

    @pytest.mark.parametrize(
        "index_type, fh_type, is_relative", VALID_INDEX_FH_COMBINATIONS
    )
    def test_predict_time_index_in_sample_full(
        self, estimator_instance, index_type, fh_type, is_relative
    ):
        """Check that predicted time index equals fh for full in-sample predictions."""
        n_columns_list = _get_n_columns(estimator_instance.get_tag("scitype:y"))

        for n_columns in n_columns_list:
            f = deepcopy(estimator_instance)
            y_train = _make_series(n_columns=n_columns, index_type=index_type)
            cutoff = y_train.index[-1]
            steps = -np.arange(len(y_train))
            fh = _make_fh(cutoff, steps, fh_type, is_relative)

            try:
                f.fit(y_train, fh=fh)
                y_pred = f.predict()
                _assert_correct_pred_time_index(y_pred.index, y_train.index[-1], fh)
            except NotImplementedError:
                pass

    def _check_pred_ints(
        self, pred_ints: pd.DataFrame, y_train: pd.Series, y_pred: pd.Series, fh
    ):
        # make iterable
        if isinstance(pred_ints, pd.DataFrame):
            pred_ints = [pred_ints]

        for pred_int in pred_ints:
            # check column naming convention
            assert list(pred_int.columns) == ["lower", "upper"]

            # check time index
            _assert_correct_pred_time_index(pred_int.index, y_train.index[-1], fh)
            # check values
            assert np.all(pred_int["upper"] >= pred_int["lower"])

            # check if errors are weakly monotonically increasing
            # pred_errors = y_pred - pred_int["lower"]
            # # assert pred_errors.is_mononotic_increasing
            # assert np.all(
            #     pred_errors.values[1:].round(4) >= pred_errors.values[:-1].round(4)
            # )

    @pytest.mark.parametrize("fh", TEST_OOS_FHS)
    @pytest.mark.parametrize("alpha", TEST_ALPHAS)
    def test_predict_interval(self, estimator_instance, fh, alpha):
        """Check prediction intervals returned by predict.

        Arguments
        ---------
        Forecaster: BaseEstimator class descendant, forecaster to test
        fh: ForecastingHorizon, fh at which to test prediction
        alpha: float, coverage at which to make prediction intervals

        Raises
        ------
        AssertionError - if Forecaster test instance has "capability:pred_int"
                and pred. int are not returned correctly when asking predict for them
        AssertionError - if Forecaster test instance does not have "capability:pred_int"
                and no NotImplementedError is raised when asking predict for pred.int
        """
        n_columns_list = _get_n_columns(estimator_instance.get_tag("scitype:y"))

        for n_columns in n_columns_list:
            f = deepcopy(estimator_instance)
            y_train = _make_series(n_columns=n_columns)
            f.fit(y_train, fh=fh)
            if f.get_tag("capability:pred_int"):
                if f._has_predict_quantiles_been_refactored():
                    y_pred = f.predict()
                    pred_ints = f.predict_interval(fh, coverage=alpha)

                    pred_ints = f._convert_new_to_old_pred_int(pred_ints, alpha)
                else:
                    y_pred, pred_ints = f.predict(return_pred_int=True, alpha=alpha)
                self._check_pred_ints(pred_ints, y_train, y_pred, fh)

            else:
                with pytest.raises(NotImplementedError, match="prediction intervals"):
                    f.predict(return_pred_int=True, alpha=alpha)

    def _check_predict_quantiles(
        self, pred_quantiles: pd.DataFrame, y_train: pd.Series, fh, alpha
    ):
        # check if the input is a dataframe
        assert isinstance(pred_quantiles, pd.DataFrame)
        # check time index (also checks forecasting horizon is more than one element)
        _assert_correct_pred_time_index(pred_quantiles.index, y_train.index[-1], fh)
        # Forecasters where name of variables do not exist
        # In this cases y_train is series - the upper level in dataframe == 'Quantiles'
        if isinstance(y_train, pd.Series):
            expected = pd.MultiIndex.from_product([["Quantiles"], [alpha]])
        else:
            # multiply variables with all alpha values
            expected = pd.MultiIndex.from_product([y_train.columns, [alpha]])
        assert all(expected == pred_quantiles.columns.to_flat_index())

        if isinstance(alpha, list):
            # sorts the columns that correspond to alpha values
            pred_quantiles = pred_quantiles.reindex(
                columns=pred_quantiles.columns.reindex(sorted(alpha), level=1)[0]
            )

            # check if values are monotonically increasing
            for var in pred_quantiles.columns.levels[0]:
                for index in range(len(pred_quantiles.index)):
                    assert pred_quantiles[var].iloc[index].is_monotonic_increasing

    @pytest.mark.parametrize("fh", TEST_OOS_FHS)
    @pytest.mark.parametrize("alpha", TEST_ALPHAS)
    def test_predict_quantiles(self, estimator_instance, fh, alpha):
        """Check prediction quantiles returned by predict.

        Arguments
        ---------
        Forecaster: BaseEstimator class descendant, forecaster to test
        fh: ForecastingHorizon, fh at which to test prediction
        alpha: float, alpha at which to make prediction intervals

        Raises
        ------
        AssertionError - if Forecaster test instance has "capability:pred_int"
                and pred. int are not returned correctly when asking predict for them
        AssertionError - if Forecaster test instance does not have "capability:pred_int"
                and no NotImplementedError is raised when asking predict for pred.int
        """
        n_columns_list = _get_n_columns(estimator_instance.get_tag("scitype:y"))
        for n_columns in n_columns_list:
            f = deepcopy(estimator_instance)
            y_train = _make_series(n_columns=n_columns)
            f.fit(y_train, fh=fh)
            try:
                quantiles = f.predict_quantiles(fh=fh, alpha=alpha)
                self._check_predict_quantiles(quantiles, y_train, fh, alpha)
            except NotImplementedError:
                pass

    def test_pred_int_tag(self, estimator_instance):
        """Checks whether the capability:pred_int tag is correctly set.

        Arguments
        ---------
        Forecaster: BaseEstimator class descendant, forecaster to test

        Raises
        ------
        ValueError - if capability:pred_int is True, but neither
            predict_interval nor predict_quantiles have implemented content
            this can be by direct implementation of _predict_interval/_predict_quantiles
            or by defaulting to each other and/or _predict_proba
        """
        f = estimator_instance
        implements_interval = f._has_implementation_of("_predict_interval")
        implements_quantiles = f._has_implementation_of("_predict_quantiles")
        implements_proba = f._has_implementation_of("_predict_proba")

        pred_int_works = implements_interval or implements_quantiles or implements_proba

        if not pred_int_works and f.get_class_tag("capability:pred_int", False):
            raise ValueError(
                f"{f.__name__} does not implement probabilistic forecasting, "
                'but "capability:pred_int" flag has been set to True incorrectly. '
                'The flag "capability:pred_int" should instead be set to False.'
            )

        if pred_int_works and not f.get_class_tag("capability:pred_int", False):
            raise ValueError(
                f"{f.__name__} does implement probabilistic forecasting, "
                'but "capability:pred_int" flag has been set to False incorrectly. '
                'The flag "capability:pred_int" should instead be set to True.'
            )

    @pytest.mark.parametrize("fh", TEST_OOS_FHS)
    def test_score(self, estimator_instance, fh):
        """Check score method."""
        n_columns_list = _get_n_columns(estimator_instance.get_tag("scitype:y"))

        for n_columns in n_columns_list:
            f = deepcopy(estimator_instance)
            y = _make_series(n_columns=n_columns)
            y_train, y_test = temporal_train_test_split(y)
            f.fit(y_train, fh=fh)
            y_pred = f.predict()

            fh_idx = check_fh(fh).to_indexer()  # get zero based index
            actual = f.score(y_test.iloc[fh_idx], fh=fh)
            expected = mean_absolute_percentage_error(
                y_pred, y_test.iloc[fh_idx], symmetric=True
            )

            # compare expected score with actual score
            actual = f.score(y_test.iloc[fh_idx], fh=fh)
            assert actual == expected

    @pytest.mark.parametrize("fh", TEST_OOS_FHS)
    @pytest.mark.parametrize("update_params", [True, False])
    def test_update_predict_single(self, estimator_instance, fh, update_params):
        """Check correct time index of update-predict."""
        f = estimator_instance.create_test_instance()
        n_columns_list = _get_n_columns(f.get_tag("scitype:y"))

        for n_columns in n_columns_list:
            f = deepcopy(estimator_instance)
            y = _make_series(n_columns=n_columns)
            y_train, y_test = temporal_train_test_split(y)
            f.fit(y_train, fh=fh)
            y_pred = f.update_predict_single(y_test, update_params=update_params)
            _assert_correct_pred_time_index(y_pred.index, y_test.index[-1], fh)

    def _check_update_predict_predicted_index(
        self, Forecaster, fh, window_length, step_length, update_params
    ):
        f = Forecaster.create_test_instance()
        n_columns_list = _get_n_columns(f.get_tag("scitype:y"))

        for n_columns in n_columns_list:
            f = Forecaster.create_test_instance()
            y = _make_series(
                n_columns=n_columns, all_positive=True, index_type="datetime"
            )
            y_train, y_test = temporal_train_test_split(y)
            cv = SlidingWindowSplitter(
                fh,
                window_length=window_length,
                step_length=step_length,
                start_with_window=False,
            )
            f.fit(y_train, fh=fh)
            y_pred = f.update_predict(y_test, cv=cv, update_params=update_params)
            assert isinstance(y_pred, (pd.Series, pd.DataFrame))
            expected = _get_expected_index_for_update_predict(y_test, fh, step_length)
            actual = y_pred.index
            np.testing.assert_array_equal(actual, expected)

    # todo: next two tests seem a case for joining by proper fixture parameterization
    # test with update_params=False and different values for steps_length
    @pytest.mark.parametrize("fh", TEST_OOS_FHS)
    @pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
    @pytest.mark.parametrize("step_length", TEST_STEP_LENGTHS_INT)
    @pytest.mark.parametrize("update_params", [False])
    def test_update_predict_predicted_index(
        self, estimator_instance, fh, window_length, step_length, update_params
    ):
        """Check predicted index in update_predict with update_params=False."""
        self._check_update_predict_predicted_index(
            estimator_instance, fh, window_length, step_length, update_params
        )

    # test with update_params=True and step_length=1
    @pytest.mark.parametrize("fh", TEST_OOS_FHS)
    @pytest.mark.parametrize("window_length", TEST_WINDOW_LENGTHS)
    @pytest.mark.parametrize("step_length", [1])
    @pytest.mark.parametrize("update_params", [True])
    def test_update_predict_predicted_index_update_params(
        self, estimator_instance, fh, window_length, step_length, update_params
    ):
        """Check predicted index in update_predict with update_params=True."""
        self._check_update_predict_predicted_index(
            estimator_instance, fh, window_length, step_length, update_params
        )

    def test__y_when_refitting(self, estimator_instance):
        """Test that _y is updated when forecaster is refitted."""
        columns = _get_n_columns(estimator_instance.get_tag("scitype:y"))
        for n_columns in columns:
            f = deepcopy(estimator_instance)
            y_train = _make_series(n_columns=n_columns)
            f.fit(y_train, fh=FH0)
            f.fit(y_train[3:], fh=FH0)
            # using np.squeeze to make the test flexible to shape differeces like
            # (50,) and (50, 1)
            assert np.all(np.squeeze(f._y) == np.squeeze(y_train[3:]))
