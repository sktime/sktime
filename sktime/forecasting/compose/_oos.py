# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of Out-of-sample Forecaster."""

__author__ = ["geetu040"]

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base import BaseForecaster
from sktime.utils.validation.forecasting import check_cv


class OosForecaster(BaseForecaster):
    """Out-of-sample Forecaster for generating in-sample predictions via refitting.

    The ``OosForecaster`` is a wrapper-forecaster that enables *out-of-sample-style*
    predictions for in-sample time points. It ensures that predictions for points
    within the training data are computed without information leakage, by refitting
    the wrapped forecaster on progressively expanding or rolling contexts.

    In standard forecasters, in-sample predictions are typically obtained directly
    from fitted values, which reuse information from the target observation itself.
    In contrast, this wrapper enforces a strictly causal prediction regime by
    repeatedly refitting the forecaster over cross-validation folds and computing
    predictions only for unseen points in each fold. This produces *true*
    out-of-sample predictions even for in-sample horizons.

    This behavior is useful in scenarios that require in-sample residuals or
    forecasts computed without lookahead bias, such as:

    * Causal or deconfounded forecasting frameworks (e.g., DoubleMLForecaster)
    * Residual-based ensembles or boosting forecasters
    * Adding in-sample forecast capability to a forecaster that does not have it.
    * Evaluating model performance under strict out-of-sample conditions

    **Algorithm**

    1. Split the forecasting horizon ``fh`` into:
       - **In-sample fh**: time points within the training period.
       - **Out-of-sample fh**: time points beyond the training cutoff.

    2. For out-of-sample forecasts, simply revert to wrapped forecaster:
       - Clone the wrapped forecaster.
       - Fit it once on the full training data.
       - Predict on the out-of-sample horizon.

    3. For in-sample forecasts:
       - Use a cross-validation splitter ``cv`` (e.g., ``ExpandingWindowSplitter``).
       - For each split, fit or update the forecaster on the current training window.
       - Predict for the corresponding test window, ensuring no future information
         is used.
       - Aggregate predictions across all splits to form the complete
         in-sample forecast.

    4. Any in-sample points not covered by a ``cv`` split are filled with ``np.nan``
       to preserve index alignment.

    This procedure guarantees that all predictions, both in-sample and out-of-sample,
    are generated under out-of-sample conditions, preserving temporal causality
    and enabling unbiased residual analysis.

    Parameters
    ----------
    forecaster : sktime forecaster
        The base forecaster to be wrapped. This forecaster is used to perform both
        out-of-sample forecasts and the iterative in-sample forecasting procedure
        over cross-validation folds. Any compatible sktime forecaster can be used.

    cv : sktime splitter, optional (default=None)
        Cross-validation splitter defining how to generate rolling or expanding
        training/test windows for in-sample residual computation. If not provided,
        defaults to ``ExpandingWindowSplitter(initial_window=2)``, which performs
        expanding window updates starting from the first 2 observation.

    Attributes
    ----------
    _in_forecaster : sktime forecaster
        Internal clone of the wrapped forecaster used for generating
        out-of-sample-style in-sample forecasts through rolling refits.

    _oos_forecaster : sktime forecaster
        Internal clone of the wrapped forecaster fitted once on the full training
        data for standard out-of-sample forecasting.

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.split import temporal_train_test_split, ExpandingWindowSplitter
    >>> from sktime.forecasting.compose import OosForecaster
    >>> from sktime.forecasting.compose import make_reduction
    >>> from sklearn.linear_model import LinearRegression
    >>>
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(
    ...     y, X, test_size=0.2
    ... )
    >>>
    >>> wrapper = OosForecaster(
    ...     forecaster=make_reduction(
    ...             estimator=LinearRegression(),
    ...             strategy="recursive",
    ...             window_length=4,
    ...     ),
    ...     cv=ExpandingWindowSplitter(initial_window=7),
    ... )
    >>>
    >>> fh = [-3, -1, 0, 1, 2, 3, 4]
    >>> wrapper.fit(y_train, X=X_train, fh=fh)
    OosForecaster(cv=ExpandingWindowSplitter(initial_window=7),
                  forecaster=RecursiveTabularRegressionForecaster(estimator=LinearRegression(),
                                                                  window_length=4))
    >>> y_pred = wrapper.predict(X=X_test)
    """

    _tags = {
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "scitype:y": "univariate",
        "capability:exogenous": True,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "capability:missing_values": True,
        "capability:categorical_in_X": True,
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "requires-fh-in-fit": False,
    }

    def __init__(self, forecaster, cv=None):
        self.forecaster = forecaster
        self.cv = cv
        self._in_forecaster = None
        self._oos_forecaster = None

        super().__init__()

        self.set_tags(
            **{
                "capability:exogenous": self.forecaster.get_tag("capability:exogenous"),
                "capability:pred_int": self.forecaster.get_tag("capability:pred_int"),
                "capability:missing_values": self.forecaster.get_tag(
                    "capability:missing_values"
                ),
                "capability:categorical_in_X": self.forecaster.get_tag(
                    "capability:categorical_in_X"
                ),
            }
        )

    def _split_fh(self, fh=None):
        if fh is None:
            return None, None

        in_fh = (
            fh.to_in_sample(self.cutoff)
            if not fh.is_all_out_of_sample(self.cutoff)
            else None
        )
        oos_fh = (
            fh.to_out_of_sample(self.cutoff)
            if not fh.is_all_in_sample(self.cutoff)
            else None
        )

        return in_fh, oos_fh

    def _fit(self, y, X, fh):
        _, oos_fh = self._split_fh(fh)

        self._oos_forecaster = clone(self.forecaster)
        self._oos_forecaster.fit(y=y, X=X, fh=oos_fh)

    def _custom_predict(self, fh, X, method_name, **method_kwargs):
        _y = self._y
        _X = self._X
        in_fh, oos_fh = self._split_fh(fh)

        # Prepare CV
        cv = self.cv
        if cv is None and in_fh is not None:
            from sktime.split import ExpandingWindowSplitter
            from sktime.utils.warnings import warn

            warn(
                "No cross-validation splitter (`cv`) was provided to "
                "`OosForecaster`. A default "
                "`ExpandingWindowSplitter(initial_window=2)` has been "
                "initialized. Ensure that the wrapped forecaster is "
                "compatible with this splitter, or pass a suitable `cv` "
                "object when constructing the wrapper.",
                category=UserWarning,
                obj=self,
            )

            cv = ExpandingWindowSplitter(initial_window=2)
        elif cv is not None:
            cv = check_cv(cv)

        # Prepare placeholder for predictions
        columns = self._get_columns(method=method_name, **method_kwargs)
        index = fh.to_absolute_index(self.cutoff)
        if isinstance(_y.index, pd.MultiIndex):
            index = pd.MultiIndex.from_product(
                _y.index.levels[:-1] + [index], names=_y.index.names
            )
        preds = pd.DataFrame(np.nan, index=index, columns=columns)

        # Out-of-sample predictions
        if oos_fh is not None:
            method = getattr(self._oos_forecaster, method_name)
            pred = method(X=X, fh=oos_fh, **method_kwargs)
            preds.loc[pred.index] = pred.values

        # In-sample predictions using cross-validation
        if in_fh is not None:
            self._in_forecaster = clone(self.forecaster)
            method = getattr(self._in_forecaster, method_name)

            # fit on the first training window
            window, horizon = next(cv.split(_y))
            new_y = _y.iloc[window]
            new_X = _X.iloc[window] if _X is not None else None
            self._in_forecaster.fit(y=new_y, X=new_X, fh=cv.get_fh())

            # update on all training windows
            for window, horizon in cv.split(_y):
                new_y = _y.iloc[window]
                new_X = _X.iloc[window] if _X is not None else None
                new__X = _X.iloc[horizon] if _X is not None else None

                self._in_forecaster.update(y=new_y, X=new_X, update_params=True)
                pred = method(X=new__X, **method_kwargs)

                common_idx = pred.index.intersection(preds.index)
                preds.loc[common_idx] = pred.loc[common_idx].values

        return preds

    def _predict(self, fh, X):
        return self._custom_predict(fh=fh, X=X, method_name="predict")

    def _predict_interval(self, fh, X=None, coverage=0.9):
        return self._custom_predict(
            fh=fh, X=X, method_name="predict_interval", coverage=coverage
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create test instances of the estimator.
        """
        from sklearn.linear_model import LinearRegression

        from sktime.forecasting.compose import make_reduction
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.split import ExpandingWindowSplitter

        # Basic test parameters
        params1 = {
            "forecaster": NaiveForecaster(strategy="drift"),
        }

        # More complex test parameters
        params2 = {
            "forecaster": make_reduction(
                estimator=LinearRegression(),
                strategy="recursive",
                window_length=2,
            ),
            "cv": ExpandingWindowSplitter(initial_window=4, step_length=1, fh=1),
        }

        return [params1, params2]
