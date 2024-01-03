"""Fit-in-transform wrapper."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["aiwalter", "fkiraly"]
__all__ = ["FitInTransform"]

from sklearn import clone

from sktime.transformations.base import BaseTransformer


class FitInTransform(BaseTransformer):
    """Transformer wrapper to delay fit to the transform phase.

    In panel settings, e.g., time series classification, it can be preferable
    (or, necessary) to fit and transform on the test set, e.g., interpolate within the
    same series that interpolation parameters are being fitted on. `FitInTransform` can
    be used to wrap any transformer to ensure that `fit` and `transform` happen always
    on the same series, by delaying the `fit` to the `transform` batch.

    Warning: The use of `FitInTransform` will typically not be useful, or can constitute
    a mistake (data leakage) when naively used in a forecasting setting.

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series.
    skip_inverse_transform : bool
        The FitInTransform will skip inverse_transform by default, of the param
        skip_inverse_transform=False, then the inverse_transform is calculated
        by means of transformer.fit(X=X, y=y).inverse_transform(X=X, y=y) where
        transformer is the inner transformer. So the inner transformer is
        fitted on the inverse_transform data. This is required to have a non-
        state changing transform() method of FitInTransform.

    Examples
    --------
    >>> from sktime.datasets import load_longley
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.compose import ForecastingPipeline
    >>> from sktime.split import temporal_train_test_split
    >>> from sktime.transformations.compose import FitInTransform
    >>> from sktime.transformations.series.impute import Imputer
    >>> y, X = load_longley()
    >>> y_train, y_test, X_train, X_test = temporal_train_test_split(y, X)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> # we want to fit the Imputer only on the predict (=transform) data.
    >>> # note that NaiveForecaster can't use X data, this is just a show case.
    >>> pipe = ForecastingPipeline(
    ...     steps=[
    ...         ("imputer", FitInTransform(Imputer(method="mean"))),
    ...         ("forecaster", NaiveForecaster()),
    ...     ]
    ... )
    >>> pipe.fit(y_train, X_train)
    ForecastingPipeline(...)
    >>> y_pred = pipe.predict(fh=fh, X=X_test)
    """

    def __init__(self, transformer, skip_inverse_transform=True):
        self.transformer = transformer
        self.skip_inverse_transform = skip_inverse_transform
        super().__init__()
        self.clone_tags(transformer, None)
        self.set_tags(
            **{
                "fit_is_empty": True,
                "skip-inverse-transform": self.skip_inverse_transform,
            }
        )

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        return clone(self.transformer).fit_transform(X=X, y=y)

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _inverse_transform must support all types in it
            Data to be inverse transformed
        y : Series or Panel of mtype y_inner_mtype, optional (default=None)
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
        """
        return clone(self.transformer).fit(X=X, y=y).inverse_transform(X=X, y=y)

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
        """
        return {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.transformations.series.boxcox import BoxCoxTransformer

        params = [
            {"transformer": BoxCoxTransformer()},
            {"transformer": BoxCoxTransformer(), "skip_inverse_transform": False},
        ]
        return params
