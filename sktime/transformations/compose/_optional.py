"""Optional passthrough transformer."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["aiwalter", "fkiraly"]
__all__ = ["OptionalPassthrough"]

from sktime.transformations._delegate import _DelegatedTransformer
from sktime.transformations.compose._common import CORE_MTYPES
from sktime.transformations.compose._id import Id


class OptionalPassthrough(_DelegatedTransformer):
    """Wrap an existing transformer to tune whether to include it in a pipeline.

    Allows tuning the implicit hyperparameter whether or not to use a
    particular transformer inside a pipeline (e.g. TransformedTargetForecaster)
    or not. This is achieved by the hyperparameter ``passthrough``
    which can be added to a tuning grid then (see example).

    Parameters
    ----------
    transformer : Estimator
        scikit-learn-like or sktime-like transformer to fit and apply to series.
        this is a "blueprint" transformer, state does not change when ``fit`` is called
    passthrough : bool, default=False
       Whether to apply the given transformer or to just
        passthrough the data (identity transformation). If, True the transformer
        is not applied and the OptionalPassthrough uses the identity
        transformation.

    Attributes
    ----------
    transformer_: transformer,
        this clone is fitted when ``fit`` is called and provides ``transform`` and
        inverse
        if passthrough = False, a clone of ``transformer``passed
        if passthrough = True, the identity transformer ``Id``

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.transformations.compose import OptionalPassthrough
    >>> from sktime.transformations.series.detrend import Deseasonalizer
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> from sktime.split import SlidingWindowSplitter
    >>> from sklearn.preprocessing import StandardScaler
    >>> # create pipeline
    >>> pipe = TransformedTargetForecaster(steps=[
    ...     ("deseasonalizer", OptionalPassthrough(Deseasonalizer())),
    ...     ("scaler", OptionalPassthrough(TabularToSeriesAdaptor(StandardScaler()))),
    ...     ("forecaster", NaiveForecaster())])  # doctest: +SKIP
    >>> # putting it all together in a grid search
    >>> cv = SlidingWindowSplitter(
    ...     initial_window=60,
    ...     window_length=24,
    ...     start_with_window=True,
    ...     step_length=48)  # doctest: +SKIP
    >>> param_grid = {
    ...     "deseasonalizer__passthrough" : [True, False],
    ...     "scaler__transformer__transformer__with_mean": [True, False],
    ...     "scaler__passthrough" : [True, False],
    ...     "forecaster__strategy": ["drift", "mean", "last"]}  # doctest: +SKIP
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=pipe,
    ...     param_grid=param_grid,
    ...     cv=cv,
    ...     n_jobs=-1)  # doctest: +SKIP
    >>> gscv_fitted = gscv.fit(load_airline())  # doctest: +SKIP
    """

    _tags = {
        "authors": ["aiwalter", "fkiraly"],
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": CORE_MTYPES,
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "fit_is_empty": False,
        "capability:inverse_transform": True,
    }

    def __init__(self, transformer, passthrough=False):
        self.transformer = transformer
        self.passthrough = passthrough

        super().__init__()

        # should be all tags, but not fit_is_empty
        #   (_fit should not be skipped)
        tags_to_clone = [
            "scitype:transform-input",
            "scitype:transform-output",
            "scitype:instancewise",
            "y_inner_mtype",
            "capability:inverse_transform",
            "handles-missing-data",
            "X-y-must-have-same-index",
            "transform-returns-same-time-index",
            "skip-inverse-transform",
        ]
        self.clone_tags(transformer, tag_names=tags_to_clone)

        if passthrough:
            self.transformer_ = Id()
        else:
            self.transformer_ = transformer.clone()

    # attribute for _DelegatedTransformer, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedTransformer docstring
    _delegate_name = "transformer_"

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.transformations.series.boxcox import BoxCoxTransformer

        return {"transformer": BoxCoxTransformer(), "passthrough": False}
