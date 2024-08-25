"""Conditional transformation based on fitted parameters."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["TransformIf"]

from sktime.datatypes import ALL_TIME_SERIES_MTYPES, mtype_to_scitype
from sktime.transformations._delegate import _DelegatedTransformer
from sktime.transformations.base import BaseTransformer
from sktime.transformations.compose._common import CORE_MTYPES
from sktime.transformations.compose._id import Id


class TransformIf(_DelegatedTransformer):
    """Conditional execution of a transformer given a condition from a fittable object.

    Compositor to construct conditionally executed transformers, e.g.,

    * compute first differences if a stationarity test is positive
    * deseasonalize if a seasonality test is positive

    This compositor allows to specify a condition, and an if/else transformer.
    The default "else" transformer is "no transformation".

    The specific algorithm implemented is as follows:

    In ``fit``, for inputs ``X``, ``y``:
    1. fits ``if_estimator`` to ``X``, ``y``
    2. checks the condition for ``if_estimator`` fitted parameter ``param``:
       whether ``param`` satisfies ``condition`` with ``condition_value``
    3. If yes, fits ``then_est`` to ``X``, ``y``, and behaves as ``then_est`` from then
    on
       If no, fits ``else_est`` to ``X``, ``y``, and behaves as ``else_est`` from then
       on

    In other methods, behaves as ``then_est`` or ``else_est``, as above.

    Note: ``then_trafo`` and ``else_trafo`` must have the same input/output signature,
    e.g., Series-to-Series, or Series-to-Primitives.

    Parameters
    ----------
    if_estimator : sktime estimator, must have ``fit``
        sktime estimator to fit and apply to series.
        this is a "blueprint" estimator, state does not change when ``fit`` is called
    param : str, optional, default = first boolean parameter of fitted if_estimator
    condition : str, optional, default = "bool"
        condition that defines whether self behaves like ``then_est`` or ``else_est``
        this estimator behaves like ``then_est`` iff:
        "bool" = if ``param`` is True
        ">", ">=", "==", "<", "<=", "!=" = if ``param condition condition_value``
    condition_value : required for some conditions, see above; otherwise optional
    then_trafo : sktime transformer, optional, default=``if_estimator``
        transformer that this behaves as if condition is satisfied
        this is a "blueprint" transformer, state does not change when ``fit`` is called
    else_trafo : sktime transformer, optional default=``Id`` (identity/no transform)
        transformer that this behaves as if condition is not satisfied
        this is a "blueprint" transformer, state does not change when ``fit`` is called

    Attributes
    ----------
    transformer_ : transformer,
        this clone is fitted when ``fit`` is called
        if condition is satisfied, a clone of ``then_est``
        if condition is not satisfied, a clone of ``else_est``
    condition_ : bool,
        True if condition was true, False if it was false
    if_estimator_ : estimator
        this clone of ``if_estimator`` is fitted when ``fit`` is called

    Examples
    --------
    >>> from sktime.param_est.seasonality import SeasonalityACF
    >>> from sktime.transformations.compose import TransformIf
    >>> from sktime.transformations.series.detrend import Deseasonalizer
    >>> from sktime.datasets import load_airline
    >>>
    >>> y = load_airline()  # doctest: +SKIP
    >>>
    >>> seasonal = SeasonalityACF(candidate_sp=12)  # doctest: +SKIP
    >>> deseason = Deseasonalizer(sp=12)  # doctest: +SKIP
    >>> cond_deseason = TransformIf(seasonal, "sp", "!=", 1, deseason)  # doctest: +SKIP
    >>> y_hat = cond_deseason.fit_transform(y)  # doctest: +SKIP
    """

    _tags = {
        "authors": ["fkiraly"],
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

    def __init__(
        self,
        if_estimator,
        param=None,
        condition="bool",
        condition_value=None,
        then_trafo=None,
        else_trafo=None,
    ):
        self.if_estimator = if_estimator
        self.param = param
        self.condition = condition
        self.condition_value = condition_value
        self.then_trafo = then_trafo
        self.else_trafo = else_trafo

        if then_trafo is None:
            self.then_trafo_ = if_estimator
        else:
            self.then_trafo_ = then_trafo

        if else_trafo is None:
            self.else_trafo_ = Id()
        else:
            self.else_trafo_ = else_trafo

        super().__init__()

        self.clone_tags(if_estimator, tag_names=["univariate-only"])
        if_scitypes = mtype_to_scitype(if_estimator.get_tag("X_inner_mtype"))
        valid_scitypes = [
            x for x in ALL_TIME_SERIES_MTYPES if mtype_to_scitype(x) in if_scitypes
        ]
        self.set_tags(**{"X_inner_mtype": valid_scitypes})

        tags_to_clone = [
            "scitype:transform-input",
            "scitype:transform-output",
            "y_inner_mtype",
            "capability:inverse_transform",
            "transform-returns-same-time-index",
        ]
        self.clone_tags(self.then_trafo_, tag_names=tags_to_clone)

    # attribute for _DelegatedTransformer, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedTransformer docstring
    _delegate_name = "transformer_"

    def _evaluate_condition(self):
        """Evaluate the condition, as described in the docstring of the class."""
        param = self.param

        params = self.if_estimator_.get_fitted_params()

        # if param is None, get the first boolean parameter
        if param is None:
            bool_params = [k for k, v in params.items() if isinstance(v, bool)]
            param = bool_params[0]

        param_val = params[param]

        # evaluate condition
        condition = self.condition
        condition_value = self.condition_value

        if condition == "bool":
            cond_bool = param_val
        elif condition in [">=", ">", "==", "!=", "<", "<="]:
            cond_bool = eval(f"{param_val} {condition} {condition_value}")
        else:
            raise ValueError(
                f"unsupported value for parameter 'condition' found in "
                f"TransformIf: {condition}"
            )

        if cond_bool:
            return "if"
        else:
            return "else"

    def _fit(self, X, y=None):
        """Fit transformer to X, optionally to y.

        State change:
            Changes state to "fitted".

        Writes to self:
        model attributes (ending in "_") : dependent on estimator

        Parameters
        ----------
        X : Series or Panel, any supported mtype
            Data to fit transform to, of python type as follows:
                Series: pd.Series, pd.DataFrame, or np.ndarray (1D or 2D)
                Panel: pd.DataFrame with 2-level MultiIndex, list of pd.DataFrame,
                    nested pd.DataFrame, or pd.DataFrame in long/wide format
                subject to sktime mtype format specifications, for further details see
                    examples/AA_datatypes_and_datasets.ipynb
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        self : a fitted instance of the estimator
        """
        from sktime.registry import scitype

        if_estimator_ = self.if_estimator.clone()

        if scitype(if_estimator_) == "forecaster":
            self.if_estimator_ = if_estimator_.fit(y=X, X=y)
        elif scitype(if_estimator_) == "transformer":
            self.if_estimator_ = if_estimator_.fit(X=X, y=y)
        else:
            try:
                self.if_estimator_ = if_estimator_.fit(X, y)
            except Exception:
                self.if_estimator_ = if_estimator_.fit(X)

        if_or_else = self._evaluate_condition()

        if if_or_else == "if":
            self.transformer_ = self.then_trafo_.clone()
            self.condition_ = True
        elif if_or_else == "else":
            self.transformer_ = self.else_trafo_.clone()
            self.condition_ = False
        else:
            raise RuntimeError(
                "unexpected condition, bug in _evaluate_condition return"
            )

        self.transformer_.fit(X, y)
        return self

    # we also override _get_fitted_params with the original
    # to see the condition and then/else_trafo under a stable indexing
    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        return BaseTransformer._get_fitted_params(self)

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
        from sktime.param_est.fixed import FixedParams
        from sktime.transformations.series.boxcox import BoxCoxTransformer

        params1 = {
            "if_estimator": BoxCoxTransformer(),
            "param": "lambda",
            "condition": ">",
            "condition_value": 1.0,
        }

        params2 = {
            "if_estimator": FixedParams(param_dict={"foo": False}),
            "then_trafo": Id(),
            "else_trafo": BoxCoxTransformer(bounds=(2.0, 3.0)),
        }

        return [params1, params2]
