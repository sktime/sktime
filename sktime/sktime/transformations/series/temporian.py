# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Implementation of TemporianTransformer. Contributed by the Temporian team."""

__author__ = ["ianspektor", "javiber"]

from sktime.transformations.base import BaseTransformer


class TemporianTransformer(BaseTransformer):
    """Applies a Temporian function to the input time series.

    This transformer applies a Temporian [1]_ function to the input time series.

    The conversion from sktime's internal representation to Temporian's EventSet [2]_
    and back is handled automatically by the transformer.

    Parameters
    ----------
    function : Callable[[temporian.EventSet], temporian.EventSet]
        Temporian function to apply to the input time series. The function must receive
        and return a single Temporian EventSet, and can apply an arbitrary number of
        Temporian operators to its input.
    compile : bool, default=False
        If True, the function will be compiled using Temporian's @tp.compile [3]_
        decorator, which can lead to significant speedups by optimizing the graph of
        operations.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.series.temporian import TemporianTransformer
    >>> import temporian as tp  # doctest: +SKIP
    >>>
    >>> def function(evset):  # doctest: +SKIP
    ...     return evset.simple_moving_average(tp.duration.days(3 * 365))  \
        # doctest: +SKIP
    >>> transformer = TemporianTransformer(function=function)  # doctest: +SKIP
    >>> X = load_airline()  # doctest: +SKIP
    >>> X_averaged = transformer.fit_transform(X)  # doctest: +SKIP

    References
    ----------
    .. [1] https://temporian.readthedocs.io/en/stable/
    .. [2] https://temporian.readthedocs.io/en/stable/reference/temporian/EventSet/
    .. [3] https://temporian.readthedocs.io/en/stable/reference/temporian/compile/
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["ianspektor", "javiber"],
        "maintainers": ["ianspektor", "javiber"],
        "python_dependencies": ["temporian"],
        "python_version": ">=3.8",
        # estimator tags
        # --------------
        "univariate-only": False,
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "handles-missing-data": False,
    }

    def __init__(self, function, compile=False):
        self.function = function
        self.compile = compile

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : pd.DataFrame
            if self.get_tag("univariate-only")==True:
                guaranteed to have a single column
            if self.get_tag("univariate-only")==False: no restrictions apply
        y : None, present only for interface compatibility

        Returns
        -------
        transformed version of X
        """
        import temporian as tp

        X_noindex = X.reset_index(drop=False)
        X_noindex.columns = X_noindex.columns.astype(str)
        timestamps_col = X_noindex.columns[0]
        evset = tp.from_pandas(X_noindex, timestamps=timestamps_col)

        function = tp.compile(self.function) if self.compile else self.function

        res = function(evset)
        if not isinstance(res, tp.EventSet):
            raise TypeError(
                f"Expected return type to be an EventSet but received a {type(res)}"
            )

        # Test that the sampling was not modified, otherwise the conversion back to
        # pandas and the set_index would not work as expected
        try:
            res.check_same_sampling(evset)
        except ValueError as exc:
            raise ValueError(
                "The resulting EventSet must have the same sampling as the input. "
                "Visit our docs for more info "
                "https://temporian.readthedocs.io/en/stable/user_guide/#sampling"
            ) from exc

        res = tp.to_pandas(res, timestamps=False)
        res = res.set_index(X.index)

        return res

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
        return [
            {"function": _test_function_single_op, "compile": False},
            {"function": _test_function_many_ops, "compile": True},
        ]


def _test_function_single_op(evset):
    return evset + 1


def _test_function_many_ops(evset):
    return evset.simple_moving_average(10).lag(10).resample(evset)
