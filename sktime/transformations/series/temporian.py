# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Implementation of TemporianTransformer. Contributed by the Temporian team."""

__author__ = ["ianspektor"]  # TODO: add other Temporian devs who contributed to this

from sktime.transformations.base import BaseTransformer


# TODO: add usage example to docstring
# see https://www.sktime.net/en/latest/developer_guide/dependencies.html#dependencies
# for how to handle soft deps (temporian) there (need to not run doc tests on it)
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

    References
    ----------
    .. [1] https://temporian.readthedocs.io/en/stable/
    .. [2] https://temporian.readthedocs.io/en/stable/reference/temporian/EventSet/
    .. [3] https://temporian.readthedocs.io/en/stable/reference/temporian/compile/
    """

    _tags = {
        "univariate-only": False,
        "authors": [
            "ianspektor"
        ],  # TODO: add other Temporian devs who contributed to this
        "maintainers": [
            "ianspektor"
        ],  # TODO: add other Temporian devs who will maintain this
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        # TODO: extend to pd-multiindex and pd_multiindex_hier
        "X_inner_mtype": "pd.DataFrame",
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "handles-missing-data": False,
        "python_dependencies": ["temporian"],
        "python_version": "<3.12",
    }

    def __init__(self, function, compile=False):
        self.function = function
        self.compile = compile

        super().__init__()

        # TODO: ensure function receives a single EventSet/param with inspect module?
        #       or is failing in runtime OK?
        # TODO: @tp.compile the function if self.compile is True

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

        # TODO: check function returns a single EventSet
        res = self.function(evset)

        res = tp.to_pandas(res).rename(columns={"timestamp": timestamps_col})
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
