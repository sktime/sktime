# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

"""Implementation of TemporianTransformer. Contributed by the Temporian team."""

__author__ = ["ianspektor"]  # TODO: add other Temporian devs who contributed to this

from sktime.transformations.base import BaseTransformer


# TODO: add usage example to docstring
# see https://www.sktime.net/en/latest/developer_guide/dependencies.html#dependencies
# for how to handle soft deps (temporian) there (need to not run doc tests on it)
class TemporianTransformer(BaseTransformer):
    """Applies a Temporian function to the input time series.

    This transformer applies a [Temporian](https://temporian.readthedocs.io/en/stable/)
    function to the input time series.

    The conversion from sktime's internal representation to Temporian's
    [EventSet](https://temporian.readthedocs.io/en/stable/reference/temporian/EventSet/)
    and back is handled automatically by the transformer.

    Parameters
    ----------
    function : Callable[[temporian.EventSet], temporian.EventSet]
        Temporian function to apply to the input time series. The function must receive
        and return a single Temporian EventSet, and can apply an arbitrary number of
        Temporian operators to its input.
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

    def __init__(self, function):
        self.function = function

        super().__init__()

        # TODO: ensure function receives a single EventSet/param with inspect module?
        #       or is failing in runtime OK?
        # TODO: @tp.compile the function

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

        timestamps_col = X.index.name
        X_noindex = X.reset_index(drop=False)
        evset = tp.from_pandas(X_noindex, timestamps=timestamps_col)

        # TODO: check function returns a single EventSet
        res = self.function(evset)

        res = tp.to_pandas(res).rename(columns={"timestamp": timestamps_col})
        res = res.set_index(timestamps_col).to_period()
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
        # dependency check is required here, as this method is used in tests
        # this is currently a small hack for the unusual case where all valid
        # parameter dictionaries already require the dependency
        from sktime.utils.validation._dependencies import _check_soft_dependencies

        deps = cls.get_class_tag("python_dependencies")
        _check_soft_dependencies(deps, severity="error")

        return [
            {"function": _test_function},
            {"function": _test_function2},
        ]


def _test_function(evset):
    return evset + 1


def _test_function2(evset):
    return evset.simple_moving_average(10)
