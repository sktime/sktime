"""Invert transform wrapper."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["InvertTransform"]

from sktime.transformations._delegate import _DelegatedTransformer
from sktime.utils.warnings import warn


class InvertTransform(_DelegatedTransformer):
    """Invert a series-to-series transformation.

    Switches `transform` and `inverse_transform`, leaves `fit` and `update` the same.

    Parameters
    ----------
    transformer : sktime transformer, must transform Series input to Series output
        this is a "blueprint" transformer, state does not change when `fit` is called

    Attributes
    ----------
    transformer_: transformer,
        this clone is fitted when `fit` is called and provides `transform` and inverse

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.transformations.compose import InvertTransform
    >>> from sktime.transformations.series.exponent import ExponentTransformer
    >>>
    >>> inverse_exponent = InvertTransform(ExponentTransformer(power=3))
    >>> X = load_airline()
    >>> Xt = inverse_exponent.fit_transform(X)  # computes 3rd square root
    """

    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for y?
        "univariate-only": False,
        "fit_is_empty": False,
        "capability:inverse_transform": True,
    }

    def __init__(self, transformer):
        self.transformer = transformer

        super().__init__()

        self.transformer_ = transformer.clone()

        # should be all tags, but not fit_is_empty
        #   (_fit should not be skipped)
        tags_to_clone = [
            "scitype:transform-input",
            "scitype:transform-output",
            "scitype:instancewise",
            "X_inner_mtype",
            "y_inner_mtype",
            "handles-missing-data",
            "X-y-must-have-same-index",
            "transform-returns-same-time-index",
            "skip-inverse-transform",
        ]
        self.clone_tags(transformer, tag_names=tags_to_clone)

        if not transformer.get_tag("capability:inverse_transform", False):
            warn(
                "transformer does not have capability to inverse transform, "
                "according to capability:inverse_transform tag. "
                "If the tag was correctly set, this transformer will likely crash",
                obj=self,
            )
        inner_output = transformer.get_tag("scitype:transform-output")
        if transformer.get_tag("scitype:transform-output") != "Series":
            warn(
                f"transformer output is not Series but {inner_output}, "
                "according to scitype:transform-output tag. "
                "The InvertTransform wrapper supports only Series output, therefore"
                " this transformer will likely crash on input.",
                obj=self,
            )

    # attribute for _DelegatedTransformer, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedTransformer docstring
    _delegate_name = "transformer_"

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Returns a transformed version of X by iterating over specified
        columns and applying the wrapped transformer to them.

        Parameters
        ----------
        X : sktime compatible time series container
            Data to be transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : sktime compatible time series container
            transformed version of X
        """
        return self.transformer_.inverse_transform(X=X, y=y)

    def _inverse_transform(self, X, y=None):
        """Logic used by `inverse_transform` to reverse transformation on `X`.

        Returns an inverse-transformed version of X by iterating over specified
        columns and applying the univariate series transformer to them.

        Only works if `self.transformer` has an `inverse_transform` method.

        Parameters
        ----------
        X : sktime compatible time series container
            Data to be inverse transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        Xt : sktime compatible time series container
            inverse transformed version of X
        """
        return self.transformer_.transform(X=X, y=y)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.transformations.series.boxcox import BoxCoxTransformer
        from sktime.transformations.series.exponent import ExponentTransformer

        # ExponentTransformer skips fit
        params1 = {"transformer": ExponentTransformer()}
        # BoxCoxTransformer has fit
        params2 = {"transformer": BoxCoxTransformer()}

        return [params1, params2]
