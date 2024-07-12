# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformer to add binary column based on financial market holidays."""

import pandas

from sktime.transformations.base import BaseTransformer

__all__ = ["FinancialHolidaysTransformer"]
__author__ = ["yarnabrina"]


class FinancialHolidaysTransformer(BaseTransformer):
    """Financial Holidays Transformer.

    This implementation wraps over holidays [1]_ by vacanza.

    Based on the index of ``X``, dates are extracted and passed to ``holidays``. Then
    upon generating the holiday information for that day (or absence of it) based on
    passed financial market information, a boolean series is prepared where ``True``
    indicates the date being a holiday and ``False`` otherwise. ``fit`` is a no-op for
    this transformer.

    Parameters
    ----------
    market : str
        An ISO 3166-1 Alpha-2 market code [2]_; not implemented for all countries (see
        documentation [3]_).
    years : Union[int, Iterable[int]], optional
        The year(s) to pre-calculate public holidays for at instantiation.
    expand : bool, optional
        Whether the entire year is calculated when one date from that year is
        requested.
    observed : bool, optional
        Whether to include the dates of when public holiday are observed (e.g. a
        holiday falling on a Sunday being observed the following Monday). False may not
        work for all countries.
    name : str, optional
        name of transformed series.

    References
    ----------
    .. [1] https://github.com/vacanza/python-holidays
    .. [2] https://www.iso20022.org/market-identifier-codes
    .. [3] https://python-holidays.readthedocs.io/en/latest/#available-financial-markets

    Notes
    -----
    If ``name`` is missing (by default), it is auto-populated as ``{market}_holidays``.

    Examples
    --------
    >>> from sktime.transformations.series.holiday import FinancialHolidaysTransformer
    >>>
    >>> import numpy
    >>> data = numpy.random.default_rng(seed=0).random(size=365)
    >>>
    >>> import pandas
    >>> index = pandas.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    >>>
    >>> y = pandas.Series(data, index=index, name="random")
    >>>
    >>> y_t = FinancialHolidaysTransformer("XNYS").fit_transform(y)  # doctest: +SKIP
    >>> y_t.dtype  # doctest: +SKIP
    dtype('bool')
    >>> y_t.sum()  # doctest: +SKIP
    10
    >>> y_t.name  # doctest: +SKIP
    'XNYS_holidays'
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "yarnabrina",
        "maintainers": "yarnabrina",
        "python_dependencies": ["holidays"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "univariate-only": False,
        "fit_is_empty": True,
        "enforce_index_type": [pandas.DatetimeIndex, pandas.PeriodIndex],
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "handles-missing-data": True,
    }

    def __init__(self, market, years=None, expand=True, observed=True, name=None):
        self.market = market
        self.years = years
        self.expand = expand
        self.observed = observed
        self.name = name

        super().__init__()

    @property
    def _name(self):
        """Generate name of transformed series."""
        return self.name if self.name else f"{self.market}_holidays"

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing the core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series or Panel, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        transformed version of X
        """
        from holidays import financial_holidays

        holidays_data = financial_holidays(
            self.market, years=self.years, expand=self.expand, observed=self.observed
        )

        if isinstance(X.index, pandas.DatetimeIndex):
            dates = X.index
        elif isinstance(X.index, pandas.PeriodIndex):
            dates = X.index.to_timestamp()
        else:
            # should be non-reachable
            # enforce_index_type tag is set
            raise TypeError(
                f"{self.__class__.__name__} does not support {type(X.index)}"
            )

        # dates.isin(holidays_data) behave surprisingly
        # it fails to detect correctly first time
        # but works correctly while using one at a time
        financial_holidays = [date in holidays_data for date in dates]

        return pandas.Series(financial_holidays, name=self._name)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class

        Raises
        ------
        ValueError
            If an unknown parameter_set is provided.
        """
        del parameter_set  # avoid being detected as unused by ``vulture`` like tools

        params = [{"market": "XNYS"}, {"market": "ECB"}]

        return params
