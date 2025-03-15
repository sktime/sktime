# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements transformer to add binary column based on country holidays."""

import pandas

from sktime.transformations.base import BaseTransformer

__author__ = ["yarnabrina"]


class CountryHolidaysTransformer(BaseTransformer):
    """Country Holidays Transformer.

    This implementation wraps over holidays [1]_ by vacanza.

    Based on the index of ``X``, dates are extracted and passed to ``holidays``. Then
    upon generating the holiday information for that day (or absence of it) based on
    passed country (and subdivision) information, a boolean series is prepared where
    ``True`` indicates the date being a holiday and ``False`` otherwise. ``fit`` is a
    no-op for this transformer.

    Parameters
    ----------
    country : str
        An ISO 3166-1 Alpha-2 country code. [2]_
    subdiv : str, optional
        The subdivision (e.g. state or province); not implemented for all countries
        (see documentation [3]_). [2]_
    years : Union[int, Iterable[int]], optional
        The year(s) to pre-calculate public holidays for at instantiation.
    expand : bool, optional
        Whether the entire year is calculated when one date from that year is
        requested.
    observed : bool, optional
        Whether to include the dates of when public holiday are observed (e.g. a
        holiday falling on a Sunday being observed the following Monday). False may not
        work for all countries.
    categories : Tuple[str], optional
        requested holiday categories.
    name : str, optional
        name of transformed series.

    References
    ----------
    .. [1] https://github.com/vacanza/python-holidays
    .. [2] https://www.iso.org/obp/ui/#search/code/
    .. [3] https://python-holidays.readthedocs.io/en/latest/#available-countries

    Notes
    -----
    If ``name`` is missing (by default), it is auto-populated using ``country`` and
    ``subdiv`` as ``{country}_holidays`` or ``{country}_{subdiv}_holidays``.

    Examples
    --------
    >>> from sktime.transformations.series.holiday import CountryHolidaysTransformer
    >>>
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>>
    >>> y_t = CountryHolidaysTransformer("US").fit_transform(y)  # doctest: +SKIP
    >>> y_t.dtype  # doctest: +SKIP
    dtype('bool')
    >>> y_t.sum()  # doctest: +SKIP
    14
    >>> y_t.name  # doctest: +SKIP
    'US_holidays'
    """

    _tags = {
        # packaging info
        # --------------
        "authors": "yarnabrina",
        "maintainers": "yarnabrina",
        "python_version": ">=3.8",
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

    def __init__(
        self,
        country,
        subdiv=None,
        years=None,
        expand=True,
        observed=True,
        categories=None,
        name=None,
    ):
        self.country = country
        self.subdiv = subdiv
        self.years = years
        self.expand = expand
        self.observed = observed
        self.categories = categories
        self.name = name

        super().__init__()

    @property
    def _name(self):
        """Generate name of transformed series."""
        if self.name:
            return self.name

        if self.subdiv:
            return f"{self.country}_{self.subdiv}_holidays"

        return f"{self.country}_holidays"

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
        from holidays import country_holidays

        holidays_data = country_holidays(
            self.country,
            subdiv=self.subdiv,
            years=self.years,
            expand=self.expand,
            observed=self.observed,
            categories=self.categories,
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
        country_holidays = [date in holidays_data for date in dates]

        return pandas.Series(country_holidays, name=self._name)

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
        """
        del parameter_set  # avoid being detected as unused by ``vulture`` like tools

        params = [
            {"country": "US"},
            {"country": "US", "subdiv": "CA"},
            {"country": "US", "years": 2000},
        ]

        return params


__all__ = ["CountryHolidaysTransformer"]
