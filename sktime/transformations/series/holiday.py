# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements holidays transformers."""
import pandas

from sktime.transformations.base import BaseTransformer

__all__ = ["CountryHolidaysTransformer", "FinancialHolidaysTransformer"]
__author__ = ["yarnabrina"]


class CountryHolidaysTransformer(BaseTransformer):
    """Country Holidays Transformer.

    This implementation wraps over holidays [1]_ by dr-prodigy.

    Parameters
    ----------
    country : str
        An ISO 3166-1 Alpha-2 country code.
    subdiv : str, optional
        The subdivision (e.g. state or province); not implemented for all countries
        (see documentation).
    years : Union[int, Iterable[int]], optional
        The year(s) to pre-calculate public holidays for at instantiation.
    expand : bool, optional
        Whether the entire year is calculated when one date from that year is
        requested.
    observed : bool, optional
        Whether to include the dates of when public holiday are observed (e.g. a
        holiday falling on a Sunday being observed the following Monday). False may not
        work for all countries.
    language : str, optional
        The language which the returned holiday names will be translated into. It must
        be an ISO 639-1 (2-letter) language code. If the language translation is not
        supported the original holiday names will be used.
    categories : Tuple[str], optional
        requested holiday categories.

    References
    ----------
    .. [1] https://github.com/dr-prodigy/python-holidays

    Examples
    --------
    >>> from sktime.transformations.series.holiday import CountryHolidaysTransformer
    >>>
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>>
    >>> y_t = CountryHolidaysTransformer("US").fit_transform(y)
    >>> y_t.columns
    Index(['Number of airline passengers', 'country_holidays'], dtype='object')
    >>> y_t.dtypes.to_dict()
    {'Number of airline passengers': dtype('float64'), 'country_holidays': dtype('bool')}  # noqa: E501
    >>> y_t["country_holidays"].sum()
    14
    """

    _tags = {
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
        "python_version": ">=3.8",
        "python_dependencies": ["holidays"],
    }

    def __init__(
        self,
        country,
        subdiv=None,
        years=None,
        expand=True,
        observed=True,
        language=None,
        categories=None,
    ):
        self.country = country
        self.subdiv = subdiv
        self.years = years
        self.expand = expand
        self.observed = observed
        self.language = language
        self.categories = categories

        super().__init__()

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
            language=self.language,
            categories=self.categories,
        )

        Xt = X.copy(deep=True)

        if isinstance(Xt.index, pandas.DatetimeIndex):
            dates = Xt.index
        elif isinstance(Xt.index, pandas.PeriodIndex):
            dates = Xt.index.to_timestamp()
        else:
            raise TypeError(f"index type is unsupported: {type(Xt.index)}")

        # dates.isin(holidays_data) behave surprisingly
        # it fails to detect correctly first time
        # but works correctly while using one at a time
        Xt["country_holidays"] = [date in holidays_data for date in dates]

        return Xt

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
        del parameter_set  # avoid being detected as unused by ``vulture`` like tools

        params = [
            {"country": "US"},
            {"country": "US", "subdiv": "CA"},
            {"country": "US", "years": 2000},
        ]

        return params


class FinancialHolidaysTransformer(BaseTransformer):
    """Financial Holidays Transformer.

    This implementation wraps over holidays [1]_ by dr-prodigy.

    Parameters
    ----------
    market : str
        An ISO 3166-1 Alpha-2 market code.
    years : Union[int, Iterable[int]], optional
        The year(s) to pre-calculate public holidays for at instantiation.
    expand : bool, optional
        Whether the entire year is calculated when one date from that year is
        requested.
    observed : bool, optional
        Whether to include the dates of when public holiday are observed (e.g. a
        holiday falling on a Sunday being observed the following Monday). False may not
        work for all countries.
    language : str, optional
        The language which the returned holiday names will be translated into. It must
        be an ISO 639-1 (2-letter) language code. If the language translation is not
        supported the original holiday names will be used.

    References
    ----------
    .. [1] https://github.com/dr-prodigy/python-holidays

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
    >>> y_t = FinancialHolidaysTransformer("NYSE").fit_transform(y)
    >>> y_t.columns
    Index(['random', 'financial_holidays'], dtype='object')
    >>> y_t.dtypes.to_dict()
    {'random': dtype('float64'), 'financial_holidays': dtype('bool')}
    >>> y_t["financial_holidays"].sum()
    10
    """

    _tags = {
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
        "python_version": ">=3.8",
        "python_dependencies": ["holidays"],
    }

    def __init__(self, market, years=None, expand=True, observed=True, language=None):
        self.market = market
        self.years = years
        self.expand = expand
        self.observed = observed
        self.language = language

        super().__init__()

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
            self.market,
            years=self.years,
            expand=self.expand,
            observed=self.observed,
            language=self.language,
        )

        Xt = X.copy(deep=True)

        if isinstance(Xt.index, pandas.DatetimeIndex):
            dates = Xt.index
        elif isinstance(Xt.index, pandas.PeriodIndex):
            dates = Xt.index.to_timestamp()
        else:
            raise TypeError(f"index type is unsupported: {type(Xt.index)}")

        # dates.isin(holidays_data) behave surprisingly
        # it fails to detect correctly first time
        # but works correctly while using one at a time
        Xt["financial_holidays"] = [date in holidays_data for date in dates]

        return Xt

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
        del parameter_set  # avoid being detected as unused by ``vulture`` like tools

        params = [{"market": "NYSE"}]

        return params
