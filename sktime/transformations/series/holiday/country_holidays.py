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
    language : str, optional
        The language which the returned holiday names will be translated into. It must
        be an ISO 639-1 (2-letter) language code. [4]_ If the language translation is
        not supported the original holiday names will be used.
    categories : Tuple[str], optional
        requested holiday categories.

    References
    ----------
    .. [1] https://github.com/vacanza/python-holidays
    .. [2] https://www.iso.org/obp/ui/#search/code/
    .. [3] https://python-holidays.readthedocs.io/en/latest/#available-countries
    .. [4] https://www.loc.gov/standards/iso639-2/php/English_list.php

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

        return pandas.Series(country_holidays)

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
        """
        del parameter_set  # avoid being detected as unused by ``vulture`` like tools

        params = [
            {"country": "US"},
            {"country": "US", "subdiv": "CA"},
            {"country": "US", "years": 2000},
        ]

        return params


__all__ = ["CountryHolidaysTransformer"]
