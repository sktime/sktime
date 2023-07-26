"""Base classes for adapting other forecasters to sktime framework."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = [
    "_ProphetAdapter",
    "_StatsModelsAdapter",
    "_TbatsAdapter",
    "_PmdArimaAdapter",
    "_StatsForecastAdapter",
    "_GeneralisedStatsForecastAdapter",
]

from sktime.forecasting.base.adapters._fbprophet import _ProphetAdapter
from sktime.forecasting.base.adapters._generalised_statsforecast import (
    _GeneralisedStatsForecastAdapter,
)
from sktime.forecasting.base.adapters._pmdarima import _PmdArimaAdapter
from sktime.forecasting.base.adapters._statsforecast import _StatsForecastAdapter
from sktime.forecasting.base.adapters._statsmodels import _StatsModelsAdapter
from sktime.forecasting.base.adapters._tbats import _TbatsAdapter
