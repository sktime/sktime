
.. _param_est_ref:

Parameter estimation
====================

The :mod:`sktime.param_est` module contains parameter estimators, e.g., for
seasonality, and utilities for plugging the estimated parameters into other estimators.
For example, seasonality estimators can be combined with any seasonal forecaster
to an auto-seasonality version.

All parameter estimators in ``sktime`` can be listed using the
``sktime.registry.all_estimators`` utility,
using ``estimator_types="param_est"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

A full table with tag based search is also available on the
`Estimator Search Page <https://www.sktime.net/en/latest/estimator_overview.html>`_
(select "parameter estimator" in the "Estimator type" dropdown).


Parameter estimators
--------------------

Composition
~~~~~~~~~~~

.. currentmodule:: sktime.param_est.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ParamFitterPipeline

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FunctionParamFitter

.. currentmodule:: sktime.param_est.plugin

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PluginParamsForecaster
    PluginParamsTransformer

Naive
~~~~~

.. currentmodule:: sktime.param_est.fixed

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FixedParams

Seasonality estimators
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.param_est.seasonality

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SeasonalityACF
    SeasonalityACFqstat
    SeasonalityPeriodogram

Stationarity estimators
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.param_est.stationarity

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    StationarityADF
    StationarityKPSS
    StationarityADFArch
    StationarityDFGLS
    StationarityPhillipsPerron
    StationarityKPSSArch
    StationarityZivotAndrews
    StationarityVarianceRatio
