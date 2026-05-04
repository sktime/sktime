
.. _param-est_ref:

Parameter estimation
====================

The :mod:`sktime.param-est` module contains parameter estimators, e.g., for
seasonality, and utilities for plugging the estimated parameters into other estimators.
For example, seasonality estimators can be combined with any seasonal forecaster
to an auto-seasonality version.

All parameter estimators in ``sktime`` can be listed using the
``sktime.registry.all_estimators`` utility,
using ``estimator-types="param-est"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

A full table with tag based search is also available on the
:doc:`Estimator Search Page </estimator-overview>`
(select "parameter estimator" in the "Estimator type" dropdown).


Parameter estimators
--------------------

Composition
~~~~~~~~~~~

.. currentmodule:: sktime.param-est.compose

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ParamFitterPipeline

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FunctionParamFitter

.. currentmodule:: sktime.param-est.plugin

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PluginParamsForecaster
    PluginParamsTransformer

Naive parameter estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.param-est.fixed

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    FixedParams

Seasonality estimators
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.param-est.seasonality

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SeasonalityACF
    SeasonalityACFqstat
    SeasonalityPeriodogram

Stationarity tests
~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.param-est.stationarity

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
    BreakvarHeteroskedasticityTest

Lag and autocorrelation estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.param-est.lag

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ARLagOrderSelector
    AcorrLjungbox

Residual tests and estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.param-est.residuals

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    JarqueBera

Cointegration
~~~~~~~~~~~~~

.. currentmodule:: sktime.param-est.cointegration

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    JohansenCointegration


Impulse and Shock Response Estimation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.param_est.impulse

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ImpulseResponseFunction
