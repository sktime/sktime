
.. _performance_metric_ref:

Performance metrics
===================

The :mod:`sktime.performance_metrics` module contains metrics for evaluating and tuning time series models.

.. automodule:: sktime.performance_metrics
    :no-members:
    :no-inherited-members:

Forecasting
-----------

Point forecasts - classes
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.performance_metrics.forecasting

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    MeanAbsoluteScaledError
    MedianAbsoluteScaledError
    MeanSquaredScaledError
    MedianSquaredScaledError
    MeanAbsoluteError
    MeanSquaredError
    MedianAbsoluteError
    MedianSquaredError
    GeometricMeanAbsoluteError
    GeometricMeanSquaredError
    MeanAbsolutePercentageError
    MedianAbsolutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError
    MeanRelativeAbsoluteError
    MedianRelativeAbsoluteError
    GeometricMeanRelativeAbsoluteError
    GeometricMeanRelativeSquaredError
    MeanAsymmetricError
    MeanLinexError
    RelativeLoss

Point forecasts - functions
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.performance_metrics.forecasting

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    make_forecasting_scorer
    mean_absolute_scaled_error
    median_absolute_scaled_error
    mean_squared_scaled_error
    median_squared_scaled_error
    mean_absolute_error
    mean_squared_error
    median_absolute_error
    median_squared_error
    geometric_mean_absolute_error
    geometric_mean_squared_error
    mean_absolute_percentage_error
    median_absolute_percentage_error
    mean_squared_percentage_error
    median_squared_percentage_error
    mean_relative_absolute_error
    median_relative_absolute_error
    geometric_mean_relative_absolute_error
    geometric_mean_relative_squared_error
    mean_asymmetric_error
    mean_linex_error
    relative_loss

Quantile and interval forecasts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.performance_metrics.forecasting.probabilistic

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    PinballLoss
    EmpiricalCoverage
    ConstraintViolation

Distribution forecasts
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.performance_metrics.forecasting.probabilistic

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    CRPS
    LogLoss
    SquaredDistrLoss


Time series segmentation
------------------------

.. currentmodule:: sktime.performance_metrics.annotation

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    count_error
    hausdorff_error
    prediction_ratio
