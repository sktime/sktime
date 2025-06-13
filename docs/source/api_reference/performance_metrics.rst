
.. _performance_metric_ref:

Performance metrics
===================

The :mod:`sktime.performance_metrics` module contains metrics for evaluating and tuning time series models.

All parameter estimators in ``sktime`` can be listed using the
``sktime.registry.all_estimators`` utility,
using ``estimator_types="metric"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.

A full table with tag based search is also available on the
:doc:`Estimator Search Page </estimator_overview>`
(select "metric" in the "Estimator type" dropdown).


.. automodule:: sktime.performance_metrics
    :no-members:
    :no-inherited-members:

Forecasting
-----------

Point forecasts - classes
~~~~~~~~~~~~~~~~~~~~~~~~~

Average losses
^^^^^^^^^^^^^^

.. currentmodule:: sktime.performance_metrics.forecasting

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    MeanAbsoluteError
    MeanSquaredError
    MedianAbsoluteError
    MedianSquaredError

Percentage errors
^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    MeanAbsolutePercentageError
    MedianAbsolutePercentageError
    MeanSquaredPercentageError
    MedianSquaredPercentageError
    MeanSquaredErrorPercentage

Scaled errors
^^^^^^^^^^^^^


.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    MeanAbsoluteScaledError
    MedianAbsoluteScaledError
    MeanSquaredScaledError
    MedianSquaredScaledError

Relative errors
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    MeanRelativeAbsoluteError
    MedianRelativeAbsoluteError
    RelativeLoss

Geometric errors
^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    GeometricMeanAbsoluteError
    GeometricMeanSquaredError
    GeometricMeanRelativeAbsoluteError
    GeometricMeanRelativeSquaredError

Under- and over-prediction errors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    MeanAsymmetricError
    MeanLinexError


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
    IntervalWidth

Distribution forecasts
~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: sktime.performance_metrics.forecasting.probabilistic

.. autosummary::
    :toctree: auto_generated/
    :template: class_with_call.rst

    AUCalibration
    CRPS
    LogLoss
    SquaredDistrLoss


Detection tasks
---------------

Detection metrics can be applied to compare ground truth events with detected events,
and ground truth segments with detected segments.

Detection metrics are typically designed for either:

* point events, i.e., annotated time stamps, or
* segments, i.e., annotated time intervals.

The metrics in ``sktime`` can be used for both types of detection tasks:

* segmentation metrics interpret point events as segment boundaries, separating consecutive segments
* point event metrics are applied to segments by considering their boundaries as point events


Event detection - anomalies, outliers
-------------------------------------

.. currentmodule:: sktime.performance_metrics.detection

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: function.rst

    DirectedChamfer
    DirectedHausdorff
    DetectionCount
    WindowedF1Score
    TimeSeriesAUPRC

Segment detection
-----------------

.. currentmodule:: sktime.performance_metrics.detection

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    RandIndex


Legacy detection metrics
------------------------

These metrics do not follow the standard API and will be deprecated in the future.

.. currentmodule:: sktime.performance_metrics.annotation

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    count_error
    hausdorff_error
    prediction_ratio
