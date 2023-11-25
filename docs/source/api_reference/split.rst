
.. _split_ref:

Splitters
=========

The :mod:`sktime.split` module contains algorithms for splitting and resampling data.

All splitters in ``sktime`` can be listed using the ``sktime.registry.all_estimators`` utility,
using ``estimator_types="splitter"``, optionally filtered by tags.
Valid tags can be listed using ``sktime.registry.all_tags``.


Splitting utilities
-------------------

``temporal_train_test_split`` is a quick utility function for
splitting a single time series into training and test fold.

Forecasting users interested in performance evaluation are advised
to use full backtesting instead of a single split, e.g., via ``evaluate``,
see :ref:`forecasting API reference <forecasting_ref>`.

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    temporal_train_test_split


Time index splitters
--------------------

Time index splitters split one or multiple time series by temporal order.
They are typically used in both evaluation and tuning of forecasters.

.. currentmodule:: sktime.split

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    CutoffSplitter
    SingleWindowSplitter
    SlidingWindowSplitter
    ExpandingWindowSplitter
    ExpandingGreedySplitter
    TemporalTrainTestSplitter


Time index splitter composition
-------------------------------

The following splitters are compositions that can be used to create
more complex time index based splitting strategies.

.. currentmodule:: sktime.split

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SameLocSplitter
    TestPlusTrainSplitter
