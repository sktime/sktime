
.. _benchmarking_ref:

Benchmarking
============

The :mod:`sktime.benchmarking` module contains functionality to perform benchmarking.

Base
----

.. currentmodule:: sktime.benchmarking.benchmarks

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseBenchmark

.. currentmodule:: sktime.benchmarking.forecasting

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ForecastingBenchmark

.. currentmodule:: sktime.benchmarking.base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseMetric
    BaseResults
    BaseDataset
    HDDBaseResults
    HDDBaseDataset

.. currentmodule:: sktime.benchmarking.data

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    UEADataset
    RAMDataset


.. currentmodule:: sktime.benchmarking.evaluation

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Evaluator

.. currentmodule:: sktime.benchmarking.experiments

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    run_clustering_experiment
    load_and_run_clustering_experiment
    run_classification_experiment
    load_and_run_classification_experiment

.. currentmodule:: sktime.benchmarking.orchestration

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    Orchestrator

.. currentmodule:: sktime.benchmarking.results

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    RAMResults
    HDDResults

.. currentmodule:: sktime.benchmarking.strategies

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseStrategy
    BaseSupervisedLearningStrategy
    TSCStrategy
    TSRStrategy

.. currentmodule:: sktime.benchmarking.tasks

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseTask
    TSCTask
    TSRTask

.. currentmodule:: sktime.benchmarking.metrics

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PairwiseMetric
    AggregateMetric
