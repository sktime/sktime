
.. _benchmarking_ref:

Benchmarking
============

The :mod:`sktime.benchmarking` module contains functionality to perform benchmarking.

Base
----

.. currentmodule:: sktime.benchmarking.benchmarks

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    BaseBenchmark

.. currentmodule:: sktime.benchmarking.forecasting

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    ForecastingBenchmark

.. currentmodule:: sktime.benchmarking.base

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    BaseMetric
    BaseResults
    BaseDataset
    HDDBaseResults
    HDDBaseDataset

.. currentmodule:: sktime.benchmarking.data

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    UEADataset
    RAMDataset


.. currentmodule:: sktime.benchmarking.evaluation

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    Evaluator

.. currentmodule:: sktime.benchmarking.experiments

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: function.rst

    run_clustering_experiment
    load_and_run_clustering_experiment
    run_classification_experiment
    load_and_run_classification_experiment

.. currentmodule:: sktime.benchmarking.orchestration

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    Orchestrator

.. currentmodule:: sktime.benchmarking.results

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    RAMResults
    HDDResults

.. currentmodule:: sktime.benchmarking.strategies

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    BaseStrategy
    BaseSupervisedLearningStrategy
    TSCStrategy
    TSRStrategy

.. currentmodule:: sktime.benchmarking.tasks

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    BaseTask
    TSCTask
    TSRTask

.. currentmodule:: sktime.benchmarking.metrics

.. autosummary::
    :signatures: none
    :toctree: auto_generated/
    :template: class.rst

    PairwiseMetric
    AggregateMetric
