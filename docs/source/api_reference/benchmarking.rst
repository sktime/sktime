
.. _benchmarking_ref:

Benchmarking
============

The :mod:`sktime.benchmarking` module contains functionality to perform benchmarking.

Benchmarking Framework v2
-------------------------

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

.. currentmodule:: sktime.benchmarking.classification

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ClassificationBenchmark


Storage Backends
----------------

.. currentmodule:: sktime.benchmarking._storage_handlers
.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    JSONStorageHandler
    ParquetStorageHandler
    CSVStorageHandler
    NullStorageHandler


Post-hoc evaluators
-------------------

Post-hoc statistical evaluators consume the results of ``BaseBenchmark.run`` and
compute ranking, omnibus / pairwise significance tests, and critical-difference
diagrams.

.. currentmodule:: sktime.benchmarking.post_hoc

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    BaseBenchmarkAnalyzer
    RankEvaluator
    FriedmanEvaluator
    NemenyiEvaluator
    WilcoxonEvaluator
    SignTestEvaluator
    RanksumEvaluator
    TTestEvaluator
    CriticalDifferenceDiagram


Benchmarking Framework v1
-------------------------

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


Post-hoc tests and utilities
----------------------------

.. currentmodule:: sktime.benchmarking.critical_difference

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    plot_critical_difference
