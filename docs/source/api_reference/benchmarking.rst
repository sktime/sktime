
.. _benchmarking_ref:

Benchmarking
============

The :mod:`sktime.benchmarking` module contains functionality to perform benchmarking.

Benchmarking Framework
----------------------

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


Benchmark analyzers
-------------------

Benchmark analyzers consume the results of ``BaseBenchmark.run`` and
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
