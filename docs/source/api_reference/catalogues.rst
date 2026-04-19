.. _catalogues_ref:

Catalogues
==========

``catalogues`` provide a unified interface to list, query, and retrieve
collections of sktime objects such as datasets, estimators, metrics, and
cross-validation strategies.

Catalogues can make reproducing and extending benchmarking studies and
competitions easier and convenient, by reducing boilerplate code.

A catalogue can be used to:

* inspect what items exist in a given collection,
* retrieve items by category (e.g., only datasets, only metrics),
* obtain either string specifications or fully constructed objects,
* describe metadata about the catalogue such as number of items, authorship,
  and source.

Forecasting Catalogues
----------------------

.. currentmodule:: sktime.catalogues.forecasting

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    dummy._dummy_forecasting.DummyForecastingCatalogue

Classification Catalogues
-------------------------

.. currentmodule:: sktime.catalogues.classification

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    dummy._dummy_classification.DummyClassificationCatalogue
