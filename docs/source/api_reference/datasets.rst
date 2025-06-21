.. _datasets_ref:

Datasets
========

The ``datasets`` module contains:

* dataset objects, which are in-memory representations of time series datasets
* loaders which fetch datasets from data repositories on the internet,
  and retrieve them as in-memory datasets in ``sktime`` compatible formats
* loaders which fetch an individual dataset, usually for illustration purposes
* toy data generators for didactic and illustrative purposes
* utilities to write to, and load from, time series specific file formats

Forecasting datasets
--------------------
.. currentmodule:: sktime.datasets.forecasting

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    airline.Airline
    hierarchical_sales_toydata.HierarchicalSalesToydata
    longley.Longley
    lynx.Lynx
    m5_competition.M5Dataset
    macroeconomic.Macroeconomic
    shampoo_sales.ShampooSales
    solar.Solar
    uschange.USChange

Classification datasets
-----------------------

.. currentmodule:: sktime.datasets.classification

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    arrow_head.ArrowHead
    basic_motions.BasicMotions
    gunpoint.GunPoint
    italy_power_demand.ItalyPowerDemand
    japanese_vowels.JapaneseVowels
    osuleaf.OSULeaf
    plaid.PLAID

Regression datasets
-------------------

.. currentmodule:: sktime.datasets.regression

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    tecator.Tecator

Creating Custom Classification Datasets
------------------------

You can define your own classification dataset by subclassing BaseDataset from sktime.datasets._base.

Example:
.. code-block:: python
   from sktime.datasets._base import BaseDataset
   
   class MyClassificationDataset(BaseDataset):
      def __init__(self):
           super().__init__()
           self.metadata = {
               "task": "classification",
               "n_classes": 3,
               "n_instances": 150,
               "univariate": True,
               "equal_length": True,
           }

      def _load(self):
        # Load or generate your dataset here
        from sklearn.datasets import make_classification
        import pandas as pd
        import numpy as np

        X_raw, y = make_classification(n_samples=150, n_features=5, n_classes=3, random_state=42)
        X = pd.DataFrame([pd.Series(row) for row in X_raw])
        return X, y

Usage:

.. code-block:: python
   dataset = MyClassificationDataset()
   X, y = dataset.load()
   X_train, y_train, X_test, y_test = dataset.load(split=["X_train", "y_train", "X_test", "y_test"])

Loaders
-------

Loaders from dataset repositories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These loaders access dataset repositories on the internet and fetch one or multiple
datasets from there, individual datasets specifiable as strings.

These loaders can be used to access reference datasets for benchmarking.

.. automodule:: sktime.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: sktime.datasets

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    load_forecastingdata
    load_fpp3
    load_m5
    load_UCR_UEA_dataset


Individual datasets
~~~~~~~~~~~~~~~~~~~

These loaders fetch a commonly used individual dataset,
usually for illustration purposes.

Single time series
^^^^^^^^^^^^^^^^^^

.. automodule:: sktime.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: sktime.datasets

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    load_airline
    load_longley
    load_lynx
    load_macroeconomic
    load_shampoo_sales
    load_solar
    load_uschange

Panels of time series
^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sktime.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: sktime.datasets

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    load_acsf1
    load_arrow_head
    load_basic_motions
    load_gunpoint
    load_italy_power_demand
    load_japanese_vowels
    load_macroeconomic
    load_osuleaf
    load_tecator


Toy data generators
~~~~~~~~~~~~~~~~~~~

Hierarchical time series data
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: sktime.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: sktime.datasets

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    load_hierarchical_sales_toydata


Loading from and writing to files
---------------------------------

These utilities load and write from time series specific data formats.

Note: for loading/writing from formats not specific to time series,
use common utilities such as ``pandas.read_csv``

.. automodule:: sktime.datasets
    :no-members:
    :no-inherited-members:

.. currentmodule:: sktime.datasets

.. autosummary::
    :toctree: auto_generated/
    :template: function.rst

    load_from_arff_to_dataframe
    load_from_tsfile
    load_from_tsfile_to_dataframe
    load_from_ucr_tsv_to_dataframe
    load_from_long_to_dataframe
    load_tsf_to_dataframe
    write_panel_to_tsfile
    write_dataframe_to_tsfile
    write_ndarray_to_tsfile
    write_tabular_transformation_to_arff
    write_results_to_uea_format
