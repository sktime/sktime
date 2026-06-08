.. _datasets_ref:

Datasets
========

The ``datasets`` module contains:

* dataset objects, which are in-memory representations of time series datasets.
  The programmatic way to represent and access datasets in ``sktime``.
* loaders which fetch datasets from data repositories on the internet,
  and retrieve them as in-memory datasets in ``sktime`` compatible formats
* loaders which fetch an individual dataset, usually for illustration purposes
* toy data generators for didactic and illustrative purposes
* utilities to write to, and load from, time series specific file formats

Forecasting datasets
--------------------

Dataset repositories
~~~~~~~~~~~~~~~~~~~~

Interfaces to dataset repositories, instances of the classes
represent different datasets. Downloaded from the ``sktime`` ``huggingface`` space,
cached on first use.

.. currentmodule:: sktime.datasets.forecasting

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    monash.ForecastingData

Individual datasets - onboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Individual datasets distributed with ``sktime``, available without internet access.

.. currentmodule:: sktime.datasets.forecasting

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    airline.Airline
    monash.ForecastingData
    hierarchical_sales_toydata.HierarchicalSalesToydata
    longley.Longley
    lynx.Lynx
    m5_competition.M5Dataset
    macroeconomic.Macroeconomic
    pbs.PBS
    shampoo_sales.ShampooSales
    uschange.USChange

Individual datasets - downloaded
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Individual datasets downloadable from the ``sktime`` ``huggingface`` space,
cached on first use.

.. currentmodule:: sktime.datasets.forecasting

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    m5_competition.M5Dataset
    solar.Solar

Classification datasets
-----------------------

Dataset repositories
~~~~~~~~~~~~~~~~~~~~

Interfaces to dataset repositories, instances of the classes
represent different datasets. Downloaded from the ``sktime`` ``huggingface`` space,
cached on first use.


.. currentmodule:: sktime.datasets.classification

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    ucr_uea_archive.UCRUEADataset

Individual datasets - onboard
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Individual datasets distributed with ``sktime``, available without internet access.

.. currentmodule:: sktime.datasets.classification

.. autosummary::
    :recursive:
    :toctree: auto_generated/
    :template: class.rst

    acsf1.ACSF1
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

Dataset loader functions
------------------------

Loaders are raw functions which return datasets in ``sktime`` compatible formats.

For programmatic access to datasets, the dataset objects above should be preferred.

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


File format loaders and writers
-------------------------------

These utilities load and write from time series specific data file formats.

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
