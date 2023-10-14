.. _datasets_ref:

Datasets
========

The ``datasets`` module contains:

* loaders which fetch datasets from data repositories on the internet,
  and retrieve them as in-memory datasets in ``sktime`` compatible formats
* loaders which fetch an individual dataset, usually for illustration purposes
* utilities to write to, and load from, time series specific file formats

Loaders from dataset repositories
---------------------------------

These loaders access dataset respositories on the internet and fetch one or multiple
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
    load_UCR_UEA_dataset


Individual datasets
-------------------

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
