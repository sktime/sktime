.. _data_format:

Data Format Specifications
==========================

This section provides specifications for:

* python in-memory data containers used in ``sktime`` (e.g., time series, panel data, etc.)
* serialized file formats used by ``sktime`` (e.g., ts)

For utilities to check and convert data formats, see the API reference on :ref:`utils_ref`.


In-memory Data Specifications
-----------------------------

``sktime`` uses a variety of in-memory data containers to represent time series data.

The in-memory specifications are listed by abstract data type, also referred to
as data :term:`scitype` (scientific type) throughout the documentation.

The core scitypes in ``sktime`` are:

- ``Series``: a single time series
- ``Panel``: a flat collection of time series, also called panel of time series, or panel data
- ``Hierarchical``: a hierarchical collection of time series
- ``Table``: a data frame table, as implemented for instance by ``pandas.DataFrame``

Each scitype is sub-typed with property fields such as ``is_univariate``
(the time series is univariate yes/no),
``n_instances`` (number of instances in a panel or hierarchical collection), etc.

Concrete data types in ``sktime`` are implementations of these abstract data types,
also referred to as data :term:`mtype` (machine type) throughout the documentation.

Full specifications of the abstract data types,
with their subtypes, can be accessed below:

.. currentmodule:: sktime.datatypes._series._base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScitypeSeries

.. currentmodule:: sktime.datatypes._panel._base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScitypePanel

.. currentmodule:: sktime.datatypes._hierarchical._base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScitypeHierarchical

.. currentmodule:: sktime.datatypes._table._base

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    ScitypeTable


.. _mtypes_series:

``Series`` mtype specifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Series`` mtype represents a single time series.

.. currentmodule:: sktime.datatypes._series._check

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    SeriesPdDataFrame
    SeriesPdSeries
    SeriesNp2D
    SeriesXarray
    SeriesDask
    SeriesPolarsEager
    SeriesGluontsList
    SeriesGluontsPandas


.. _mtypes_panel:

``Panel`` mtype specifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Panel`` mtype represents a flat collection of time series.

.. currentmodule:: sktime.datatypes._panel._check

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    PanelPdMultiIndex
    PanelNp3D
    PanelDfList
    PanelDask
    PanelPolarsEager
    PanelGluontsList
    PanelGluontsPandas


.. _mtypes_hierarchical:

``Hierarchical`` mtype specifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Hierarchical`` mtype represents a hierarchical collection of time series.

.. currentmodule:: sktime.datatypes._hierarchical._check

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    HierarchicalPdMultiIndex
    HierarchicalDask
    HierarchicalPolarsEager


.. _mtypes_table:

``Table`` mtype specifications
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Table`` mtype represents a (non-temporal) data frame table.

.. currentmodule:: sktime.datatypes._table._check

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    TablePdDataFrame
    TablePdSeries
    TableNp1D
    TableNp2D
    TableListOfDict
    TablePolarsEager

Serialized File Format Specifications
-------------------------------------

``sktime`` supports a variety of file formats for serialized data,
specific to storing time series data.

Specifications for file formats specific to ``sktime`` are provided below:

.. toctree::
    :maxdepth: 1

    file_specifications/ts
