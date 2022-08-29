.. _data_format:

File Format Specifications
==========================

.. toctree::
    :maxdepth: 1
    :hidden: true

    Purpose
    Overview
    Instructions

This document formalizes string identifiers for `.ts` and `.tsf` file format.
String identifiers refer to strings beginning with ``@`` in the dataset.

To rather understand how data is stored and loaded from these formats, visit
`loading data. <https://github.com/alan-turing-institute/sktime/blob/main/examples/loading_data.ipynb>`_

Purpose
-------

These identifiers serve the purpose of parsing metadata for the provided dataset and mark the beginning of data.
``sktime's`` core loader/writer functions relies on their existence to correctly load data into memory.

Metadata include:

* Name of the dataset
* Does it include timestamps
* Does it include missing values
* Does it contain only one variable
* Number of dimensions, in case of a multivariate problem
* Do all instances have the same length
* Labels for the class

Overview
--------
These string identifiers are present before the actual data and are to be written at the start of line only.
The general format of string identifiers are: ``@<identifier> [value]``, except for ``@data`` where there is no trailing information.

.. note::
    Since these datasets are often from different sources (see `tsregression`_ and `timeseriesclassification.com`_)
    There may be minor conflict in their naming conventions (lowercase vs. camelCase). ``sktime`` internally takes care of such inconsistencies.

    However, if you run into an inconsistency that isn't already taken care of, kindly consider opening an `issue`_.

.. list-table::
    :widths: 10 10 40 40
    :header-rows: 1

    * - Identifier
      - Datatype
      - Description
      - Additional Comments
    * - ``@problemname``
      - ``string``
      - The name of the dataset.
      -
    * - ``@timestamps``
      - ``bool``
      - Whether timestamps are present.
      -
    * - ``@missing``
      - ``bool``
      - Whether there are missing values.
      -
    * - ``@univariate``
      - ``bool``
      - Whether there is only one variable.
      -
    * - ``@dimensions``
      - ``int``
      - The number of variables.
      - Only present when ``@univariate=False``.
    * - ``@equallength``
      - ``bool``
      - Whether each instance has equal length.
      -
    * - ``@serieslength``
      - ``int``
      - Number of timestamps in each instance.
      - Only present if ``@equallength=True``.
    * - ``@targetlabel``
      - ``bool``
      - Whether there is a target label.
      - Exclusive to regression data.
    * - ``@classlabel``
      - ``bool`` / ``string``
      - Whether class labels are present.
      - Exclusive to classification data; when ``True``, also contains space-seperated strings as labels.
    * - ``@data``
      -
      - Marks the beginning of data.
      - The data begins from the next line.

Instructions
------------

.. _issue: https://github.com/alan-turing-institute/sktime/issues
.. _tsregression: http://tseregression.org/
.. _timeseriesclassification.com: http://www.timeseriesclassification.com/index.php
