.. _data_format:

File Format Specifications
==========================

.. toctree::
    :maxdepth: 1
    :hidden: false

    Introduction
    Purpose
    Overview
    Instructions

Introduction
------------
Suppose you have comma seperated values for your dataset as:

.. code-block:: text
   :linenos:
   :name: data-format-intro-1

   (2004-08-10 18:00:00,1130.0),(2004-08-10 19:00:00,1217.75),(2004-08-10 20:00:00,1134.75),(2004-08-10 21:00:00,1155.5),
   (2004-08-10 22:00:00,1151.0):(2004-08-10 18:00:00,1144.24),(2004-08-11 19:00:00,1111.25),(2004-08-11 20:00:00,1065.75),
   (2004-08-11 21:00:00,992.5),(2004-08-11 22:00:00,905.76):(2004-08-11 18:00:00,903.35),(2004-08-11 19:00:00,941.0),
   (2004-08-11 20:00:00,1073.6666666667),(2004-08-11 21:00:00,1113.5),(2004-08-11 22:00:00,1100.6):3.2

or maybe without timestamps:

.. code-block:: text
   :linenos:
   :name: data-format-intro-2

   0.190118275,0.278452167,0.36494119700000005,0.44635393,0.519787268,0.582824302,0.633666574,
   0.44362301,2.39413786,0.0,0.0,0.32292562,0.286182571,0.0,0.0,0.0,0.565279958:7.9

How would ``sktime`` know how to interpret the dataset (whether or not it contains multiple dimensions, whether or not
all instances are of same length etc.)? Additionally, how would another user know about these specifications regarding the dataset?
The answer is by having additional information at the start of file, simply by
adding lines in a specific format at the start of the file.

This document formalizes string identifiers used in `.ts` and `.tsf` file format. Such files can be opened via any basic
editor like notepad for visual inspection. String identifiers refer to strings beginning with ``@`` in the file.

To rather understand how data is stored and loaded from these formats, visit
`loading data. <https://github.com/alan-turing-institute/sktime/blob/main/examples/loading_data.ipynb>`_

Purpose
-------
``.ts`` files contains information in the following order:

1. Comments to describe the dataset. (begins with ``#`` in the file)
2. String Identifiers to give information about the metadata (begins with ``@`` in the file).
3. Actual dataset.

`Basic Motion.ts`_ might be helpful for a concrete example.

These identifiers serve the purpose of containing metadata for the provided dataset and also marks the beginning of data.
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
Each identifier must be present at a separate line.
The format of string identifiers are: ``@<identifier> [value]``, except for ``@data`` where there is no trailing information.

.. note::
    Since these datasets are often from different sources (see `tsregression`_ and `timeseriesclassification.com`_)
    There may be minor conflict in their naming conventions (lowercase vs. camelCase). ``sktime`` internally takes care of such inconsistencies.

    For this document, we will only use lowercase to represent the identifier.

    However, if you run into an inconsistency that isn't already taken care of, kindly consider opening an `issue`_.

.. list-table::
    :widths: 10 40 30 20
    :header-rows: 1

    * - Identifier
      - Description
      - Additional Comments
      - Example
    * - ``@problemname``
      - The name of the dataset.
      - Value cannot be space seperated
      - ``BasicMotions``
    * - ``@timestamps``
      - Whether timestamps are present.
      - ``true`` / ``false`` only
      - ``false``
    * - ``@missing``
      - Whether there are missing values.
      - ``true`` / ``false`` only
      - ``false``
    * - ``@univariate``
      - Whether there is only one variable.
      - ``true`` / ``false`` only
      - ``false``
    * - ``@dimension``
      - The number of variables.
      - Only present when ``@univariate=false``.
      - 6
    * - ``@equallength``
      - Whether each instance has equal length.
      - ``true`` / ``false`` only
      - ``true``
    * - ``@serieslength``
      - Number of timestamps in each instance.
      - Only present if ``@equallength=true``.
      - 100
    * - ``@targetlabel``
      - Whether there is a target label.
      - Exclusive to regression data; ``true`` / ``false`` only
      - ``true``
    * - ``@classlabel``
      - Whether class labels are present.
      - Exclusive to classification data; when ``true``, also contains space-seperated int/strings as labels.
      - ``true Standing Running Walking Badminton``
    * - ``@data``
      - Marks the beginning of data.
      - The data begins from the next line.
      - \-

Instructions
------------
This section provides full set of instructions to create your format specification for your dataset that is compatible with ``sktime``.
Remember that this begins with the assumption that you have the dataset readily available in
expected `format <https://github.com/alan-turing-institute/sktime/blob/main/examples/loading_data.ipynb>`_.

Few points to keep in mind while creating the dataset:

  1. The general order of identifiers **does not** matter with the exception that ``@data`` should be the last string identifier.
  2. For lines containing an identifier, it **must** begin with it.
  3. The **only** place to give a space is between identifier and its corresponding value.
  4. Avoid having newline characters in between lines.
  5. Follow the "comments, identifiers, data" order.

.. note::
   The running example dataset that we will be using for demonstration is at the start of this page.

1. *Create a descriptive comment*
    Few initial lines of the file should ideally be given to describing the dataset. This is optional but gives context
    about the dataset. A comment line begins with ``#``.

.. code-block:: text
   :linenos:
   :name: data-format-step1-text

    # The following dataset is generated using sensor S in the apparatus A as shown in the following
    # link: https://example.com/. We receive three individual variables, collected within the time duration of 4 hours.
    # There are no missing values in the dataset and timestamps are also included.
    # For more information about how data was collected, visit the above mentioned link.

2. *Add those metadata that are common to both classification and regression data*

    - Add the problem name: ``@problemName Example``
    - Add info about having missing contents: ``@missing false``
    - Add info about timestamps: ``@timestamps true``
    - Add info if dataset has only one dimension: ``@univariate false``
    - Since univariate is ``false``, add info about number of dimensions, skip otherwise: ``@dimension 3``
    - Add info whether all instances have equal length: ``@equallength true`` (Shown part has only instance for brevity)
    - If above is true, add info about length of an instance, skip otherwise: ``@serieslength 5``

The resulting file should now look like:

.. code-block:: text
   :linenos:
   :name: data-format-step2-text

    # The following dataset is generated using sensor S in the apparatus A as shown in the following
    # link: https://example.com/. We receive three individual variables, collected within the time duration of 4 hours.
    # There are no missing values in the dataset and timestamps are also included.
    # For more information about how data was collected, visit the above mentioned link.
    @problemName Example
    @missing false
    @timestamps true
    @univariate false
    @dimension 3
    @equallength true
    @serieslength 5

3. Now depending if your dataset is:

   a. Regression-based: add ``@targetlabel`` identifier.
      For our example, it will be ``@targetlabel true`` since we have a response variable, otherwise it will be ``false``.
   b. Classification-based: add ``@classlabel`` identifier.
      If there is no response variable it will have a value of ``false``. If ``true``, you can optionally provide the class labels
      in space seperated manner:

      - eg: Three string labels: ``@classlabel true good bad neutral``
      - eg: Two integer labels: ``@classlabel true 0 1``

4. Finally, add the identifier ``@data`` followed by the values in the newline.

For our example, that would result in:

.. code-block:: text
   :linenos:
   :name: data-format-result

    # The following dataset is generated using sensor S in the apparatus A as shown in the following
    # link: https://example.com/. We receive three individual variables, collected within the time duration of 4 hours.
    # There are no missing values in the dataset and timestamps are also included.
    # For more information about how data was collected, visit the above mentioned link.
    @problemName Example
    @missing false
    @timestamps true
    @univariate false
    @dimension 3
    @equallength true
    @serieslength 5
    @targetlabel true
    @data
    (2004-08-10 18:00:00,1130.0),(2004-08-10 19:00:00,1217.75),(2004-08-10 20:00:00,1134.75),(2004-08-10 21:00:00,1155.5),
    (2004-08-10 22:00:00,1151.0):(2004-08-10 18:00:00,1144.24),(2004-08-11 19:00:00,1111.25),(2004-08-11 20:00:00,1065.75),
    (2004-08-11 21:00:00,992.5),(2004-08-11 22:00:00,905.76):(2004-08-11 18:00:00,903.35),(2004-08-11 19:00:00,941.0),
    (2004-08-11 20:00:00,1073.6666666667),(2004-08-11 21:00:00,1113.5),(2004-08-11 22:00:00,1100.6):3.2


This concludes how to create string identifiers for `.ts` and `.tsf` format. To learn more about ``sktime``, visit
`tutorials`_ page.


.. _Basic Motion.ts: https://github.com/alan-turing-institute/sktime/blob/main/sktime/datasets/data/BasicMotions/BasicMotions_TEST.ts
.. _issue: https://github.com/alan-turing-institute/sktime/issues
.. _tsregression: http://tseregression.org/
.. _timeseriesclassification.com: http://www.timeseriesclassification.com/index.php
.. _tutorials: https://www.sktime.org/en/stable/tutorials.html
