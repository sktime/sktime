.. _data_format:

File Format Specifications
==========================

.. toctree::
    :maxdepth: 1
    :hidden: false

    Overview
    Introduction
    Description
    Instructions
    Illustration

Overview
--------
This document has the following content:

- Introduction: What is a ``.ts`` file, why and when to use it.
- Description: What are the individual components of a ``.ts`` file.
- Instructions: How to create your own ``.ts`` file.
- Illustrations: A running example to tie up above sections.

Introduction
------------
This document formalizes string identifiers used in ``.ts`` file format.
Encoded in ``utf-8``, ``.ts`` files stores time-series dataset and its corresponding metadata and can be
opened via any basic editor like notepad for visual inspection.
String identifiers refer to strings beginning with ``@`` in the file.

``.ts`` files contains information blocks in the following order:

1. All the comments to describe the dataset. (begins with ``#`` in the file)
2. All the String Identifiers to give information about the metadata (begins with ``@`` in the file).
3. Actual dataset.

See `Basic Motion.ts`_ for a concrete example.

These identifiers serve the purpose of containing metadata for the dataset.
``sktime``'s core loader/writer functions relies on their existence to correctly load data into memory.
This is also helpful to provide information about the dataset to a different user not familiar with the dataset.

To understand how the actual data is stored and loaded, visit `loading data`_.

Description
-----------
This section describes different string identifiers present in a ``.ts`` file and how are they employed in the file.

Information that are included in the metadata:

* Name of the dataset
* Does it include timestamps
* Does it include missing values
* Does it contain only one dimension
* Number of dimensions, in case of a multivariate problem
* Do all instances have the same length
* Labels for the class

String identifiers are present before the actual data and are to be written at the start of line only.
Each identifier must be present at a separate line.
The format of individual string identifier is: ``@<identifier> [value]``,
except for ``@data`` where there is no trailing information.

.. note::
    Since these datasets are often from different sources (see `tsregression`_ and `timeseriesclassification.com`_)
    There may be minor conflict in their naming conventions (lowercase vs. camelCase).
    ``sktime`` internally takes care of such inconsistencies.

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
      - Value cannot be space separated
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
      - Whether there is only one dimension for the time series.
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
This section provides full set of instructions to create a format specification ``.ts`` file
for your dataset that is compatible with ``sktime``.

Remember that this begins with the assumption that you have the dataset readily available in
expected `format <https://github.com/alan-turing-institute/sktime/blob/main/examples/loading_data.ipynb>`_.

Few points to keep in mind while creating the dataset:

  1. The general order of identifiers **does not** matter with the exception that ``@data`` should be the last string identifier.
  2. Lines containing an identifier **must** begin with it.
  3. The **only** place a space is allowed is between an identifier and its corresponding value.
  4. Avoid having newline characters in between lines.
  5. Follow the "comments, identifiers, data" order

1. *Create an empty file*
    Open your favorite text editor (even notepad works). We'll add contents into this file before finally saving as a ``.ts`` file.

2. *Write a descriptive comment*
    Few initial lines of the file should ideally be given to describing the dataset. This is optional but gives context
    about the dataset. A comment line begins with ``#``.

3. *Add those metadata that are common to both classification and regression data*

    - Add the problem name: eg:``@problemName Example``
    - Add info about having missing contents: eg:``@missing false``
    - Add info about timestamps: eg:``@timestamps true``
    - Add info if dataset has only one dimension: eg:``@univariate false``
    - Since univariate is eg:``false``, add info about number of dimensions, skip otherwise: eg:``@dimension 3``
    - Add info whether all instances have equal length: eg:``@equallength true``
    - If above is true, add info about length of an instance, skip otherwise: eg:``@serieslength 5``

4. Now depending if your dataset is:

   a. Regression-based: add ``@targetlabel`` identifier followed by ``true`` if the response variable exists, otherwise ``false``.

   b. Classification-based: add ``@classlabel`` identifier.
      If there is no response variable it will have a value of ``false``. If ``true``, you can optionally provide the class labels
      in space seperated manner:

      - eg: Three string labels: ``@classlabel true good bad neutral``
      - eg: Two integer labels:  ``@classlabel true 0 1``

5. Add the identifier ``@data`` followed by the values in the newline.

6. Finally, save the file as ``<CHOOSE_NAME>.ts``. The encoding should be ``utf-8``.

.. Tip::
   File still showing as ``<CHOSEN_NAME>.ts.txt``? Rename it to ``<CHOSEN_NAME>.txt`` then open your terminal and write
   in that directory ``mv <CHOSEN_NAME>.txt <CHOSEN_NAME>.ts``.

Illustration
------------
Here, we provide a running example showing how your file will look like after performing each step in the instructions.

The sample dataset that we will use for this is:

.. code-block:: text
   :linenos:
   :name: data-format-eg-dataset

   (2004-08-10 18:00:00,1130.0),(2004-08-10 19:00:00,1217.75),(2004-08-10 20:00:00,1134.75),(2004-08-10 21:00:00,1155.5),
   (2004-08-10 22:00:00,1151.0):(2004-08-10 18:00:00,1144.24),(2004-08-11 19:00:00,1111.25),(2004-08-11 20:00:00,1065.75),
   (2004-08-11 21:00:00,992.5),(2004-08-11 22:00:00,905.76):(2004-08-11 18:00:00,903.35),(2004-08-11 19:00:00,941.0),
   (2004-08-11 20:00:00,1073.6666666667),(2004-08-11 21:00:00,1113.5),(2004-08-11 22:00:00,1100.6):3.2

1. Let's add some comments to give some context about the dataset:

.. code-block:: text
   :linenos:
   :name: data-format-step-1

    # The following dataset is generated using sensor S in the apparatus A as shown in the following
    # link: https://example.com/. We receive three individual variables, collected within the time duration of 4 hours.
    # There are no missing values in the dataset and timestamps are also included.
    # For more information about how data was collected, visit the datacollection.com.

2. Now, let's add metadata that are common to both classification and regression dataset:

.. code-block:: text
   :linenos:
   :name: data-format-step-2

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

3. Since we have a regression dataset, let's add ``@targetlabel`` as ``true``:

.. code-block:: text
   :linenos:
   :name: data-format-step-3

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

4. Finally, let's mark the beginning of the dataset by adding ``@data`` followed by the data in the newline.

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

5. After saving it as ``sample.ts``, the file is ready to be loaded via sktime.

This concludes how to create string identifiers for ``.ts``  format. To learn more about ``sktime``, visit
`tutorials`_ page.

.. _loading data: https://github.com/alan-turing-institute/sktime/blob/main/examples/loading_data.ipynb
.. _Basic Motion.ts: https://github.com/alan-turing-institute/sktime/blob/main/sktime/datasets/data/BasicMotions/BasicMotions_TEST.ts
.. _issue: https://github.com/alan-turing-institute/sktime/issues
.. _tsregression: http://tseregression.org/
.. _timeseriesclassification.com: http://www.timeseriesclassification.com/index.php
.. _tutorials: https://www.sktime.org/en/stable/tutorials.html
