.. _getting_started:

Getting Started
===============

The following information is designed to get users up and running with ``sktime`` quickly. For more detailed information, see the links in each of the subsections.

Installation
------------

``sktime`` currently supports:

* environments with python version 3.6, 3.7, or 3.8.
* operating systems Mac OS X, Unix-like OS, Windows 8.1 and higher


``sktime`` releases are available via ``PyPI`` and ``conda`` .

To install ``sktime`` with its core dependencies via ``pip`` use:

.. code-block:: bash

    pip install sktime

To install ``sktime`` via ``pip`` with maximum dependencies, including soft dependencies, install using the ``all_extras`` modifier:

.. code-block:: bash

    pip install sktime[all_extras]


To install ``sktime`` via ``conda`` from ``conda-forge`` use:

.. code-block:: bash

    conda install -c conda-forge sktime

This will install ``sktime`` with core dependencies, excluding soft dependencies.

There is not currently a easy route to install ``sktime`` with maximum dependencies via ``conda``. Community contributions towards this, e.g., via conda metapackages, would be appreciated.

For more detailed installation instructions see our more detailed `installation`_ instructions.

Key Terminology
---------------

``sktime`` seeks to provide a unified framework for multiple time series machine learning tasks. While this (hopefully) makes ``sktime's`` functionality more intuitive for users and lets developers extend the framework more easily, having a key set of common terminology is also important.

NEED TERMINOLOGY HERE (e.g. time series, univariate, multivariate, scitype, reduction, and any other jargon we take for granted).

For more information on the terminology used by ``sktime`` see INSERT LINK TO OUR NEW GLOSSARY HERE.

Quickstart
----------
The code snippets below are designed to introduce ``sktime's`` functionality so you can start using its functionality quickly. For more detailed information see the `tutorials`_, `user_guide`_ and `api_reference`_ in ``sktime's`` `user_documentation`_.

Forecasting
~~~~~~~~~~~

.. code-block:: python

    from sktime.datasets import load_airline
    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.forecasting.theta import ThetaForecaster
    from sktime.performance_metrics.forecasting import mean_absolute_percentage_error

    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)
    fh = ForecastingHorizon(y_test.index, is_relative=False)
    forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    mean_absolute_percentage_error(y_test, y_pred)
    >>> 0.08661467738190656

Time Series Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python
    from sktime.classification.interval_based import TimeSeriesForestClassifier
    from sktime.datasets import load_arrow_head
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X, y = load_arrow_head(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    classifier = TimeSeriesForestClassifier()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy_score(y_test, y_pred)
    >>> 0.8679245283018868

Time Series Clustering
~~~~~~~~~~~~~~~~~~~~~~

OBLIGATORY WARNING ABOUT BEING EXPERIMENTAL HERE.

INSERT EXAMPLE HERE

Transformations
~~~~~~~~~~~~~~~

INSERT EXAMPLES (WE SHOULD SHOW AT LEAST ONE COMMON TRANSFORMER OF EACH TRANSFORMER TYPE) HERE

Annotation
~~~~~~~~~~
OBLIGATORY WARNING ABOUT BEING EXPERIMENTAL HERE.

INSERT EXAMPLE HERE
