.. _get_started:

===========
Get Started
===========

The following information is designed to get users up and running with ``sktime`` quickly. For more detailed information, see the links in each of the subsections.

Installation
------------

``sktime`` currently supports:

* environments with python version 3.8, 3.9, 3.10, 3.11, or 3.12.
* operating systems Mac OS X, Unix-like OS, Windows 8.1 and higher
* installation via ``PyPi`` or ``conda``

Please see the :ref:`installation <installation>` guide for step-by-step instructions on the package installation.

Key Concepts
------------

``sktime`` seeks to provide a unified framework for multiple time series machine learning tasks. This (hopefully) makes ``sktime's`` functionality intuitive for users
and lets developers extend the framework more easily. But time series data and the related scientific use cases each can take multiple forms.
Therefore, a key set of common concepts and terminology is important.

Data Types
~~~~~~~~~~

``sktime`` is designed for time series machine learning. Time series data refers to data where the variables are ordered over time or
an index indicating the position of an observation in the sequence of values.

In ``sktime`` time series data can refer to data that is univariate, multivariate or panel, with the difference relating to the number and interrelation
between time series :term:`variables <variable>`, as well as the number of :term:`instances <instance>` for which each variable is observed.

- :term:`Univariate time series` data refers to data where a single :term:`variable` is tracked over time.
- :term:`Multivariate time series` data refers to data where multiple :term:`variables <variable>` are tracked over time for the same :term:`instance`. For example, multiple quarterly economic indicators for a country or multiple sensor readings from the same machine.
- :term:`Panel time series` data refers to data where the variables (univariate or multivariate) are tracked for multiple :term:`instances <instance>`. For example, multiple quarterly economic indicators for several countries or multiple sensor readings for multiple machines.

Learning Tasks
~~~~~~~~~~~~~~

``sktime's`` functionality for each learning tasks is centered around providing a set of code artifacts that match a common interface to a given
scientific purpose (i.e. :term:`scientific type` or :term:`scitype`). For example, ``sktime`` includes a common interface for "forecaster" classes designed to predict future values
of a time series.

``sktime's`` interface currently supports:

- :term:`Time series classification` where the time series data for a given instance are used to predict a categorical target class.
- :term:`Time series regression` where the time series data for a given instance are used to predict a continuous target value.
- :term:`Time series clustering` where the goal is to discover groups consisting of instances with similar time series.
- :term:`Forecasting` where the goal is to predict future values of the input series.
- :term:`Time series annotation` which is focused on outlier detection, anomaly detection, change point detection and segmentation.

Reduction
~~~~~~~~~

While the list above presents each learning task separately, in many cases it is possible to adapt one learning task to help solve another related learning task. For example,
one approach to forecasting would be to use a regression model that explicitly accounts for the data's time dimension. However, another approach is to reduce the forecasting problem
to cross-sectional regression, where the input data are tabularized and lags of the data are treated as independent features in `scikit-learn` style
tabular regression algorithms. Likewise one approach to the time series annotation task like anomaly detection is to reduce the problem to using forecaster to predict future values and flag
observations that are too far from these predictions as anomalies. ``sktime`` typically incorporates these type of :term:`reductions <reduction>` through the use of composable classes that
let users adapt one learning task to solve another related one.

For more information on ``sktime's`` terminology and functionality see the :ref:`glossary` and the :ref:`notebook examples <examples>`.

Quickstart
----------
The code snippets below are designed to introduce ``sktime's`` functionality so you can start using its functionality quickly. For more detailed information see the :ref:`tutorials`,  :ref:`examples` and :ref:`api_reference` in ``sktime's`` :ref:`user_documentation`.

Forecasting
~~~~~~~~~~~

.. code-block:: python

    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.theta import ThetaForecaster
    >>> from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
    >>> from sktime.split import temporal_train_test_split

    >>> y = load_airline()
    >>> y_train, y_test = temporal_train_test_split(y)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False)
    >>> forecaster = ThetaForecaster(sp=12)  # monthly seasonal periodicity
    >>> forecaster.fit(y_train)
    >>> y_pred = forecaster.predict(fh)
    >>> mean_absolute_percentage_error(y_test, y_pred)
    0.08661467738190656

Time Series Classification
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    >>> from sktime.classification.interval_based import TimeSeriesForestClassifier
    >>> from sktime.datasets import load_arrow_head
    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.metrics import accuracy_score

    >>> X, y = load_arrow_head()
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
    >>> classifier = TimeSeriesForestClassifier()
    >>> classifier.fit(X_train, y_train)
    >>> y_pred = classifier.predict(X_test)
    >>> accuracy_score(y_test, y_pred)
    0.8679245283018868

Time Series Regression
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    >>> from sktime.datasets import load_covid_3month
    >>> from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
    >>> from sklearn.metrics import mean_squared_error

    >>> X_train, y_train = load_covid_3month(split="train")
    >>> y_train = y_train.astype("float")
    >>> X_test, y_test = load_covid_3month(split="test")
    >>> y_test = y_test.astype("float")
    >>> regressor = KNeighborsTimeSeriesRegressor()
    >>> regressor.fit(X_train, y_train)
    >>> y_pred = regressor.predict(X_test)
    >>> mean_squared_error(y_test, y_pred)

Time Series Clustering
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    >>> from sklearn.model_selection import train_test_split
    >>> from sktime.clustering.k_means import TimeSeriesKMeans
    >>> from sktime.clustering.utils.plotting._plot_partitions import plot_cluster_algorithm
    >>> from sktime.datasets import load_arrow_head

    >>> X, y = load_arrow_head()
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y)

    >>> k_means = TimeSeriesKMeans(n_clusters=5, init_algorithm="forgy", metric="dtw")
    >>> k_means.fit(X_train)
    >>> plot_cluster_algorithm(k_means, X_test, k_means.n_clusters)

Time Series Annotation
~~~~~~~~~~~~~~~~~~~~~~

.. warning::

   The time series annotation API is experimental,
   and may change in future releases.

.. code-block:: python

    >>> from sktime.annotation.adapters import PyODAnnotator
    >>> from pyod.models.iforest import IForest
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> pyod_model = IForest()
    >>> pyod_sktime_annotator = PyODAnnotator(pyod_model)
    >>> pyod_sktime_annotator.fit(y)
    >>> annotated_series = pyod_sktime_annotator.predict(y)
