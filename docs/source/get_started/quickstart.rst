.. _get_started_quickstart:

==========
Quickstart
==========

The code snippets below are designed to introduce sktime's functionality so you can start using it quickly. 

For more detailed information see the :ref:`tutorials <tutorials_index>`, :ref:`examples <examples>` and :ref:`API reference <api_reference>`.

Prerequisites
-------------

Before running these examples, make sure you have sktime installed. See our :ref:`installation guide <installation>` for instructions.

Forecasting
-----------

Predict future values of a time series:

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
---------------------------

Classify time series into categories:

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
-----------------------

Predict continuous values from time series:

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
-----------------------

Group similar time series together:

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
-----------------------

Detect anomalies and change points in time series:

.. warning::

   The time series annotation API is experimental and may change in future releases.

.. code-block:: python

    >>> from sktime.detection.adapters import PyODAnnotator
    >>> from pyod.models.iforest import IForest
    >>> from sktime.datasets import load_airline
    >>> y = load_airline()
    >>> pyod_model = IForest()
    >>> pyod_sktime_annotator = PyODAnnotator(pyod_model)
    >>> pyod_sktime_annotator.fit(y)
    >>> annotated_series = pyod_sktime_annotator.predict(y)

Next Steps
----------

Now that you've seen sktime in action:

- **Learn more**: Check out our :ref:`tutorials <tutorials_index>` for detailed guidance
- **Solve problems**: Browse our :ref:`how-to guides <how_to_index>` for specific tasks
- **Explore examples**: See our :ref:`example notebooks <examples>` for more complex use cases
- **Understand concepts**: Read about :ref:`key concepts <get_started_key_concepts>` for deeper understanding
