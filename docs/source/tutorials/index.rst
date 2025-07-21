.. _tutorials_index:

Tutorials
=========

Step-by-step tutorials to learn sktime from the ground up.

These tutorials provide hands-on experience with sktime's key features through 
practical examples. Each tutorial is a Jupyter notebook that you can run 
interactively or view online.

.. toctree::
   :maxdepth: 2
   :caption: Forecasting Tutorials
   :hidden:

   forecasting/forecasting_univariate_timeseries
   forecasting/forecasting_with_exogenous
   forecasting/multivariate_forecasting
   forecasting/hierarchical_forecasting
   forecasting/probabilistic_forecasting
   forecasting/transformations
   forecasting/pipelines
   forecasting/hyperparameter_tuning
   forecasting/cross_validation_and_metrics

.. toctree::
   :maxdepth: 2
   :caption: Classification Tutorials
   :hidden:

   classification/time_series_classification
   classification/feature_extraction
   classification/early_classification

.. toctree::
   :maxdepth: 2
   :caption: Clustering Tutorials
   :hidden:

   clustering/time_series_clustering
   clustering/time_series_distances

.. toctree::
   :maxdepth: 2
   :caption: Data Types & Handling
   :hidden:

   data_types/mtypes_and_scitypes
   data_types/time_series_data_containers
   data_types/data_conversion

.. toctree::
   :maxdepth: 2
   :caption: Detection & Segmentation
   :hidden:

   detection_segmentation/anomaly_detection
   detection_segmentation/change_point_detection
   detection_segmentation/time_series_segmentation

Forecasting Tutorials
----------------------

Learn time series forecasting with sktime, from basic univariate forecasting to advanced techniques.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card::
        :text-align: center

        Univariate Forecasting

        ^^^

        Start here: Learn the basics of time series forecasting with sktime.

        +++

        .. button-ref:: forecasting/forecasting_univariate_timeseries
            :color: primary
            :click-parent:
            :expand:

            Start Tutorial

    .. grid-item-card::
        :text-align: center

        Forecasting with Exogenous Variables

        ^^^

        Use external variables to improve your forecasts.

        +++

        .. button-ref:: forecasting/forecasting_with_exogenous
            :color: primary
            :click-parent:
            :expand:

            View Tutorial

    .. grid-item-card::
        :text-align: center

        Multivariate Forecasting

        ^^^

        Forecast multiple time series simultaneously.

        +++

        .. button-ref:: forecasting/multivariate_forecasting
            :color: primary
            :click-parent:
            :expand:

            View Tutorial

    .. grid-item-card::
        :text-align: center

        Hierarchical Forecasting

        ^^^

        Work with hierarchical and grouped time series.

        +++

        .. button-ref:: forecasting/hierarchical_forecasting
            :color: primary
            :click-parent:
            :expand:

            View Tutorial

    .. grid-item-card::
        :text-align: center

        Probabilistic Forecasting

        ^^^

        Generate prediction intervals and probability distributions.

        +++

        .. button-ref:: forecasting/probabilistic_forecasting
            :color: primary
            :click-parent:
            :expand:

            View Tutorial

    .. grid-item-card::
        :text-align: center

        Transformations

        ^^^

        Preprocess and transform your time series data.

        +++

        .. button-ref:: forecasting/transformations
            :color: primary
            :click-parent:
            :expand:

            View Tutorial

Classification Tutorials
-------------------------

Learn time series classification techniques and applications.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card::
        :text-align: center

        Time Series Classification

        ^^^

        Classify time series using various algorithms.

        +++

        .. button-ref:: classification/time_series_classification
            :color: primary
            :click-parent:
            :expand:

            Start Tutorial

    .. grid-item-card::
        :text-align: center

        Feature Extraction

        ^^^

        Extract features from time series for classification.

        +++

        .. button-ref:: classification/feature_extraction
            :color: primary
            :click-parent:
            :expand:

            View Tutorial

    .. grid-item-card::
        :text-align: center

        Early Classification

        ^^^

        Classify time series before observing the full sequence.

        +++

        .. button-ref:: classification/early_classification
            :color: primary
            :click-parent:
            :expand:

            View Tutorial

Additional Tutorials
--------------------

Explore more specialized topics and advanced techniques.

.. grid:: 1 2 2 2
    :gutter: 3

    .. grid-item-card::
        :text-align: center

        Time Series Clustering

        ^^^

        Group similar time series using clustering algorithms.

        +++

        .. button-ref:: clustering/time_series_clustering
            :color: secondary
            :click-parent:
            :expand:

            View Tutorial

    .. grid-item-card::
        :text-align: center

        Time Series Distances

        ^^^

        Measure similarity between time series.

        +++

        .. button-ref:: clustering/time_series_distances
            :color: secondary
            :click-parent:
            :expand:

            View Tutorial

    .. grid-item-card::
        :text-align: center

        Data Types & Containers

        ^^^

        Understanding sktime's data structures.

        +++

        .. button-ref:: data_types/mtypes_and_scitypes
            :color: secondary
            :click-parent:
            :expand:

            View Tutorial

    .. grid-item-card::
        :text-align: center

        Anomaly Detection

        ^^^

        Detect outliers and anomalies in time series.

        +++

        .. button-ref:: detection_segmentation/anomaly_detection
            :color: secondary
            :click-parent:
            :expand:

            View Tutorial

    .. grid-item-card::
        :text-align: center

        Change Point Detection

        ^^^

        Identify structural breaks in time series.

        +++

        .. button-ref:: detection_segmentation/change_point_detection
            :color: secondary
            :click-parent:
            :expand:

            View Tutorial

    .. grid-item-card::
        :text-align: center

        Time Series Segmentation

        ^^^

        Segment time series into meaningful parts.

        +++

        .. button-ref:: detection_segmentation/time_series_segmentation
            :color: secondary
            :click-parent:
            :expand:

            View Tutorial

Getting Started
---------------

If you're new to sktime, we recommend starting with:

1. **Univariate Forecasting Tutorial** - Learn the basic concepts and workflow
2. **Time Series Classification Tutorial** - Understand classification tasks
3. **Transformations Tutorial** - Learn data preprocessing techniques

Each tutorial includes:

- Clear explanations of concepts
- Runnable code examples
- Practical exercises
- Links to relevant API documentation

.. note::

    All tutorials are available as Jupyter notebooks that you can download 
    and run locally. Look for the download button at the top of each tutorial page.