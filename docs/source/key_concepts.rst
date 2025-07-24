.. _key_concepts:

============
Key Concepts
============

Understanding sktime's core concepts will help you use the framework effectively.

sktime seeks to provide a unified framework for multiple time series machine learning tasks.
This makes sktime's functionality intuitive for users and lets developers extend the framework more easily.

Data Types
----------

sktime is designed for time series machine learning. Time series data refers to data where the variables are ordered over time or
an index indicating the position of an observation in the sequence of values.

In sktime, time series data can refer to data that is univariate, multivariate or panel, with the difference relating to the number and interrelation
between time series :term:`variables <variable>`, as well as the number of :term:`instances <instance>` for which each variable is observed.

Univariate Time Series
~~~~~~~~~~~~~~~~~~~~~~

:term:`Univariate time series` data refers to data where a single :term:`variable` is tracked over time.

**Example**: Daily temperature readings for a city over a year.

Multivariate Time Series
~~~~~~~~~~~~~~~~~~~~~~~~

:term:`Multivariate time series` data refers to data where multiple :term:`variables <variable>` are tracked over time for the same :term:`instance`.

**Examples**:
- Multiple quarterly economic indicators for a country
- Multiple sensor readings from the same machine

Panel Time Series
~~~~~~~~~~~~~~~~~

:term:`Panel time series` data refers to data where the variables (univariate or multivariate) are tracked for multiple :term:`instances <instance>`.

**Examples**:
- Multiple quarterly economic indicators for several countries
- Multiple sensor readings for multiple machines

Learning Tasks
--------------

sktime's functionality for each learning task is centered around providing a set of code artifacts that match a common interface to a given
scientific purpose (i.e. :term:`scientific type` or :term:`scitype`).

For example, sktime includes a common interface for "forecaster" classes designed to predict future values of a time series.

Supported Learning Tasks
~~~~~~~~~~~~~~~~~~~~~~~~

sktime's interface currently supports:

**Time Series Classification**
    :term:`Time series classification` where the time series data for a given instance are used to predict a categorical target class.

    *Example*: Classify ECG signals as normal or abnormal.

**Time Series Regression**
    :term:`Time series regression` where the time series data for a given instance are used to predict a continuous target value.

    *Example*: Predict house prices based on historical price trends.

**Time Series Clustering**
    :term:`Time series clustering` where the goal is to discover groups consisting of instances with similar time series.

    *Example*: Group customers based on their purchasing patterns over time.

**Forecasting**
    :term:`Forecasting` where the goal is to predict future values of the input series.

    *Example*: Predict next month's sales based on historical data.

**Time Series Annotation**
    :term:`Time series annotation` which is focused on outlier detection, anomaly detection, change point detection and segmentation.

    *Example*: Detect unusual spikes in server response times.

Reduction
---------

While the list above presents each learning task separately, in many cases it is possible to adapt one learning task to help solve another related learning task.

**Forecasting to Regression**
    One approach to forecasting is to reduce the forecasting problem to cross-sectional regression, where the input data are tabularized and lags of the data are treated as independent features in scikit-learn style tabular regression algorithms.

**Forecasting to Anomaly Detection**
    One approach to anomaly detection is to use a forecaster to predict future values and flag observations that are too far from these predictions as anomalies.

sktime typically incorporates these type of :term:`reductions <reduction>` through the use of composable classes that let users adapt one learning task to solve another related one.

API Design Principles
---------------------

**Unified Interface**
    All estimators in sktime follow consistent patterns for fitting, predicting, and transforming data.

**Scikit-learn Compatible**
    sktime follows scikit-learn conventions where possible, making it familiar to existing users.

**Composable**
    Components can be combined into pipelines and complex workflows.

**Extensible**
    New algorithms can be easily added by following sktime's base class patterns.

For More Information
--------------------

- **Detailed terminology**: See our :ref:`glossary <glossary>`
- **Practical examples**: Check out our :ref:`notebook examples <examples>`
- **In-depth learning**: Explore our :ref:`tutorials <tutorials_index>`
- **API details**: Browse our :ref:`API reference <api_reference>`
