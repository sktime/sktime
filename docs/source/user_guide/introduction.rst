.. _user_guide_introduction:

Introduction
============

.. note::

    The user guide is under development. We have created a basic
    structure and are looking for contributions to develop the user guide
    further. For more details, please go to issue `#361 <https://github
    .com/sktime/sktime/issues/361>`_ on GitHub.

sktime is a unified toolbox for machine learning with time series. Time
series give rise to multiple learning tasks (e.g.
forecasting and time series classification). The goal of sktime is to
provide all the necessary tools to solve these tasks, including dedicated time
series algorithms as well as tools for building, tuning and evaluating
composite models.

Many of these tasks are related. An algorithm that can
solve one of them can often be re-used to help solve another one, an idea
called reduction. sktime's unified interface allows to easily adapt an
algorithm for one task to another.

For example, to use a regression algorithm to solve a forecasting task, we
can simply write:

.. code-block:: python

    import numpy as np
    from sktime.datasets import load_airline
    from sktime.forecasting.compose import make_reduction
    from sklearn.ensemble import RandomForestRegressor
    from sktime.forecasting.model_selection import temporal_train_test_split
    from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

    y = load_airline()
    y_train, y_test = temporal_train_test_split(y)
    fh = np.arange(1, len(y_test) + 1)  # forecasting horizon
    regressor = RandomForestRegressor()
    forecaster = make_reduction(
    	regressor,
    	strategy="recursive",
    	window_length=12,
    )
    forecaster.fit(y_train)
    y_pred = forecaster.predict(fh)
    smape = MeanAbsolutePercentageError()
    smape(y_test, y_pred)
    >>> 0.1261192310833735

For more details, check out our `paper
<http://learningsys.org/neurips19/assets/papers/sktime_ml_systems_neurips2019.pdf>`__.

Currently, sktime provides:

* State-of-the-art algorithms for time series classification and regression, ported from the Java-based `tsml <https://github.com/uea-machine-learning/tsml/>`__ toolkit, as well as forecasting,
* Transformers, including single-series transformations (e.g. detrending or deseasonalization) and series-as-features transformations (e.g. feature extractors), as well as tools to compose different transformers,
* Pipelining,
* Tuning,
* Ensembling, such as a fully customisable random forest for time-series classification and regression, as well as ensembling for multivariate problems,

For a list of implemented methods, see our `estimator overview <https://github.com/sktime/sktime/blob/main/ESTIMATOR_OVERVIEW.md>`_.

In addition, sktime includes an experimental high-level API that unifies multiple learning tasks, partially inspired by the APIs of `mlr <https://mlr.mlr-org.com>`__ and `openML <https://www.openml.org>`__.
