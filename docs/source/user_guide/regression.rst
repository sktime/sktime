.. _user_guide_regression:

Time Series Regression
======================
.. note::

    Before we dive into Time Series Regression, we would explain how crossectional data are different from time-series data. Crossectional data are observed at a fixed point in time. This means that no time sequence occurred. The collection of numbers of students in every classroom of a school is an example of crossectional data. If this data is observed at time intervals, for instance, the collection of the number of students in every classroom of a school daily, this kind of data is called a time-series data.



1.1 Time Series Forest Regressor
================================

A time series forest is an ensemble of decision trees built on random intervals.

The mean, standard deviation and slope are new features extracted at each random interval. These are trained on a decision tree. The predicted regression target of an input sample is computed as the mean predicted regression targets of the trees in the forest.
This implementation deviates from the original in minor ways. It samples intervals with replacement and does not use the splitting criteria tiny refinement described in `[1] <https://arxiv.org/abs/1302.2277>`_. This is an intentionally stripped down, non configurable version for use as a hive-cote component. For a configurable tree based ensemble, see `sktime.classifiers.ensemble.TimeSeriesForestClassifier <https://www.sktime.org/en/latest/api_reference.html#sktime-classification-time-series-classification>`_.

.. code-block:: python

 import numpy as np
 from sklearn.metrics import r2_score
 from sklearn.model_selection import train_test_split
 from sklearn.pipeline import Pipeline
 from sklearn.tree import DecisionTreeRegressor
 from sktime.datasets import load_uschange
 from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
 from sktime.regression import TimeSeriesForestRegressor
 from sktime.utils.slope_and_trend import _slope

 #uschange : X - Dataframe =  load_uschange()[1]
 #uschange : y - consumption is the target variable (consumption = load_uschange()[0])

 X = load_uschange()[1].to_numpy()
 #transforming X to a univariate data

 X = np.reshape(X,(187,1,4))
 y = load_uschange()[0].to_numpy()
 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)
 pipes = [
  ("feature-extraction",
  RandomIntervalFeatureExtractor(n_intervals="sqrt", features=[np.mean, np.std, _slope])),
  ("regressor", DecisionTreeRegressor()),
 ]
 time_series_pipeline = Pipeline(pipes)

 #Time Series Forest Regressor
 tsfr = TimeSeriesForestRegressor(
     estimator=time_series_pipeline,
     n_estimators=100,
     bootstrap=True,
     random_state=1,
     n_jobs=-1,
 )
 tsfr.fit(X_train, y_train)
 y_pred = tsf.predict(X_test)

 #Model Evaluation
 r2_score(y_test,y_pred)

.. code-block:: output

output : 0.4193283712646667
