.. _forecasting:

Forecasting
===========

.. note::

    The user guide is under development. We have created a basic
    structure and are looking for contributions to develop the user guide
    further. For more details, please go to issue `#361 <https://github
    .com/alan-turing-institute/sktime/issues/361>`_ on GitHub.

Introduction
------------
Forecasating is making forward temporal predictions based on past data. The simplest case is the univariate case.

Inputs.

* Univariate time series, :math:`y = y(0), y(1), y(2), ..., y(N)`
* Forecasting horizon, :math:`fh = N+1, N+2, ..., N+h`

Output.

* Predictions of :math:`y` at the times in :math:`fh`, :math:`\hat{y} = \hat{y}(N+1), \hat{y}(N+2), ..., \hat{y}(N+h)`

Examples.

* Forecasting the global population
* Forecasting the price of a stock
* Forecasting the daily maximum temperature in a given location

Example
-------
(Example about flights from tutorial)


Algorithms included in sktime
-----------------------------
* Naive and seasonal naive
* Trend
* Exponential Smoothing
* ARIMA
* Theta
* BATS and TBATS
* fbProphet

Reductions included in sktime
-----------------------------


Variations in generative setting
--------------------------------


Evaluation and model selection
------------------------------


Algorithms not included in sktime
---------------------------------


Further reading
_______________

