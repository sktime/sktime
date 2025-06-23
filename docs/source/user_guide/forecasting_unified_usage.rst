
Forecasting Unified Usage
=========================

This guide demonstrates how to apply multiple forecasting models in a consistent way using `sktime`.

We will forecast airline passenger data using a unified interface and compare the predictions.

Data Preparation
----------------

We load the monthly international airline passenger dataset:

.. code-block:: python

    from sktime.datasets import load_airline
    y = load_airline()
    y.index.freq = "MS"

Train-Test Split
----------------

We split the data into training and test sets.

.. code-block:: python

    from sktime.forecasting.model_selection import temporal_train_test_split

    y_train, y_test = temporal_train_test_split(y)

    fh = range(1, len(y_test) + 1)

Define Models
-------------

We define three models using a common interface:

- **NaiveForecaster**: Uses the last observed value to make forecasts.
- **ExponentialSmoothing**: Applies exponential decay to past observations.
- **ARIMA**: Combines autoregression and moving average components.



.. code-block:: python

    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    from sktime.forecasting.arima import ARIMA

    models = {
        "Naive": NaiveForecaster(strategy="last"),
        "ETS": ExponentialSmoothing(),
        "ARIMA": ARIMA()
    }

Fit and Forecast
----------------

We fit each model and generate predictions.

.. code-block:: python

    fitted_models = {}
    y_preds = {}

    for name, model in models.items():
        fitted_models[name] = model.fit(y_train)
        y_preds[name] = model.predict(fh)

Visualize Forecasts
-------------------

We plot and compare forecasts from each model.

.. code-block:: python

    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label="True", linewidth=2)

    for name, y_pred in y_preds.items():
        plt.plot(y_test.index, y_pred, label=name)

    plt.title("Forecast Comparison: Airline Dataset")
    plt.xlabel("Date")
    plt.ylabel("Passengers")
    plt.legend()
    plt.tight_layout()
    plt.show()
