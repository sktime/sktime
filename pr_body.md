## Description

Adds a new forecaster that implements recursive moving average prediction, as requested in #3992.

This forecaster predicts future values as the recursive moving average of past observations, where forecasts beyond horizon 1 use previously computed forecasts recursively.

## Key Features

- Uses `_DelegatedForecaster` with `make_reduction(strategy='recursive')`
- Includes helper `_AverageXEstimator` that returns row-wise mean of features
- `window_length` parameter with validation (must be int >= 1)
- Proper docstrings, tags, and `get_test_params()`

## Difference from NaiveForecaster

`NaiveForecaster(strategy='mean')` predicts a **constant** (mean of last window) for all future horizons.

`RecursiveMovingAverageForecaster` applies the moving average **recursively**:
- h=1: mean of last `window_length` actual observations
- h=2: mean using forecast from h=1
- h=3: mean using forecasts from h=1 and h=2
- And so on...

## Example Usage

```python
from sktime.forecasting.moving_average import RecursiveMovingAverageForecaster
from sktime.datasets import load_airline

y = load_airline()
forecaster = RecursiveMovingAverageForecaster(window_length=3)
forecaster.fit(y)
y_pred = forecaster.predict(fh=[1, 2, 3, 4, 5])
```

## References

- Closes #3992
- Based on approach suggested by @fkiraly in https://github.com/sktime/sktime/issues/3992#issuecomment-1365128879
- Linked to my comment: https://github.com/sktime/sktime/issues/3992#issuecomment-2817892542
