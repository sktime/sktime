def test_reconciler_forecaster():
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import KFold

    from sktime.forecasting.base import ForecastingHorizon
    from sktime.forecasting.naive import NaiveForecaster
    from sktime.forecasting.reconciler_forecaster import ReconcilerForecaster

    """
    Test ReconcilerForecaster functionality.

    Test ReconcilerForecaster functionality.
    """

    # Generate random data
    y = np.random.rand(100)

    # Set up base forecaster and regressor
    base_forecaster = NaiveForecaster(strategy="mean")
    regressor = LinearRegression()
    cv = KFold(n_splits=5)

    # Create an instance of the reconciler forecaster
    reconciler = ReconcilerForecaster(base_forecaster, regressor, cv)

    # Define a forecasting horizon
    fh = ForecastingHorizon([1, 2, 3], is_relative=True)

    # Fit the reconciler forecaster
    reconciler.fit(y, fh=fh)

    # Predict with the reconciler forecaster
    predictions = reconciler.predict(fh)
    assert len(predictions) == len(fh)
