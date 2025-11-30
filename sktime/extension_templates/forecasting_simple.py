from sktime.forecasting.naive import NaiveForecaster

# Simple forecasting template
forecaster = NaiveForecaster(strategy="mean")
forecaster.fit(y_train)
y_pred = forecaster.predict(fh=[1, 2, 3])
