import numpy as np
import pandas as pd
from sktime.forecasting.sarimax import SARIMAX
from sktime.split import ExpandingWindowSplitter

def main():

    """ Simulated dataset"""
    np.random.seed(42)
    date_range = pd.date_range(start="2024-01-01", end="2025-12-31")
    data = {
        "feature_1": np.random.randn(len(date_range)),
        "feature_2": np.random.randn(len(date_range)),
        "target": np.random.randn(len(date_range)),
    }
    df = pd.DataFrame(data, index=date_range)
    df.index.name = "date"

    """ Split data"""
    X = df[["feature_1", "feature_2"]]
    y = df[["target"]]
    X_train, X_test = X.loc["2024"], X.loc["2025"]
    y_train, y_test = y.loc["2024"], y.loc["2025"]

    """ Align indices """
    X_test, y_test = X_test.align(y_test, join="inner", axis=0)

    """ Fit SARIMAX with proper exog handling """
    forecaster = SARIMAX(order=(1, 1, 0), seasonal_order=None)
    model = forecaster.fit(y=y_train, X=X_train)

    """ Create forecasting horizon """
    fh = np.arange(1, len(y_test) + 1)
    
    """ Get predictions with exog support """
    
    y_pred = model.predict(fh=fh, X=X_test)
    print(f"Successful predictions:\n{y_pred}")

if __name__ == "__main__":
    main()
