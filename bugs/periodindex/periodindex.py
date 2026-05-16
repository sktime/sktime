from sktime.forecasting.darts import DartsLinearRegressionModel
import pandas as pd
import sklearn

X, y = sklearn.datasets.make_regression(n_samples=1000, n_features=2, noise=0.1, random_state=42)

X = pd.DataFrame(X, columns=['feature1', 'feature2'])

dt_idx = pd.date_range(start='2020-01-01', periods=1000, freq='15min')
date_range1 = pd.PeriodIndex(dt_idx[:10], freq='15min')
date_range2 = pd.PeriodIndex(dt_idx[20:], freq='15min')

non_continuous_date_range = date_range1.append(date_range2)
X = X.iloc[:990]
y = y[:990]
X.index = non_continuous_date_range
y = pd.Series(y, index=non_continuous_date_range)

dlrm = DartsLinearRegressionModel(
    lags=3,
    output_chunk_length=1,
    random_state=42
)

dlrm.fit(y, X)
