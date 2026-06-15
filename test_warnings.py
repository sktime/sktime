import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sktime.datasets import load_airline
from sktime.forecasting.compose import (
    make_reduction,
)
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.model_selection import ForecastingGridSearchCV
from sktime.split import ExpandingGreedySplitter
from sktime.transformations.series.date import DateTimeFeatures
from sktime.transformations.series.summarize import WindowSummarizer

y = load_airline()
fh = np.arange(1, 36)

# =============================================================================
# 1) FutureWarning: ChainedAssignment + CategoricalDtype in DateTimeFeatures
#    sktime uses df["col"][idx] = value which pandas 3.0 CoW will break.
#    Also uses .replace() on categorical Series (deprecated).
#    Source: sktime/transformations/series/date.py:438,439,445,451,455,457,462
# =============================================================================
X_daily = pd.DataFrame({"exo": range(90)}, index=pd.date_range("2024-01-01", periods=90))

DateTimeFeatures().fit_transform(X_daily)

# =============================================================================
# 2) FutureWarning: ChainedAssignment in DateTimeFeatures with manual_selection
#    cd[funcs] = np.int64(cd[funcs]) assigns on a copy.
#    Source: sktime/transformations/series/date.py:406
# =============================================================================
DateTimeFeatures(manual_selection=["day_of_week", "month_of_year"]).fit_transform(X_daily)

# =============================================================================
# 3) FutureWarning: ChainedAssignment in WindowSummarizer
#    func_dict["window"] = ... assigns on a copy during fit_transform.
#    Source: sktime/transformations/series/summarize.py:308
# =============================================================================
X_exo = pd.DataFrame({"exo": np.random.poisson(lam=5, size=len(y))}, index=y.index)

window_summarizer = WindowSummarizer(
    lag_feature={"lag": [0, 1], "mean": [[1, 2], [2, 3]], "std": [[1, 2]]},
    target_cols=["exo"],
)
window_summarizer.fit_transform(X_exo, y)

# =============================================================================
# 4) FutureWarning: ChainedAssignment in make_reduction with pooling="global"
#    Inplace method on a DataFrame copy during predict().
#    Source: sktime/forecasting/compose/_reduce.py:453
# =============================================================================
forecaster = make_reduction(
    LinearRegression(),
    transformers=[WindowSummarizer(lag_feature={"lag": [1, 12]})],
    window_length=None,
    pooling="global",
)
forecaster.fit(y, fh=fh)
forecaster.predict()

# =============================================================================
# 5) FutureWarning: ChainedAssignment in ForecastingGridSearchCV
#    CV results stored via chained assignment.
#    Source: sktime/forecasting/model_selection/_base.py:223
# =============================================================================
grid_cv = ForecastingGridSearchCV(
    forecaster=ExponentialSmoothing(seasonal="multiplicative"),
    param_grid={"trend": ["add", "mul"], "sp": [6, 12]},
    cv=ExpandingGreedySplitter(test_size=12, folds=2),
)
grid_cv.fit(y, fh=fh)