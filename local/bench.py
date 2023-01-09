from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline


from sktime.forecasting.compose import ForecastingPipeline, make_reduction
from sktime.utils._testing.hierarchical import _make_hierarchical
from sktime.forecasting.compose._reduce import _RecursiveReducer
from sklearn.linear_model import LinearRegression
from sktime.transformations.series.summarize import WindowSummarizer
from sktime.transformations.series.date import DateTimeFeatures

regressor = make_pipeline(
    LinearRegression(),
)

kwargs = {
    "lag_feature": {
        "lag": [1],
    }
}


forecaster_global = make_reduction(
    regressor,
    scitype="tabular-regressor",
    transformers=[WindowSummarizer(**kwargs, n_jobs=1, truncate="bfill")],
    window_length=None,
    strategy="recursive",
    pooling="global"
)

y = _make_hierarchical(hierarchy_levels=(100,), min_timepoints=1000, max_timepoints=1000)
from sktime.datatypes._utilities import get_time_index

def Main():


    # from time import perf_counter
    # t1_start = perf_counter()

    _ = forecaster_global.fit(y)

    #y_pred_global = forecaster_global.predict(fh=[1,2])

    # t1_stop = perf_counter()
    # print("Elapsed time during the whole program in seconds:", t1_stop-t1_start)

import cProfile
import pstats
cProfile.run("Main()","output.dat")

with open("time.txt", "w") as f:
    p =pstats.Stats("output.dat", stream=f)
    p.sort_stats("time").print_stats()