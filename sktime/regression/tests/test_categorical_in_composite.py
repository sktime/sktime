import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from sktime.regression.distance_based import KNeighborsTimeSeriesRegressor
from sktime.regression.model_selection import TSRGridSearchCV
from sktime.transformations.series.adapt import TabularToSeriesAdaptor
from sktime.transformations.series.subset import ColumnSelect


def test_pipeline_with_categorical():
    """Test that pipeline with categorical feature works."""
    data = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [1, 2, 1, 2, 1, 2, 1, 2],
            "color": ["red", "blue", "green", "blue", "red", "green", "red", "blue"],
            "weight": [1.2, 3.4, 2.3, 4.5, 3.2, 2.1, 1.8, 3.3],
        }
    )
    y = pd.DataFrame({"id": [1, 2, 3, 4], "label": [10.5, 20.3, 15.7, 25.1]})
    test_data = pd.DataFrame(
        {
            "id": [5, 5, 6, 6],
            "time": [1, 2, 1, 2],
            "color": ["blue", "red", "green", "blue"],
            "weight": [2.5, 3.1, 1.9, 2.8],
        }
    )

    test_labels = pd.DataFrame({"id": [5, 6], "label": [22.0, 18.5]})
    y.set_index(["id"], inplace=True)
    data.set_index(["id", "time"], inplace=True)
    test_data.set_index(["id", "time"], inplace=True)
    test_labels.set_index(["id"], inplace=True)

    encoder = TabularToSeriesAdaptor(OneHotEncoder(), pooling="global")

    pipeline = (
        ColumnSelect(columns=["color"], index_treatment="keep")
        * encoder
        * KNeighborsTimeSeriesRegressor(n_neighbors=1)
    )
    pipeline.fit(data, y)
    pipeline.predict(test_data)


def test_gridsearch_with_categorical():
    data = pd.DataFrame(
        {
            "id": [1, 1, 2, 2, 3, 3, 4, 4],
            "time": [1, 2, 1, 2, 1, 2, 1, 2],
            "color": ["red", "blue", "green", "blue", "red", "green", "red", "blue"],
            "weight": [1.2, 3.4, 2.3, 4.5, 3.2, 2.1, 1.8, 3.3],
        }
    )
    y = pd.DataFrame({"id": [1, 2, 3, 4], "label": [10.5, 20.3, 15.7, 25.1]})
    y.set_index(["id"], inplace=True)
    data.set_index(["id", "time"], inplace=True)

    encoder = TabularToSeriesAdaptor(OneHotEncoder(), pooling="global")

    pipeline = (
        ColumnSelect(columns=["color"], index_treatment="keep")
        * encoder
        * KNeighborsTimeSeriesRegressor()
    )

    grid = TSRGridSearchCV(
        pipeline, {"kneighborstimeseriesregressor__n_neighbors": [1, 3, 4]}, cv=2
    )
    grid.fit(data, y)
