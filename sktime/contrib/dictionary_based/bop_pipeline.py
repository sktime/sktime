from sklearn.model_selection import GridSearchCV

from sktime.classifiers.time_series_neighbors import KNeighborsTimeSeriesClassifier
from sktime.datasets import load_italy_power_demand
from sktime.pipeline import Pipeline
from sktime.contrib.dictionary_based.SAX import SAX
from sktime.contrib.dictionary_based.dictionary_distances import euclidean_distance


def bop_pipeline(X, y):
    steps = [
        ('transform', SAX(remove_repeat_words=True)),
        ('clf', KNeighborsTimeSeriesClassifier(n_neighbors=1, metric=euclidean_distance))
    ]
    pipeline = Pipeline(steps)

    series_length = X.iloc[0, 0].shape[0]
    max_window_searches = series_length / 4
    win_inc = int((series_length - 10) / max_window_searches)
    if win_inc < 1:
        win_inc = 1
    window_sizes = [win_size for win_size in range(10, series_length + 1, win_inc)]

    cv_params = {
        'transform__word_length': [8, 10, 12, 14, 16],
        'transform__alphabet_size': [2, 3, 4],
        'transform__window_size': window_sizes
    }
    model = GridSearchCV(pipeline,
                         cv_params,
                         cv=5
                         )
    model.fit(X, y)
    return model


if __name__ == "__main__":
    X_train, y_train = load_italy_power_demand(split='TRAIN', return_X_y=True)
    X_test, y_test = load_italy_power_demand(split='TEST', return_X_y=True)

    model = bop_pipeline(X_train, y_train)
    model.predict(X_test)
