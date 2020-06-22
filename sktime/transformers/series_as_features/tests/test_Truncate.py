



from sklearn.pipeline import Pipeline
from sktime.transformers.series_as_features.truncation import TruncationTransformer
from sklearn.ensemble import RandomForestClassifier
from sktime.datasets.base import load_gunpoint

def test_truncation_transformer():
    # load data
    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)

    truncated_transformer = TruncationTransformer(0, 1)
    Xt = truncated_transformer.transform(X_train)

    # steps = [('truncate', truncated_transformer),
    #         ('rfestimator', RandomForestClassifier(n_estimators=2))]
    # model = Pipeline(steps=steps)
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # print(y_pred)
