from sktime.datasets import load_gunpoint
from sktime.transformers.series_as_features.matrix_profile import MatrixProfile


# TODO remove this test as this is covered in
#  test_all_series_as_features_transformers.py
def test_matrix_profile_transformer():
    X, y = load_gunpoint(return_X_y=True)
    Xt = MatrixProfile(10).fit_transform(X)

    assert Xt.shape[0] == X.shape[0]
