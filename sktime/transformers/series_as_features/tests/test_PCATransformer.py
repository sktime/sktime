import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from sktime.exceptions import NotFittedError
from sktime.transformers.series_as_features.pca import PCATransformer
from sktime.utils.data_container import detabularize
from sktime.utils.data_container import get_time_index
from sktime.utils.data_container import tabularize
from sktime.utils._testing import generate_df_from_array


# Check that exception is raised for bad input args.
@pytest.mark.parametrize("bad_components", ['str', 1.2, -1.2, -1, 11])
def test_bad_input_args(bad_components):
    X = generate_df_from_array(np.ones(10), n_rows=10, n_cols=1)

    if isinstance(bad_components, str):
        with pytest.raises(TypeError):
            PCATransformer(n_components=bad_components).fit(X)
    else:
        with pytest.raises(ValueError):
            PCATransformer(n_components=bad_components).fit(X)


# Test that keywords can be passed to PCA
@pytest.mark.parametrize("kwargs", [{'copy': False},
                                    {'whiten': True},
                                    {'svd_solver': 'arpack'},
                                    {'tol': 10e-6},
                                    {'iterated_power': 10},
                                    {'random_state': 42}])
def test_pca_kwargs(kwargs):
    np.random.seed(42)
    X = detabularize(pd.DataFrame(data=np.random.randn(10, 5)))
    pca = PCATransformer(n_components=1, **kwargs)
    pca.fit_transform(X)


# Test that PCATransformer fails if attempt to transform before fit
def test_early_trans_fail():
    X = generate_df_from_array(np.ones(10), n_rows=1, n_cols=1)
    pca = PCATransformer(n_components=1)

    with pytest.raises(NotFittedError):
        pca.transform(X)


# Test output format and dimensions.
@pytest.mark.parametrize(
    "n_instances,len_series,n_components", [
        (5, 2, 1),
        (5, 10, 1),
        (5, 10, 3),
        (5, 10, 5),
    ])
def test_output_format_dim(len_series, n_instances, n_components):
    np.random.seed(42)
    X = detabularize(
        pd.DataFrame(data=np.random.randn(n_instances, len_series)))

    trans = PCATransformer(n_components=n_components)
    Xt = trans.fit_transform(X)

    # Check number of rows and output type.
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == X.shape[0]

    # Check number of principal components in the output.
    assert tabularize(Xt).shape[1] == min(n_components, tabularize(X).shape[1])


# Check that the returned values agree with those produced by
# ``sklearn.decomposition.PCA``
@pytest.mark.parametrize("n_components", [1, 5, 0.9, 'mle'])
def test_pca_results(n_components):
    np.random.seed(42)

    # sklearn
    X = pd.DataFrame(data=np.random.randn(10, 5))
    pca = PCA(n_components=n_components)
    Xt1 = pca.fit_transform(X)

    # sktime
    Xs = detabularize(X)
    pca_transform = PCATransformer(n_components=n_components)
    Xt2 = pca_transform.fit_transform(Xs)

    assert np.allclose(np.asarray(Xt1), np.asarray(tabularize(Xt2)))


# Check output indices (row indices and columns the same, time indices start
# from 0)
@pytest.mark.parametrize("n_components", [1, 5, None])
def test_indices(n_components):
    np.random.seed(42)
    X = detabularize(pd.DataFrame(data=np.random.randn(10, 5)))
    X.columns = pd.CategoricalIndex(['col_0'])
    X.index = pd.Int64Index([i + 10 for i in range(10)])

    pca = PCATransformer(n_components=n_components)
    Xt = pca.fit_transform(X)

    assert X.columns.equals(Xt.columns)
    assert X.index.equals(Xt.index)
    assert get_time_index(Xt).equals(
        pd.Int64Index(range(pca.pca.n_components_)))
