import numpy as np
import pytest

from sktime.classification.distance_based._shape_dtw import ShapeDTW
from sktime.utils._testing import generate_df_from_array

from sktime.transformers.series_as_features.paa_multivariate \
    import PAA_Multivariate
from sktime.transformers.series_as_features.dwt import DWT
from sktime.transformers.series_as_features.slope import Slope
from sktime.transformers.series_as_features.derivative import Derivative
from sktime.transformers.series_as_features.hog1d import HOG1D

from sktime.datasets import load_gunpoint


# Check that exception is raised for bad subsequence length.
@pytest.mark.parametrize("bad_subsequence_length", ['str', 0.5, -1, -0.5, {}])
def test_subsequence_length(bad_subsequence_length):
    X = generate_df_from_array(np.ones(10), n_rows=10, n_cols=1)
    y = np.zeros(10)

    if isinstance(bad_subsequence_length, type(None)):
        ShapeDTW(subsequence_length=bad_subsequence_length).fit(X, y)
    if not isinstance(bad_subsequence_length, int):
        with pytest.raises(TypeError):
            ShapeDTW(subsequence_length=bad_subsequence_length).fit(X, y)
    else:
        with pytest.raises(ValueError):
            ShapeDTW(subsequence_length=bad_subsequence_length).fit(X, y)


# Check shape_descriptor_function parameter
@pytest.mark.parametrize("bad_sdf", [3, "raw4", {}])
def test_shape_descriptor_function(bad_sdf):
    X = generate_df_from_array(np.ones(10), n_rows=10, n_cols=1)
    y = np.zeros(10)

    if not isinstance(bad_sdf, str):
        with pytest.raises(TypeError):
            ShapeDTW(shape_descriptor_function=bad_sdf).fit(X, y)
    else:
        with pytest.raises(ValueError):
            ShapeDTW(shape_descriptor_function=bad_sdf).fit(X, y)


# Check shape_descriptor_functions parameter
@pytest.mark.parametrize("bad_sdfs", [[], ["raw"], ["raw", "derivative"]])
def test_shape_descriptor_functions(bad_sdfs):
    X = generate_df_from_array(np.ones(10), n_rows=10, n_cols=1)
    y = np.zeros(10)

    if not len(bad_sdfs) == 2:
        with pytest.raises(ValueError):
            ShapeDTW(shape_descriptor_function="compound",
                     shape_descriptor_functions=bad_sdfs).fit(X, y)
    else:
        ShapeDTW(shape_descriptor_function="compound",
                 shape_descriptor_functions=bad_sdfs).fit(X, y)


# check that the metric_params are being fed in correctly
@pytest.mark.parametrize("metric_params", [None])
def test_metric_params(metric_params):

    X = generate_df_from_array(np.ones(10), n_rows=10, n_cols=1)
    y = np.zeros(10)

    # test the raw shape descriptor
    shp = ShapeDTW()
    assert shp.get_transformer("rAw") is None

    # test the paa shape descriptor
    shp = ShapeDTW(metric_params={"num_intERvals_paa": 3})
    assert shp.get_transformer("pAA").num_intervals == 3
    shp = ShapeDTW()
    assert shp.get_transformer("pAA").num_intervals == 8
    assert isinstance(shp.get_transformer("paa"), PAA_Multivariate)

    # test the dwt shape descriptor
    assert shp.get_transformer("dWt").num_levels == 3
    shp = ShapeDTW(metric_params={"num_LEvEls_dwt": 5})
    assert shp.get_transformer("Dwt").num_levels == 5
    assert isinstance(shp.get_transformer("dwt"), DWT)

    # test the slope shape descriptor
    shp = ShapeDTW()
    assert shp.get_transformer("sLoPe").num_intervals == 8
    shp = ShapeDTW(metric_params={"num_inTErvals_slope": 2})
    assert shp.get_transformer("slope").num_intervals == 2
    assert isinstance(shp.get_transformer("slope"), Slope)

    # test the derivative shape descriptor
    shp = ShapeDTW()
    assert isinstance(shp.get_transformer("derivative"), Derivative)

    # test the hog1d shape descriptor
    assert shp.get_transformer("hOG1d").num_intervals == 2 and \
           shp.get_transformer("hOG1d").num_bins == 8 and \
           shp.get_transformer("hog1d").scaling_factor == 0.1

    # test hog1d with only 1 custom parameter
    shp = ShapeDTW(metric_params={"NUM_intervals_hog1d": 5})
    assert shp.get_transformer("hoG1d").num_intervals == 5 and \
           shp.get_transformer("hOG1d").num_bins == 8 and \
           shp.get_transformer("hog1d").scaling_factor == 0.1

    shp = ShapeDTW(metric_params={"nUM_BinS_hog1d": 63})
    assert shp.get_transformer("hoG1d").num_intervals == 2 and \
           shp.get_transformer("hOG1d").num_bins == 63 and \
           shp.get_transformer("hog1d").scaling_factor == 0.1

    shp = ShapeDTW(metric_params={"scaling_factor_hog1d": 0.5})
    assert shp.get_transformer("hoG1d").num_intervals == 2 and \
           shp.get_transformer("hOG1d").num_bins == 8 and \
           shp.get_transformer("hog1d").scaling_factor == 0.5

    # test hog1d with 2 custom parameters
    shp = ShapeDTW(metric_params={"NUM_intervals_hog1d": 5,
                                  "nUM_BinS_hog1d": 63})
    assert shp.get_transformer("hoG1d").num_intervals == 5 and \
           shp.get_transformer("hOG1d").num_bins == 63 and \
           shp.get_transformer("hog1d").scaling_factor == 0.1

    shp = ShapeDTW(metric_params={"NUM_bins_hog1d": 63,
                                  "scaling_factor_hog1d": 0.5})
    assert shp.get_transformer("hoG1d").num_intervals == 2 and \
           shp.get_transformer("hOG1d").num_bins == 63 and \
           shp.get_transformer("hog1d").scaling_factor == 0.5

    shp = ShapeDTW(metric_params={"scaling_factor_hog1d": 0.5,
                                  "nUM_intervals_hog1d": 5})
    assert shp.get_transformer("hoG1d").num_intervals == 5 and \
           shp.get_transformer("hOG1d").num_bins == 8 and \
           shp.get_transformer("hog1d").scaling_factor == 0.5

    # test hog1d with all 3 custom parameters
    shp = ShapeDTW(metric_params={"scaling_factor_hog1d": 0.5,
                                  "nUM_intervals_hog1d": 5,
                                  "num_bins_hog1d": 63})
    assert shp.get_transformer("hoG1d").num_intervals == 5 and \
           shp.get_transformer("hOG1d").num_bins == 63 and \
           shp.get_transformer("hog1d").scaling_factor == 0.5

    shp = ShapeDTW()
    assert isinstance(shp.get_transformer("hog1d"), HOG1D)

    # test compound shape descriptor (mix upper and lower cases)
    shp = ShapeDTW(shape_descriptor_function="compound",
                   shape_descriptor_functions=["raw", "derivative"],
                   metric_params={"weighting_FACtor": 20})
    shp.fit(X, y)
    assert shp.fit(X, y).weighting_factor == 20

    with pytest.raises(ValueError):
        ShapeDTW(shape_descriptor_function="paa",
                 metric_params={"num_intervals": 8}).fit(X, y)


# check that shapeDTW works for performing classification using
# default settings
@pytest.mark.parametrize("shape_descriptor_function", ['raw', 'paa', 'dwt',
                                                       'slope', 'derivative',
                                                       'hog1d', 'compound'])
def test_classification_functionality(shape_descriptor_function):

    X_train, y_train = load_gunpoint(split='train', return_X_y=True)
    X_test, y_test = load_gunpoint(split='test', return_X_y=True)

    shp = ShapeDTW(shape_descriptor_function=shape_descriptor_function)
    shp.fit(X_train, y_train)
    print(shp.score(X_test, y_test))
