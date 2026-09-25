"""MultiRocketMultivariate test code."""

import numpy as np
import pytest
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sktime.datasets import load_basic_motions
from sktime.tests.test_switch import run_test_for_class
from sktime.transformations.rocket import MultiRocketMultivariate


@pytest.mark.skipif(
    not run_test_for_class(MultiRocketMultivariate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multirocket_multivariate_on_basic_motions():
    """Test of MultiRocketMultivariate on basic motions."""
    # load training data
    X_training, Y_training = load_basic_motions(split="train", return_X_y=True)

    # 'fit' MultiRocket -> infer data dimensions, generate random kernels
    multirocket = MultiRocketMultivariate(random_state=0)
    multirocket.fit(X_training)

    # transform training data
    X_training_transform = multirocket.transform(X_training)

    # test shape of transformed training data -> (number of training
    # examples, nearest multiple of 4*84=336 < 50,000 (2*4*6_250))
    np.testing.assert_equal(X_training_transform.shape, (len(X_training), 49_728))

    # fit classifier
    classifier = make_pipeline(
        StandardScaler(with_mean=False),
        RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
    )
    classifier.fit(X_training_transform, Y_training)

    # load test data
    X_test, Y_test = load_basic_motions(split="test", return_X_y=True)

    # transform test data
    X_test_transform = multirocket.transform(X_test)

    # test shape of transformed test data -> (number of test examples,
    # nearest multiple of 4*84=336 < 50,000 (2*4*6_250))
    np.testing.assert_equal(X_test_transform.shape, (len(X_test), 49_728))

    # predict (alternatively: 'classifier.score(X_test_transform, Y_test)')
    predictions = classifier.predict(X_test_transform)
    accuracy = accuracy_score(predictions, Y_test)

    # test predictions (on BasicMotions, should be 100% accurate)
    assert accuracy == 1.0


@pytest.mark.skipif(
    not run_test_for_class(MultiRocketMultivariate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multirocket_multivariate_difference_features_match_base():
    """Features of the differenced half must match those of the base half.

    The differenced half convolves diff(X), so it must produce what the base
    half produces when diff(X) is the input. Regression test for #11291, where
    the differenced half sized its convolution windows from the undifferenced
    length and reused the first combination's channel selection for every
    kernel. n_timepoints=65 (ERing) is a length at which the two halves fit
    different numbers of dilations, which is what exposes the latter.
    """
    X = np.random.RandomState(0).normal(size=(4, 3, 65))
    X_diff = np.diff(X, 1)

    trf = MultiRocketMultivariate(random_state=0)
    full = trf.fit(X).transform(X).to_numpy()
    base = trf.fit(X_diff).transform(X_diff).to_numpy()
    half = full.shape[1] // 2

    np.testing.assert_allclose(full[:, half:], base[:, :half], rtol=1e-4, atol=1e-5)


@pytest.mark.skipif(
    not run_test_for_class(MultiRocketMultivariate),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
def test_multirocket_multivariate_original_implementation():
    """original_implementation=True must restore the reference behaviour.

    In the reference implementation every kernel of the differenced half
    reuses the first combination's channel selection, so channels outside it
    do not affect that half of the features at all.
    """
    rng = np.random.RandomState(0)
    X = rng.normal(size=(4, 4, 65))

    trf = MultiRocketMultivariate(random_state=0, original_implementation=True).fit(X)
    used = set(trf.parameter[1][: trf.parameter[0][0]].tolist())
    unused = next(c for c in range(X.shape[1]) if c not in used)

    X2 = X.copy()
    X2[:, unused] = rng.normal(size=(X.shape[0], X.shape[2]))
    features, features2 = trf.transform(X).to_numpy(), trf.transform(X2).to_numpy()
    half = features.shape[1] // 2

    np.testing.assert_array_equal(features[:, half:], features2[:, half:])
    assert not np.array_equal(features[:, :half], features2[:, :half])
