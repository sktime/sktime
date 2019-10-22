import numpy as np
import pytest
from sktime.datasets import load_gunpoint
from sktime.transformers.matrix_profile import MatrixProfile

def test_matrix_profile_transformer():

    X, y = load_gunpoint(return_X_y=True)

    Xt = MatrixProfile(10).transform(X)

    assert Xt.shape[0] == X.shape[0]  # checks if the output has the expected dimensions
