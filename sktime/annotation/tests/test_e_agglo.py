# -*- coding: utf-8 -*-
"""Tests for EAGGLO."""

__author__ = ["KatieBuc"]

import numpy as np
import pandas as pd

from sktime.annotation.e_agglo import EAGGLO


def test_process_data():
    """Test data."""
    X = pd.DataFrame([-7.207066, -5.722571, 5.889715, 5.488990])
    member = np.array([0, 0, 1, 2])

    model = EAGGLO(member=member)
    model._process_data(X)
    model.N_
    model.sizes_
    model.D_
    model.left_
    model.right_
    model.open_
    model.merged_
    model.D_
    model.fit_
    model.progression_
    model.lm_
