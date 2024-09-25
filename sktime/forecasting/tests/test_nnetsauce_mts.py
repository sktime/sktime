import pytest
import numpy as np
import pandas as pd
from sktime.utils.estimator_checks import check_estimator, parametrize_with_checks
from sktime.forecasting.nnetsaucemts import MTS


@parametrize_with_checks(MTS)
def test_sktime_api_compliance(obj, test_name):
    check_estimator(obj, tests_to_run=test_name, raise_exceptions=True)
