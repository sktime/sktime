'''
A module to facilitate testing sktime estimators and its extensions
'''
# building upon pre-existing tests from scikit-learn
from sklearn.utils.estimator_checks import *


def _yield_ts_checks():
    '''
    list of checks to be made
    '''
    # TODO: remove/add appropriate tests (now many tests are removed and some commented)
    yield check_estimators_dtypes  # TODO: add test for xpandas dtypes
    yield check_fit_score_takes_y
#    yield check_dtype_object
    yield check_sample_weights_pandas_series
    yield check_sample_weights_list
    yield check_sample_weights_invariance
    yield check_estimators_fit_returns_self
    yield partial(check_estimators_fit_returns_self, readonly_memmap=True)
    yield check_complex_data
#    yield check_fit2d_predict1d
    yield check_methods_subset_invariance
    yield check_fit2d_1sample
    yield check_fit2d_1feature
#    yield check_fit1d
    yield check_get_params_invariance
    yield check_set_params
    yield check_dict_unchanged
    yield check_dont_overwrite_parameters


def check_ts_estimator(Estimator):
    '''
    Check if estimator adheres to sktime conventions.
    Parameters
    ----------
    estimator : estimator object or class
        Estimator to check. Estimator is a class object or instance.
    '''
    if isinstance(Estimator, type):
        # got a class
        name = Estimator.__name__
        estimator = Estimator()
        check_parameters_default_constructible(name, Estimator)
        check_no_attributes_set_in_init(name, estimator)
    else:
        # got an instance
        estimator = Estimator
        name = type(estimator).__name__

    for check in _yield_ts_checks():
        check(name, estimator)
