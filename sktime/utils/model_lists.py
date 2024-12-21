from sktime.classification import (OSCNNClassifier,
                                   OSCNNRegressor)
def construct_all_classifiers(nb_epochs=None):
    """
    Creates a list of all classification networks ready for testing
    Parameters
    ----------
    nb_epochs: int, if not None, value shall be set for all networks that accept it
    Returns
    -------
    map of strings to sktime_dl BaseDeepRegressor implementations
    """
    if nb_epochs is not None:
        # potentially quicker versions for tests
        return {
                        'OSCNNClassifier_quick': OSCNNClassifier(),
        }
    else:
        # the 'literature-conforming' versions
        return {
            'OSCNNClassifier': OSCNNClassifier()
        }
    
def construct_all_regressors(nb_epochs=None):
    """
    Creates a list of all regression networks ready for testing
    :param nb_epochs: int, if not None, value shall be set for all networks
    that accept it
    :return: map of strings to sktime_dl BaseDeepRegressor implementations
    """
    if nb_epochs is not None:
        # potentially quicker versions for tests
        return {
            'OSCNNRegressor_quick': OSCNNRegressor()
        }
    else:
        # the 'literature-conforming' versions
        return {
            'OSCNNRegressor': OSCNNRegressor()
        }
