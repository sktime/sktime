#//Implementation Of Echo State Networks (ESN) for Time Series Classification
__author__ = ["sharma-kshitij-ks"]

from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from scipy.sparse import random
from sklearn.linear_model import RidgeClassifier
from sktime.utils.validation import check_array
from sktime.utils.validation.series import check_series
from sktime.classification.base import BaseClassifier

class EchoStateNetwork(BaseClassifier):
  """Echo State Network (ESN) Classifier.
  
  Overview:
  Echo State Networks are a type of reservoir computing model for
  time series analysis. The core idea is to use a fixed, randomly
  initialized reservoir of recurrently connected units, combined
  with a linear readout layer trained using ridge regression.
  
  Parameters:
  ----------
  base_classifier: sklearn classifier, default=RidgeClassifier(alpha=1.0, random_state=None)
      The base classifier to use for training the readout layer. 
  n_reservoir: int, default=100
      Number of reservoir units.
  spectral_radius: float, default=0.99
      Spectral radius of the reservoir weight matrix.
  leaking_rate: float, default=0.3
      Leaking rate controls the amount of information that is
      propagated from one time step to the next in the reservoir.
  regularization: float, default=1e-6
      Regularization parameter for ridge regression.
  random_state: int or None, default=None
      Seed for random number generator.
  
  Attributes:
  ----------
  W_in_: array-like, shape (n_reservoir, n_features)
      Input weights for the reservoir units.
  W_res_: array-like, shape (n_reservoir, n_reservoir)
      Reservoir weights.
  W_out_: array-like, shape (n_reservoir, n_classes)
      Output weights for the readout layer.
  
  References:
  -----------
  - Lukoševičius, Mantas. "A practical guide to applying echo state networks."
    Neural Networks: Tricks of the Trade. Springer, Berlin, Heidelberg, 2012. 659-686.
  - https://en.wikipedia.org/wiki/Echo_state_network
  """
  
  _tags = {
    "authors": ["sharma-kshitij-ks"],
    "maintainers": ["fkiraly", "sharma-kshitij-ks"]
  }

  def __init__(
    self,
    base_classifier=RidgeClassifier(alpha=1.0, random_state=None),
    n_reservoir=100,
    spectral_radius=0.99,
    leaking_rate=0.3,
    regularization=1e-6,
    random_state=None):
      
    self.base_classifier = base_classifier
    self.n_reservoir = n_reservoir
    self.spectral_radius = spectral_radius
    self.leaking_rate = leaking_rate
    self.regularization = regularization
    self.random_state = random_state

    self.W_in_ = None
    self.W_res_ = None
    self.W_out_ = None

  def _fit(self, X, y):
    """Fit the Echo State Network classifier.

    Parameters:
    ----------
    X: array-like, shape (n_samples, n_features)
        Training time series data.
    y: array-like, shape (n_samples,)
        Training labels.

    Returns:
    -------
    self: object """
    
    X = check_array(X, ensure_min_features=1, ensure_min_samples=2, dtype="float64")
    y = check_series(y, enforce_univariate=True, coerce_to_numpy=True)

    n_samples, n_features = X.shape
    self.W_in_ = random(n_features, self.n_reservoir, density=0.2, random_state=self.random_state).A
    self.W_res_ = random(self.n_reservoir, self.n_reservoir, density=0.2, random_state=self.random_state).A

    X_reservoir = np.zeros((n_samples, self.n_reservoir))
    for t in range(1, n_samples):
        X_reservoir[t] = np.tanh(np.dot(self.W_in_, X[t]) + np.dot(self.W_res_, X_reservoir[t - 1]))

    if not isinstance(self.base_classifier, BaseEstimator):
        self.base_classifier_ = self.base_classifier(alpha=self.regularization, random_state=self.random_state)
    else:
        self.base_classifier_ = self.base_classifier

    self.base_classifier_.fit(X_reservoir, y)
    return self


  def _predict(self, X):
    """Make predictions using the trained ESN.
    
    Parameters:
    ----------
    X: array-like, shape (n_samples, n_features)
        Test time series data.
    
    Returns:
    -------
    y_pred: array-like, shape (n_samples,)
        Predicted class labels.
    """
    X = check_array(X, ensure_min_features=1, ensure_min_samples=2, dtype="float64")
    
    n_samples, _ = X.shape
    X_reservoir = np.zeros((n_samples, self.n_reservoir))
    for t in range(1, n_samples):
        X_reservoir[t] = np.tanh(np.dot(self.W_in_, X[t]) + np.dot(self.W_res_, X_reservoir[t - 1]))
    
    y_pred = self.base_classifier_.predict(X_reservoir)
    return y_pred

  
  @classmethod
  def get_test_params(cls, parameter_set="default"):
   """Return testing parameter settings for the estimator.

    Parameters
    ----------
    parameter_set : str, default="default"
        Name of the set of test parameters to return, for use in tests. If no
        special parameters are defined for a value, will return ``"default"`` set.
        custom test set 1 = "test-param-set-1"
        custom test set 2 = "test-param-set-2"

    Returns
    -------
    params : list of dict
        Parameters to create testing instances of the class.
        Each dict are parameters to construct an "interesting" test
        instance, i.e., ``MyClass(**params)`` or ``MyClass(**params[i])``
        creates a valid test instance.
        ``create_test_instance`` uses the first (or only) dictionary in ``params``.
    """
    params = [
        {"base_classifier": None, "n_reservoir": None, "spectral_radius": None, "leaking_rate": None, "regularization": None, "random_state": None},
        {"base_classifier": DecisionTreeClassifier(max_depth=5, random_state=42), "n_reservoir": 50, "spectral_radius": 0.9, "leaking_rate": 0.5, "regularization": 1e-4, "random_state": 123},
        {"base_classifier": RandomForestClassifier(n_estimators=100, random_state=99), "n_reservoir": 75, "spectral_radius": 0.95, "leaking_rate": 0.4, "regularization": 1e-5, "random_state": 456}
    ]

    if parameter_set == "default":
      return params[0]
    elif parameter_set == "test-param-set-1":
      return params[1]
    elif parameter_set == "test-param-set-2":
      return params[2]
    else:
      raise ValueError(f"Unknown parameter_set: {parameter_set}")
