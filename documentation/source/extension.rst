Extension Guidelines
====================

Package Structure
-----------------

The (typical) directory structure of the module is shown below.::

    sktime
    ├── __init__.py
    ├── classifiers
    │   ├── __init__.py
    │   ├── base.py
    │   ├── example_classifiers.py
    │   └── time_domain_classification
    │       ├── __init__.py
    │       ├── elastic_ensemble.py
    │       └── ts_distance_measures.py
    ├── model_selection.py
    ├── regressors
    │   ├── __init__.py
    │   ├── base.py
    │   ├── example_regressors.py
    ├── tests
    │   ├── __init__.py
    │   ├── test_common.py
    │   ├── test_pipelines.py
    │   ├── test_TSDummyClassifier.py
    │   ├── test_TSDummyRegressor.py
    │   ├── test_TSExampleClassifier.py
    │   └── test_TSExampleRegressor.py
    ├── transformers
    │   ├── __init__.py
    │   ├── base.py
    │   ├── compose.py
    │   ├── example_transformers.py
    │   ├── series_to_series.py
    │   └── series_to_tabular.py
    └── utils
        ├── __init__.py
        ├── estimator_checks.py
        ├── load_data.py
        └── validation.py

Each directory (with an `__init__.py`) file is considered as a package and each `.py` file is considered as a module in that package or sub-package. Packages consist of a number of modules and each module has a number of functions or classes that could be logically grouped together.

Regressors
----------
All regressors should inherit from BaseRegressor from `sktime/regressors/base.py`. At the very least it should have a constructor, fit function and predict function.

Constructor
~~~~~~~~~~~
The `__init__()` constructor should get all hyperparameters as keyword arguments and assign them to pubic members of the same name.
Should also have a keyword argument `check_inputs=True` defaulting to True.

fit
~~~
The fit function takes `X` and `y` as inputs.
It should perform input checks if and only if the regressor is constructed with `check_inputs=True`.
Input checks can be performed using `validate_X_y(X,y)`.
The fit function, after performing the fitting operation if any, should set the is_setted flag `self.is_fitted_=True`.
It should return `self`, which is the regressor itself.

predict
~~~~~~~~
The predict function takes only `X`.
It should always check for validity of inputs.
Input checks can be performed using `validate_X(X)`.
It should check if the regressor is already fit using `check_is_fitted(self, 'is_fitted_')`
The predict function, should make predictions and return these predictions.

example
~~~~~~~
The following is an example regressor

.. code-block:: python

    # use relative imports for sub-packages and modules of sktime
    from ..utils.validation import validate_X_y, validate_X, check_is_fitted
    from .base import BaseRegressor

    class MyRegressor(BaseRegressor):
        """docstring here using numpy syntax
        """
        def __init__(self, check_inputs=True, param1=value1, param2=value2):
            self.param1 = value1
            self.param2 = value2

        def fit(self, X, y):
            """docstring here using numpy syntax
            """
            if self.check_inputs:
                validate_X_y(X, y)
            # perform fitting here
            ...
            # set completion flag
            self.is_fitted_ = True
            # `fit` should always return `self`
            return self

        def predict(self, X):
            """docstring here using numpy syntax
            """
            validate_X(X)
            check_is_fitted(self, 'is_fitted_')
            # predict here
            result = ...
            return result

Classifiers
-----------
All classifiers should inherit from BaseClassifier from `sktime/classifiers/base.py`. At the very least it should have a constructor, fit function and predict function.

Constructor
~~~~~~~~~~~
The `__init__()` constructor should get all hyperparameters as keyword arguments and assign them to pubic members of the same name.
Should also have a keyword argument `check_inputs=True` defaulting to True.

fit
~~~
The fit function takes `X` and `y` as inputs.
It should perform input checks if and only if the classifier is constructed with `check_inputs=True`.
Input checks can be performed using `validate_X_y(X,y)`.
The fit function, after performing the fitting operation if any, should set the is_setted flag `self.is_fitted_=True`.
It should return `self`, which is the classifier itself.

predict
~~~~~~~~
The predict function takes only `X`.
It should always check for validity of inputs.
Input checks can be performed using `validate_X(X)`.
It should check if the classifier is already fit using `check_is_fitted(self, 'is_fitted_')`
The predict function, should make predictions and return these predictions.

predict_proba (optional)
~~~~~~~~~~~~~~~~~~~~~~~~
The predict_proba function takes only `X`.
It should always check for validity of inputs.
Input checks can be performed using `validate_X(X)`.
It should check if the classifier is already fit using `check_is_fitted(self, 'is_fitted_')`
The predict_proba function, should make predictions and return these prediction probabilities for each classes.

example
~~~~~~~
The following is an example classifier

.. code-block:: python

    # use relative imports for sub-packages and modules of sktime
    from ..utils.validation import validate_X_y, validate_X, check_is_fitted
    from .base import BaseClassifier

    class MyClassifier(BaseClassifier):
        """docstring here using numpy syntax
        """
        def __init__(self, check_inputs=True, param1=value1, param2=value2):
            self.param1 = value1
            self.param2 = value2

        def fit(self, X, y):
            """docstring here using numpy syntax
            """
            if self.check_inputs:
                validate_X_y(X, y)
            # perform fitting here
            ...
            # set completion flag
            self.is_fitted_ = True
            # `fit` should always return `self`
            return self

        def predict(self, X):
            """docstring here using numpy syntax
            """
            validate_X(X)
            check_is_fitted(self, 'is_fitted_')
            # predict here
            result = ...
            return result

Transformers
------------
All transformers should inherit from BaseTransformer from `sktime/transformers/base.py`. At the very least it should have a constructor, fit function and predict function.
Based on the inputs types they work with and the return data type, the transformers can be classified as
 + series to series
   - per row transformer
   - per column transformer
 + series to tabular
   - per row transformer
   - per column transformer
The class definitions shoud be placed in the corresponding locations in the folder structure.

Constructor
~~~~~~~~~~~
The `__init__()` constructor should get all hyperparameters as keyword arguments and assign them to pubic members of the same name.
Should also have a keyword argument `check_inputs=True` defaulting to True.

fit
~~~
The fit function takes `X` and `y` as inputs.
`y` is a dummy input and should always take the default value None.
It should perform input checks if and only if the transformer is constructed with `check_inputs=True`.
Input checks can be performed using `validate_X(X)`.
The fit function, after performing the fitting operation if any, should set the is_setted flag `self.is_fitted_=True`.
It should return `self`, which is the transformer itself.

transform
~~~~~~~~~
The transform function takes only `X`.
It should perform input checks if and only if the transformer is constructed with `check_inputs=True`.
Input checks can be performed using `validate_X(X)`.
It should check if the transformer is already fit using `check_is_fitted(self, 'is_fitted_')`
The transform function, should return the appropriately transformed data.

example
~~~~~~~
The following is an example transformer

.. code-block:: python

    # use relative imports for sub-packages and modules of sktime
    from ..utils.validation import validate_X_y, validate_X, check_is_fitted
    from .base import BaseTransformer

    class MyTransformer(BaseTransformer):
        """docstring here using numpy syntax
        """
        def __init__(self, check_inputs=True, param1=value1, param2=value2):
            self.param1 = value1
            self.param2 = value2

        def fit(self, X, y=None):
            """docstring here using numpy syntax
            """
            if self.check_inputs:
                validate_X(X)
            # perform fitting here
            ...
            # set completion flag
            self.is_fitted_ = True
            # `fit` should always return `self`
            return self

        def transform(self, X):
            """docstring here using numpy syntax
            """
            if self.check_inputs:
                validate_X(X)
            check_is_fitted(self, 'is_fitted_')
            # transform the input here
            result = ...
            return result

