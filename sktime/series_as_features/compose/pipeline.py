import numpy as np
import pandas as pd
from joblib import Parallel
from joblib import delayed
from scipy import sparse
from sklearn.pipeline import FeatureUnion as _FeatureUnion
from sklearn.pipeline import Pipeline as _Pipeline
from sklearn.pipeline import _fit_transform_one
from sklearn.pipeline import _transform_one
from sktime.base import BaseEstimator
from sktime.base import MetaEstimatorMixin

__all__ = ["Pipeline"]
__author__ = ["Markus Löning"]


class Pipeline(_Pipeline, BaseEstimator, MetaEstimatorMixin):
    """Pipeline of transforms with a final estimator.

    Sequentially apply a list of transforms and a final estimator.
    Intermediate steps of the pipeline must be 'transforms', that is,
    they must implement fit and transform methods.
    The final estimator only needs to implement fit.
    The transformers in the pipeline can be cached using ``memory`` argument.
    The purpose of the pipeline is to assemble several steps that can be
    cross-validated together while setting different parameters.
    For this, it enables setting parameters of the various steps using their
    names and the parameter name separated by a '__', as in the example below.
    A step's estimator may be replaced entirely by setting the parameter
    with its name to another estimator, or a transformer removed by setting
    to None.

    Parameters
    ----------
    steps : list
        List of (name, transform) tuples (implementing fit/transform) that are
        chained, in the order in which they are chained, with the last object
        an estimator.
    memory : None, str or object with the joblib.Memory interface, optional
        Used to cache the fitted transformers of the pipeline. By default,
        no caching is performed. If a string is given, it is the path to
        the caching directory. Enabling caching triggers a clone of
        the transformers before fitting. Therefore, the transformer
        instance given to the pipeline cannot be inspected
        directly. Use the attribute ``named_steps`` or ``steps`` to
        inspect estimators within the pipeline. Caching the
        transformers is advantageous when fitting is time consuming.
    random_state: : int, RandomState instance or None, optional (default=None)
        Passed random state is propagated to all steps of the pipeline that
        have a random state attribute.
        - If int, random_state is the seed used by the random number generator;
        - If RandomState instance, random_state is the random number generator;
        - If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    named_steps : bunch object, a dictionary with attribute access
        Read-only attribute to access any step parameter by user given name.
        Keys are step names and values are steps parameters.
    """

    _required_parameters = ["steps"]

    def __init__(self, steps, memory=None, random_state=None):
        super(Pipeline, self).__init__(steps, memory=memory)
        self.random_state = random_state

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state

        # If random state is set for entire pipeline, set random state for
        # all random components
        if random_state is not None:
            for step in self.steps:
                if hasattr(step[1], 'random_state'):
                    step[1].set_params(**{'random_state': self.random_state})

    def fit(self, X, y=None, **fit_params):
        super(Pipeline, self).fit(X, y, **fit_params)
        self._is_fitted = True
        return self

    def predict(self, X, **predict_params):
        self.check_is_fitted()
        return super(Pipeline, self).predict(X, **predict_params)


class FeatureUnion(_FeatureUnion, BaseEstimator, MetaEstimatorMixin):
    """Concatenates results of multiple transformer objects.
        This estimator applies a list of transformer objects in parallel to the
        input data, then concatenates the results. This is useful to combine
        several feature extraction mechanisms into a single transformer.
        Parameters of the transformers may be set using its name and the
        parameter
        name separated by a '__'. A transformer may be replaced entirely by
        setting the parameter with its name to another transformer,
        or removed by setting to 'drop' or ``None``.
        Parameters
        ----------
        transformer_list : list of (string, transformer) tuples
            List of transformer objects to be applied to the data. The first
            half of each tuple is the name of the transformer.
        n_jobs : int or None, optional (default=None)
            Number of jobs to run in parallel.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend`
            context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.
        transformer_weights : dict, optional
            Multiplicative weights for features per transformer.
            Keys are transformer names, values the weights.
    """

    def __init__(
            self,
            transformer_list,
            n_jobs=None,
            transformer_weights=None,
            preserve_dataframe=True
    ):
        self.preserve_dataframe = preserve_dataframe
        super(FeatureUnion, self).__init__(
            transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights
        )

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.
        Parameters
        ----------
        X : pandas DataFrame
            Input data to be transformed.
        y : pandas Series, shape (n_samples, ...), optional
            Targets for supervised learning.

        Returns
        -------
        Xt : pandas DataFrame
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        self._validate_transformers()
        result = Parallel(n_jobs=self.n_jobs)(
            delayed(_fit_transform_one)(trans, X, y, weight,
                                        **fit_params)
            for name, trans, weight in self._iter())

        if not result:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        Xs, transformers = zip(*result)
        self._update_transformer_list(transformers)

        Xs = self._hstack(list(Xs))
        self._is_fitted = True
        return Xs

    def fit(self, X, y=None, **fit_params):
        super(FeatureUnion, self).fit(X, y, **fit_params)
        self._is_fitted = True
        return self

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : pandas DataFrame
            Input data to be transformed.
        Returns
        -------
        Xt : pandas DataFrame
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
        self.check_is_fitted()
        Xs = Parallel(n_jobs=self.n_jobs)(
            delayed(_transform_one)(trans, X, None, weight)
            for name, trans, weight in self._iter())

        if not Xs:
            # All transformers are None
            return np.zeros((X.shape[0], 0))

        else:
            return self._hstack(list(Xs))

    def _hstack(self, Xs):
        """
        Stacks X horizontally.
        Supports input types (X): list of
            numpy arrays, sparse arrays and DataFrames
        """

        if any(sparse.issparse(f) for f in Xs):
            Xs = sparse.hstack(Xs).tocsr()

        types = set(type(X) for X in Xs)
        if self.preserve_dataframe and (
                pd.Series in types or pd.DataFrame in types):
            return pd.concat(Xs, axis=1)

        else:
            return np.hstack(Xs)
