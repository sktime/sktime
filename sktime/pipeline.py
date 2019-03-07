from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.utils._joblib import Parallel, delayed
import pandas as pd
import numpy as np
from scipy import sparse


class TSPipeline(Pipeline):
    def __init__(self, steps, memory=None, random_state=None, check_input=True):
        super(TSPipeline, self).__init__(steps, memory=memory)
        self.random_state = random_state
        self.check_input = check_input

    @property
    def random_state(self):
        return self._random_state

    @random_state.setter
    def random_state(self, random_state):
        self._random_state = random_state

        # If random state is set for entire pipeline, set random state for all random components
        if random_state is not None:
            for step in self.steps:
                if hasattr(step[1], 'random_state'):
                    step[1].set_params(**{'random_state': self.random_state})

    @property
    def check_input(self):
        return self._check_input

    @check_input.setter
    def check_input(self, check_input):
        self._check_input = check_input

        # If check_input is set for entire pipeline, set check input for all components
        if not check_input:
            for step in self.steps:
                if hasattr(step[1], 'check_input'):
                    step[1].set_params(**{'check_input': self.check_input})


def _fit_one_transformer(transformer, X, y, weight=None, **fit_params):
    return transformer.fit(X, y)


def _transform_one(transformer, X, y, weight, **fit_params):
    res = X.apply(transformer.transform)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res
    return res * weight


def _fit_transform_one(transformer, X, y, weight, **fit_params):
    if hasattr(transformer, 'fit_transform'):
        res = X.apply(transformer.fit_transform, **fit_params)
    else:
        res = X.apply(transformer.fit(X, y, **fit_params).transform)
    # if we have a weight for this transformer, multiply output
    if weight is None:
        return res, transformer
    return res * weight, transformer


class TSFeatureUnion(FeatureUnion):

    def __init__(
            self,
            transformer_list,
            n_jobs=None,
            transformer_weights=None,
            preserve_dataframe=True
    ):
        self.preserve_dataframe = preserve_dataframe
        super(TSFeatureUnion, self).__init__(
            transformer_list,
            n_jobs=n_jobs,
            transformer_weights=transformer_weights
        )

    def fit_transform(self, X, y=None, **fit_params):
        """Fit all transformers, transform the data and concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        y : array-like, shape (n_samples, ...), optional
            Targets for supervised learning.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
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

        return self._hstack(list(Xs))

    def transform(self, X):
        """Transform X separately by each transformer, concatenate results.
        Parameters
        ----------
        X : iterable or array-like, depending on transformers
            Input data to be transformed.
        Returns
        -------
        X_t : array-like or sparse matrix, shape (n_samples, sum_n_components)
            hstack of results of transformers. sum_n_components is the
            sum of n_components (output dimension) over transformers.
        """
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
        if self.preserve_dataframe and (pd.Series in types or pd.DataFrame in types):
            return pd.concat(Xs, axis=1)

        else:
            return np.hstack(Xs)
