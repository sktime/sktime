"""Unified high-level interface for various time series related learning tasks."""

from inspect import signature

import numpy as np
import pandas as pd

from sktime.base import BaseObject


class BaseTask(BaseObject):
    """Abstract base task class.

    A task encapsulates metadata information such as the feature and
    target variable which to fit the data to and additional necessary
    instructions on how to fit and predict.

    Implements attributes and operations shared by all tasks,
    including compatibility checks between the concrete task type and
    passed metadata.

    Parameters
    ----------
    target : str
        The column name for the target variable to be predicted.
    features : list of str, optional, (default=None)
        The column name(s) for the feature variable. If None, every column
        apart from target will be used as a feature.
    metadata : pandas.DataFrame
        Contains the metadata that the task is expected to work with.
    """

    def __init__(self, target, features=None, metadata=None):
        # TODO input checks on target and feature args
        self._target = target
        self._features = features if features is None else pd.Index(features)

        self._metadata = None  # initialised as None, properly updated
        # through setter method below
        if metadata is not None:
            self.set_metadata(metadata)  # using the modified setter method below

    @property
    def target(self):
        """Variable target - read-only."""
        return self._target

    @property
    def features(self):
        """Variable features - read-only."""
        return self._features

    @property
    def metadata(self):
        """Variable metadata - read-only."""
        # TODO if metadata is a mutable object itself, its contents may
        #  still be mutable
        return self._metadata

    def set_metadata(self, metadata):
        """Provide missing metadata information to task if not already set.

        This method is especially useful in orchestration where metadata may
        not be
        available when specifying the task.

        Parameters
        ----------
        metadata : pandas.DataFrame
            Metadata container

        Returns
        -------
        self : an instance of self
        """
        # TODO replace whole pandas data container as input argument with
        #  separated metadata container

        # only set metadata if metadata is not already set, otherwise raise
        # error
        if self._metadata is not None:
            raise AttributeError(
                "Metadata is already set and can only be set once, create a "
                "new task for different metadata"
            )

        # check for consistency of information provided in task with given
        # metadata
        self.check_data_compatibility(metadata)

        # set default feature information (all columns but target) using
        # metadata
        if self.features is None:
            self._features = metadata.columns.drop(self.target)

        # set metadata
        self._metadata = {
            "nrow": metadata.shape[0],
            "ncol": metadata.shape[1],
            "target_type": {self.target: type(i) for i in metadata[self.target]},
            "feature_type": {
                col: {type(i) for i in metadata[col]} for col in self.features
            },
        }
        return self

    def check_data_compatibility(self, metadata):
        """Check compatibility of task with passed metadata.

        Parameters
        ----------
        metadata : pandas.DataFrame
            Metadata container
        """
        if not isinstance(metadata, pd.DataFrame):
            raise ValueError(
                f"Metadata must be provided in form of a pandas dataframe, "
                f"but found: {type(metadata)}"
            )

        if self.target not in metadata.columns:
            raise ValueError(f"Target: {self.target} not found in metadata")

        if self.features is not None:
            if not np.all(self.features.isin(metadata.columns)):
                raise ValueError(
                    f"Features: {list(self.features)} not found in metadata"
                )

        if isinstance(self, (TSCTask, TSRTask)):
            if self.features is None:
                if len(metadata.columns.drop(self.target)) == 0:
                    raise ValueError(
                        f"For task of type: {type(self)}, at least one "
                        f"feature must be given, "
                        f"but found none"
                    )

            if metadata.shape[0] <= 1:
                raise ValueError(
                    f"For task of type: {type(self)}, several samples (rows) "
                    f"must be given, but only "
                    f"found: {metadata.shape[0]} samples"
                )

    @classmethod
    def _get_param_names(cls):
        """Get parameter names for the task."""
        # fetch the constructor or the original constructor before
        # deprecation wrapping if any
        init = getattr(cls.__init__, "deprecated_original", cls.__init__)
        if init is object.__init__:
            # No explicit constructor to introspect
            return []

        # introspect the constructor arguments to find the model parameters
        # to represent
        init_signature = signature(init)
        # Consider the constructor parameters excluding 'self'
        parameters = [
            p
            for p in init_signature.parameters.values()
            if p.name != "self" and p.kind != p.VAR_KEYWORD
        ]
        # Extract and sort argument names excluding 'self'
        return sorted([p.name for p in parameters])

    def _get_params(self):
        """Get parameters of the task.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = {key: getattr(self, key, None) for key in self._get_param_names()}
        return out


class TSCTask(BaseTask):
    """Time series classification task.

    A task encapsulates metadata information such as the feature and target
    variable
    to which to fit the data to and any additional necessary instructions on
    how
    to fit and predict.

    Parameters
    ----------
    target : str
        The column name for the target variable to be predicted.
    features : list of str, optional (default=None)
        The column name(s) for the feature variable. If None, every column
        apart from target will be used as a feature.
    metadata : pandas.DataFrame, optional (default=None)
        Contains the metadata that the task is expected to work with.
    """

    def __init__(self, target, features=None, metadata=None):
        self._case = "TSC"
        super().__init__(target, features=features, metadata=metadata)


class TSRTask(BaseTask):
    """Time series regression task.

    A task encapsulates metadata information such as the feature and target
    variable
    to which to fit the data to and any additional necessary instructions on
    how
    to fit and predict.

    Parameters
    ----------
    target : str
        The column name for the target variable to be predicted.
    features : list of str, optional (default=None)
        The column name(s) for the feature variable. If None, every column
        apart from target will be used as a feature.
    metadata : pandas.DataFrame, optional (default=None)
        Contains the metadata that the task is expected to work with.
    """

    def __init__(self, target, features=None, metadata=None):
        self._case = "TSR"
        super().__init__(target, features=features, metadata=metadata)
