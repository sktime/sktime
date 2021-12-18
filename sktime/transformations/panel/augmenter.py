# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Panel transformers for time series augmentation."""

__author__ = ["MrPr3ntice", "MFehsenfeld", "iljamaurer"]
__all__ = ["SeqAugPipeline",
           "plot_augmentation_examples",
           "get_rand_input_params",
           "WhiteNoiseAugmenter",
           "ReverseAugmenter",
           "InvertAugmenter",
           "ScaleAugmenter",
           "OffsetAugmenter",
           "DriftAugmenter"]

import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from scipy.stats._distn_infrastructure import rv_frozen as random_Variable
from datetime import datetime
from sktime.transformations.base import BaseTransformer
from sktime.datatypes import get_examples


class _BasePanelAugmenter(BaseTransformer):
    """Abstract Class for all (panel) augmentation transformer.

    This class provides basic functionality for all specific time series
    augmenters. A panel augmenter transforms all instances of the selected
    variables of a time series panel. A fitting is only necessary, if
    certain augmentation parameters depend on (mostly statistical properties
    of) training data, e.g. if the augmentation should add Gaussian noise
    with a certain standard deviation (std) relative to the empirical std of the
    training data (i.e. to ensure a certain Signal-to-Noise ratio, SNR).

    Parameters
    ----------
    p: float, optional (default = 1.0)
        Probability, that a univariate time series (i.e.: defined by (instance,
        variable) of a multivariate panel) is augmented.
        Otherwise the original  time series is kept. In case of p=1.0,
        every  time series in the panel is augmented. Notice, that in case of
        multivariate time series the decision whether a certain variable of
        an instance is augmented is stochastically independent from the
        other variables of the same instance.
    param: any, optional (default = None)
        a single parameter defining the augmentation.
        In case of e.g. a scale augmentation, this might be a constant scaling
        factor or a scipy distribution to draw i.i.d. scaling factors from.
        See the documentation of the specific augmenter for details.
    use_relative_fit: bool, optional (default = False)
        Whether to a fit the augmenter regarding a statistical parameter of
        the data. If True, relative_fit_stat_fun specifies the statistical
        parameter and relative_fit_type specifies the fitting reference data.
    relative_fit_stat_fun: a function, optional (default = np.std)
        Specifies the statistical parameter to calculate. Any function
        that satisfies the following syntax is allowed: stat = fun(X:
        pd.Series). The only allowed input is a numeric pd.Series and output
        has to be float or int. The default is the standard deviation np.std.
    relative_fit_type: str, optional (default = "fit")
        "fit": fit demanded statistics with a separate train set.
        "fit-transform": fit statistics just before transformation
        regarding the whole given panel.
        "instance-wise": fit statistics just before transformation
        individually per instance (and variable).
    random_state: int, optional (default = None)
        A random state seed for reproducibility.
    excluded_var_indices: iterable of int optional (default = None)
        Iterable (e.g. tuple or list) of int, containing the indices of those
        variables to exclude from augmentation. Default is None and all
        variables are used.
    n_jobs: int, optional (default = 1)
        Integer specifying the maximum number of concurrently running
        workers on specific panel's instances. If 1 is given, no parallelism is
        used. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus
        + 1 + n_jobs) are used. For example with n_jobs=-2, all CPUs but one
        are used. Not implemented yet.
    """

    _tags = {
        "scitype:transform-input": "Panel",
        "scitype:transform-output": "Panel",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "univariate-only": False,
        "handles-missing-data": False,
        "X_inner_mtype": "nested_univ",
        "y_inner_mtype": "pd.Series",
        "X-y-must-have-same-index": False,
        "enforce_index_type": None,
        "fit-in-transform": False,
        "transform-returns-same-time-index": False,
        "skip-inverse-transform": True,
    }

    def __init__(self,
                 p: float = 1.0,
                 param=None,
                 use_relative_fit=False,
                 relative_fit_stat_fun=np.std,
                 relative_fit_type="fit",
                 random_state=None,
                 excluded_var_indices=None,
                 n_jobs=1):
        # input parameters
        self.p = p
        self._param = param
        self.use_relative_fit = use_relative_fit
        self.relative_fit_stat_fun = relative_fit_stat_fun
        self.relative_fit_type = relative_fit_type
        self.random_state = random_state
        if excluded_var_indices is None:
            self.excluded_var_indices = []
        else:
            self.excluded_var_indices = excluded_var_indices
        self.n_jobs = n_jobs
        # DataFrame of latest random variates of any random variable defined
        # by a single augmenter.
        self._last_aug_random_variates = None
        # descriptive statistics of data passed to _fit() function.
        self._stats = None
        # number of vars/channels as defined by data passed to _fit() function.
        self._n_vars = None
        # determine whether the augmenter can be fitted, if not done by subclass
        if not hasattr(self, '_is_fittable'):
            self._is_fittable = True
        # add default param description, if not done by subclass
        if not hasattr(self, '_param_desc'):
            self._param_desc = None
        # check augmentation parameters
        self._check_general_aug_params()
        self._check_specific_aug_params()
        # initialize super class
        super().__init__()

    def _fit(self, X, y=None):
        """Fit panel augmentation transformer to X.

        This function fits the augmenter regarding X if member variable
        relative_fit_type == "fit" and use_relative_fit == True.

        Parameters
        ----------
        X : Panel of pd.DataFrame
            Uni- or multivariate dataset.
        y : Series or Panel
            Always ignored, exists for compatibility.

        Returns
        -------
        self: a fitted instance of the transformer
        """
        if not self._is_fittable:
            # nothing has to be fit here
            return self
        elif self.use_relative_fit and self.relative_fit_type == "fit":
            # calculate demanded statistical param for each variable over
            # (a concatenation of) all instances in the given panel.
            self._n_vars = X.shape[1]  # get number of vars from X
            self._stats = []
            for col in range(self._n_vars):  # loop over demanded variables
                if col not in self.excluded_var_indices:
                    long_series = pd.Series(dtype='float64')
                    for row in range(X.shape[0]):  # loop over instances
                        long_series = long_series.append(
                            X.iloc[row, col], ignore_index=True)
                    self._stats.append(self.relative_fit_stat_fun(long_series))
                else:
                    self._stats.append(None)
            return self
        # if use_relative_fit is False or relative_fit_type is "fit-transform"
        # or "instance-wise"
        else:
            # in this case no fitting is necessary
            return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : Series or Panel of pd.DataFrame
            Uni- or multivariate dataset.
        y : Series or Panel, optional (default=None)
            Always ignored, exists for compatibility.

        Returns
        -------
        pd.DataFrame: The transformed version of X.
        """
        # create copied DataFrame for transformed data
        Xt = X.copy()
        # Xt = pd.DataFrame(dtype=object).reindex_like(X).astype(object)
        self._last_aug_random_variate =\
            pd.DataFrame(dtype=object).reindex_like(X).astype(object)
        # check number of vars
        if Xt.shape[1] != self._n_vars and self._n_vars is not None:
            raise ValueError(
                "The number of variables differs between input "
                "data (" + str(Xt.shape[1]) + ") and the data "
                "used for fitting (" + str(self._n_vars) + ").")

        # fit-transform
        if self.use_relative_fit and self.relative_fit_type == "fit-transform" \
                and not self._is_fittable:
            # calculate demanded statistical param over (a concatenation of)
            # all instances for each variable (like in case of "fit" but
            # directly on the data to be transformed).
            self._stats = []
            for col in range(X.shape[1]):  # loop over demanded variables
                if col not in self.excluded_var_indices:
                    long_series = pd.Series(dtype='float64')
                    for row in range(Xt.shape[0]):  # loop over instances
                        long_series = long_series.append(
                            X.iloc[row, col], ignore_index=True)
                    self._stats.append(
                        self.relative_fit_stat_fun(long_series))
                else:
                    self._stats.append(None)

        # create dummy statistics list for instance-wise and use_relative_fit ==
        # False
        if (self.use_relative_fit and
            self.relative_fit_type == "instance-wise") \
                or not self.use_relative_fit or not self._is_fittable:
            self._stats = [None] * X.shape[1]
        # loop over variables
        for col in range(X.shape[1]):
            # loop over instances (slow but consistent)
            for row in range(Xt.shape[0]):
                # throw the dice if transformation is performed or not
                if np.random.rand() <= self.p \
                        and col not in self.excluded_var_indices:
                    if self.relative_fit_type == "instance-wise"\
                            and self._is_fittable:
                        # (overwrite) statistics for certain instance
                        self._stats[col] = self.relative_fit_stat_fun(
                            X.iloc[row, col])
                    if isinstance(self._param, random_Variable):
                        # if parameter is a distribution and not a constant
                        rand_param_variate = self._param.rv()
                        self._last_aug_random_variate.iloc[row, col] = \
                            rand_param_variate
                    elif self._param is not None:  # if param is constant,
                        # not None and not random
                        rand_param_variate = self._param
                        self._last_aug_random_variate.iloc[row, col] = \
                            rand_param_variate
                    else:  # if param is None, but augmentation takes place
                        rand_param_variate = self._param
                        self._last_aug_random_variate.iloc[row, col] = None
                    # perform augmentation
                    Xt.iat[row, col] = self._univariate_ser_aug_fun(
                        X.iloc[row, col],
                        rand_param_variate,
                        self._stats[col])
                else:
                    # if no augmentation takes place -> keep original TS
                    # instance
                    self._last_aug_random_variate.iloc[row, col] = False
        return Xt

    def _univariate_ser_aug_fun(self, X, rand_param_variate, stat_param):
        """Abstract function to be overwritten by inheriting subclass"""
        raise NotImplementedError

    def _check_general_aug_params(self):
        """Checking input augmentation parameters"""
        if not 0.0 <= self.p <= 1.0:
            raise ValueError("Input value for p is not a valid probability.")
        if not isinstance(self.use_relative_fit, bool):
            raise TypeError("Type of input value use_relative_fit must be"
                            " bool.")
        if not callable(self.relative_fit_stat_fun)\
                and self.relative_fit_stat_fun is not None:
            raise TypeError("Type of input value relative_fit_stat_fun must be"
                            " function or None.")
        if self.relative_fit_type not in ("fit", "fit-transform",
                                          "instance-wise", None):
            raise ValueError("Input value for relative_fit_type is invalid.")
        if not isinstance(self.excluded_var_indices, list):
            raise TypeError("Input value excluded_var_indices must be a list "
                            "of non-negative integers.")
        if not isinstance(self.n_jobs, int):
            raise TypeError("Type of input value n_jobs must be int.")

    def _check_specific_aug_params(self):
        """Default methode for subclass-specific parameter checking"""
        if self._param_desc is not None:
            if self._param is None:
                self._param = self._param_desc["default"]
            elif isinstance(self._param, random_Variable):
                pass
            elif not self._param_desc["min"] <= self._param <= \
                    self._param_desc["max"]:
                raise ValueError(
                    "Input value for param is out of boundaries.")
        elif not isinstance(self._param, (int, float, random_Variable)) \
                and self._param is not None:
            raise TypeError(f"Type of input value param must be int, float or "
                            f"a distribution of these, not {type(self._param)}.")

    def _plot_augmentation_examples(self, X, y):
        """Plots original and augmented instance examples for each variable.

        This is a wrapper function calling the static function
        plot_augmentation_examples() for compatibility reasons.
        """
        plot_augmentation_examples(self, X, y)


class SeqAugPipeline(Pipeline):
    """ Subclass of `sklearn.pipeline.Pipeline`, adding functionality. [1]_

    References
    ----------
    .. [1] https://scikit-learn.org/stable/modules/generated/sklearn
       .pipeline.Pipeline.html

    Examples
    --------
    >>> from sktime.transformations.panel.augmenter import *
    >>> # create simple panel dataset with 2 variables and 3 instances
    >>> X = pd.DataFrame([[pd.Series([0,1,2,3])] * 2] * 3)
    >>> # set up Pipeline
    >>> pipe = SeqAugPipeline([
    >>>    ('invert', aug.InvertAugmenter(p=0.5)),
    >>>    ('reverse', aug.ReverseAugmenter(p=0.5))])
    >>> Xt = pipe.fit_transform(X)
    >>> print(Xt.iloc[0, 0])
    >>> # get information about the augmentations' random decisions
    >>> print(pipe.get_last_aug_random_variates())
    """
    def __init__(self, steps, memory=None, verbose=False):
        super().__init__(steps, memory=memory, verbose=verbose)

    def get_last_aug_random_variates(self):
        """Info about last augmentation from each transformer in pipeline.

        Returns
        -------
        list: List of pd.DataFrames of shape [n_instances, n_vars] for each
            augmenter in the pipeline (other transformers or estimators in
            `steps` are ignored). If an augmentation took place during the
            last transformation call, the cell value is the used (random)
            parameter (or None, if no parameter is needed), otherwise it is
            False.
        """
        list_of_aug_info = []
        for aug in self.steps:
            if isinstance(aug, _BasePanelAugmenter):
                list_of_aug_info.append(aug[1]._last_aug_random_variate)
        return list_of_aug_info

    @staticmethod
    def draw_random_samples(X,
                            y=None,
                            n=1.0,
                            shuffle_and_stratify=True,
                            without_replacement=True,
                            random_state=None):
        """Draw random instances form panel data.

        Parameters
        ----------
        X: pd.DataFrame
            Panel data to sample/draw from.
        y: pd.Series, optional (default = None)
            Target variable. Is Needed if shuffle_and_stratify is True.
        n: int or float, optional (default = 1.0)
            Number of instances to draw. If type of n is float,
            it is interpreted as the proportion of instances to draw compared
            to the number of instances in X. By default, the same number of
            instances as given by X is returned.
        shuffle_and_stratify: bool, optional (default = True)
            Whether to shuffle and stratify the samples to draw.
        without_replacement: bool, optional (default = True)
            Whether to draw without replacement. If True, between two
            subsequent draws of the same original instance, every other
            instance of X appears once or twice.
        random_state: int
            Random state seed.

        Returns
        -------
        pd.Dataframe: Drawn data.
        pd.Series: Corresponding target values. This is only returned if
            input y is given.
        list of int: List with the drawn indices from the original data.
        """
        # check inputs
        n_instances = X.shape[0]
        if isinstance(n, float):
            if n <= 0.0 or not np.isfinite(n):
                raise ValueError("n must be a positive, finite number.")
            n = np.ceil(n_instances * n)
        elif isinstance(n, int):
            if n < 1 or not np.isfinite(n):
                raise ValueError("n must be a finite number >= 1.")
        else:
            raise ValueError("n must be int or float, not " + str(type(n))) \
                  + "."
        # calculate indices
        if shuffle_and_stratify and without_replacement and y is not None:
            idx_list = []
            sss = StratifiedShuffleSplit(n_splits=int(np.floor(n /
                                                               n_instances)),
                                         test_size=0.5,
                                         random_state=random_state)
            for idx_a, idx_b in sss.split(X, y):
                idx_list = idx_a.tolist() + idx_b.tolist()
            sss = StratifiedShuffleSplit(n_splits=1,
                                         test_size=np.mod(n, n_instances),
                                         random_state=random_state)
            for idx_a, idx_b in sss.split(y, y):
                idx_list += idx_b.tolist()
        else:
            raise NotImplementedError("Not implemented yet.")
        # check number of indices
        if n != len(idx_list):
            raise ValueError("The index list must contain n = " + str(n) +
                             "indices, but contains " + str(len(idx_list)) +
                             " indices.")
        
        X_aug = X.iloc[idx_list]
        # Need to reset_index to pass index.is_monotonic of
        # check_pdDataFrame_Series() in datatypes/_series/_check.py
        X_aug.reset_index(inplace=True, drop=True)
        if y is not None:
            y_aug = y.iloc[idx_list]
            y_aug.index = X_aug.index
            return X_aug, y_aug, idx_list
        else:
            return X_aug, idx_list

    def _plot_augmentation_examples(self, X, y):
        """Plots original and augmented instance examples for each variable.

        This is a wrapper function calling the static function
        plot_augmentation_examples() for compatibility reasons.
        """
        plot_augmentation_examples(self, X, y)


# static functions
def plot_augmentation_examples(fitted_transformer,
                               X,
                               y=None,
                               n_instances_per_variable=5,):
    """Plots original and augmented instance examples for each variable.

    Parameters
    ----------
    fitted_transformer: fitted transformer
        A fitted (augmentation) transformer.
    X: Panel of pd.DataFrame
        Uni- or multivariate dataset.
    y: Series or Panel, optional (default=None)
        Target variable, if y is available and of categorical scale,
        it will be used to stratify the randomly drawn examples.
    n_instances_per_variable: int, optional (default = 5)
        number of time series to draw per variable (row).

    Returns
    -------
    matplotlib.figure.Figure: A figure with a [n_variables, 2] subplot-grid.
    """
    n_vars = X.shape[1]  # get number of variables of X
    # pick (stratified regarding categorical y) examples from the original input
    # data
    X, y, idx = SeqAugPipeline.draw_random_samples(
        X,
        y,
        n=n_instances_per_variable,
        shuffle_and_stratify=True,
        without_replacement=True)
    # make sure, that the given transformer is fitted
    if not fitted_transformer._is_fitted:
        fitted_transformer.fit(X, y)
    # get augmented data
    Xt = fitted_transformer.transform(X, y)

    # create description string (parameterization of the augmenter)
    try:
        ft = fitted_transformer
        if isinstance(ft._param, (float, int)):
            help_param = f'{ft._param:.4f}'
        else:
            help_param = str(ft._param)
        nl = '\n'
        param_str = \
            f"{ft.__class__.__name__} with parameters{nl}" \
            f"p={ft.p:.2f}, " \
            f"param={help_param}, " \
            f"use_relative_fit={ft.use_relative_fit},{nl}" \
            f"relative_fit_stat_fun={ft.relative_fit_stat_fun.__name__}, " \
            f"relative_fit_type='{ft.relative_fit_type}', " \
            f"random_state={ft.random_state},{nl}" \
            f"excluded_var_indices={ft.excluded_var_indices}, " \
            f"n_jobs={ft.n_jobs}.{nl}" \
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    except Exception as e:
        print(e)
        param_str = "Unknown augmenter and parameterization"

    # plot and return figure
    fig, axs = plt.subplots(n_vars, 2, figsize=(12, 1.8 + 2.2 * n_vars))
    for i in range(n_vars):
        for j in range(n_instances_per_variable):
            axs[i, 0].plot(X.iloc[j][i], label=y.iloc[j])
            axs[i, 1].plot(Xt.iloc[j][i], label=y.iloc[j])
        axs[i, 0].legend()
        axs[i, 1].legend()
        top_lim = max(*axs[i, 0].get_ylim(), *axs[i, 1].get_ylim())
        bot_lim = min(*axs[i, 0].get_ylim(), *axs[i, 1].get_ylim())
        axs[i, 0].set_ylim(top_lim, bot_lim)
        axs[i, 1].set_ylim(top_lim, bot_lim)
        axs[i, 0].set_title('Original time series from variable ' + str(i))
        axs[i, 1].set_title('Augmented time series from variable ' + str(i))
        axs[i, 0].grid()
        axs[i, 1].grid()
    plt.suptitle(param_str, y=0.96, fontsize=12)
    plt.tight_layout()
    fig.subplots_adjust(top=0.84)
    return fig


def get_rand_input_params(n_vars):
    """Draw random input parameters for an augmenter (as a dict)."""
    types = ["fit", "fit-transform", "instance-wise"]
    shuffled_var_idx = list(range(n_vars))
    np.random.shuffle(shuffled_var_idx)
    n_excluded_vars = np.random.randint(n_vars)
    excluded_var_indices = list(np.sort(shuffled_var_idx[:n_excluded_vars]))
    rtn_dict = {
        "p": np.random.rand(),
        "param": np.random.normal(5, 10),
        "use_relative_fit": np.random.rand() > 0.5,
        "relative_fit_stat_fun": np.std,
        "relative_fit_type": types[np.random.randint(0, 3)],
        "random_state": None,
        "excluded_var_indices": excluded_var_indices,
        "n_jobs": 1}
    return rtn_dict


def progress_bar(count, total, status=''):
    """Print progress bar to console. Utility for long lasting processing"""
    bar_len = 40
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)
    percents = f"{percents:.1f}"
    sys.stdout.write('[%s] %s%s | %s     \r' % (bar, percents, '%', status))
    sys.stdout.flush()


# INDIVIDUAL AUGMENTERS
class WhiteNoiseAugmenter(_BasePanelAugmenter):
    """Augmenter adding Gaussian (i.e . white) noise to the time series.

    This is a sub class inheriting its functionality from the superclass
    `_BasePanelAugmenter`. See there for more details.

    Parameters
    ----------
    param: float or (scipy) distribution, optional (default = 0)
        Standard deviation (std) of the gaussian noise. Given a distribution, an
        i.i.d. random variate will be drawn for each TS augmentation. If
        use_relative_fit is True, the actual std will be the product of the
        fitted statistical parameter and this value. If set to None, std will
        be zero (i.e. no Noise will be added).
    """
    def __init__(self, *args, **kwargs):
        self._param_desc = {"name_absolute": "std",
                            "name_relative": "scale_of_std",
                            "min": 0.0,
                            "max": np.nan_to_num(np.inf),
                            "default": 0.0,
                            "abs_inc_strength": True}
        super().__init__(*args, **kwargs)

    def _univariate_ser_aug_fun(self, X, rand_param_variate, stat_param):
        n = X.shape[0]  # length of the time series
        if self.use_relative_fit:
            return X + norm.rvs(0, rand_param_variate * stat_param, size=n)
        else:
            return X + norm.rvs(0, rand_param_variate, size=n)


class ReverseAugmenter(_BasePanelAugmenter):
    """Augmenter reversing the time series.

    This is a sub class inheriting its functionality from the superclass
    `_BasePanelAugmenter`. See there for more details.

    Parameters
    ----------
    param: any, optional (default = None)
        Ignored, as well as use_use_relative_fit.
    """
    def __init__(self, *args, **kwargs):
        self._is_fittable = False
        super().__init__(*args, **kwargs)

    def _univariate_ser_aug_fun(self, X, _, __):
        return X.loc[::-1].reset_index(drop=True, inplace=False)


class InvertAugmenter(_BasePanelAugmenter):
    """Augmenter inverting the time series (i.e. multiply each value with -1).

    This is a sub class inheriting its functionality from the superclass
    `_BasePanelAugmenter`. See there for more details.

    Parameters
    ----------
    param: any, optional (default = None)
        Ignored, as well as use_use_relative_fit.
    """
    def __init__(self, *args, **kwargs):
        self._is_fittable = False
        super().__init__(*args, **kwargs)

    def _univariate_ser_aug_fun(self, X, _, __):
        return X.mul(-1)


class ScaleAugmenter(_BasePanelAugmenter):
    """Augmenter scales (multiplies) the time series with the given parameter.

    This is a sub class inheriting its functionality from the superclass
    `_BasePanelAugmenter`. See there for more details.

    Parameters
    ----------
    param: float or (scipy) distribution, optional (default = 1.0)
        Scale factor. Given a distribution, an i.i.d. random variate will be
        drawn for each TS augmentation. If use_relative_fit is True,
        the actual scale factor will be the product of the fitted statistical
        parameter and this value.
    """
    def __init__(self, *args, **kwargs):
        self._param_desc = {"name_absolute": "scale",
                            "name_relative": "relative_scale",
                            "min": np.nan_to_num(-np.inf),
                            "max": np.nan_to_num(np.inf),
                            "default": 1.0,
                            "abs_inc_strength": True}
        super().__init__(*args, **kwargs)

    def _univariate_ser_aug_fun(self, X, rand_param_variate, stat_param):
        if self.use_relative_fit:
            return X.mul(rand_param_variate * stat_param)
        else:
            return X.mul(rand_param_variate)


class OffsetAugmenter(_BasePanelAugmenter):
    """Augmenter adds a scalar to the time series (shifting / offset).

    This is a sub class inheriting its functionality from the superclass
    `_BasePanelAugmenter`. See there for more details.

    Parameters
    ----------
    param: float or (scipy) distribution, optional (default = 0.0)
        Offset value. Given a distribution, an i.i.d. random variate will be
        drawn for each TS augmentation. If use_relative_fit is True,
        the actual offset value will be the product of the fitted statistical
        parameter and this value.
    """
    def __init__(self, *args, **kwargs):
        self._param_desc = {"name_absolute": "offset",
                            "name_relative": "relative_offset",
                            "min": np.nan_to_num(-np.inf),
                            "max": np.nan_to_num(np.inf),
                            "default": 0.0,
                            "abs_inc_strength": True}
        super().__init__(*args, **kwargs)

    def _univariate_ser_aug_fun(self, X, rand_param_variate, stat_param):
        if self.use_relative_fit:
            return X.add(rand_param_variate * stat_param)
        else:
            return X.add(rand_param_variate)


class DriftAugmenter(_BasePanelAugmenter):
    """Augmenter adds a random walk (drift) to time series.

    This is a sub class inheriting its functionality from the superclass
    `_BasePanelAugmenter`. See there for more details.

    Parameters
    ----------
    param: float or (scipy) distribution, optional (default = 0.0)
        Standard deviation (std) of the random walk. Given a distribution, an
        i.i.d. random variate will be drawn for each TS augmentation. If
        use_relative_fit is True, the actual std will be the product of the
        fitted statistical parameter and this value.
    """
    def __init__(self, *args, **kwargs):
        self._param_desc = {"name_absolute": "std_of_drift",
                            "name_relative": "relative_std_of_drift",
                            "min": 0.0,
                            "max": np.nan_to_num(np.inf),
                            "default": 0.0,
                            "abs_inc_strength": True}
        super().__init__(*args, **kwargs)

    def _univariate_ser_aug_fun(self, X, rand_param_variate, stat_param):
        n = X.shape[0]  # length of the time series
        if self.use_relative_fit:
            help = rand_param_variate * stat_param
        else:
            help = rand_param_variate
        return X.add(np.concatenate(
            ([0.0], np.cumsum(np.random.normal(0.0, help, n-1)))))


# implemented but not necessary for first PR:
"""
class ClipAugmenter(_BasePanelAugmenter):
    pass


class ClipTimeAugmenter(_BasePanelAugmenter):
    pass


class QuantizeAugmenter(_BasePanelAugmenter):
    pass


class JitterAugmenter(_BasePanelAugmenter):
    pass


class TSDropoutAugmenter(_BasePanelAugmenter):
    pass


class SampleDropoutAugmenter(_BasePanelAugmenter):
    pass


class DowntimeAugmenter(_BasePanelAugmenter):
    pass


class ResampleAugmenter(_BasePanelAugmenter):
    pass
"""


# Not implemented yet
"""
class FilterAugmenter(_BasePanelAugmenter):
    pass


class OutlierAugmenter(_BasePanelAugmenter):
    pass


class GaussianProcessAugmenter(_BasePanelAugmenter):
    pass


class ArbitraryAdditiveNoiseAugmenter(_BasePanelAugmenter):
    pass


class ArbitraryAugmenter(_BasePanelAugmenter):
    pass


class ShiftingTimeAugmenter(_BasePanelAugmenter):
    pass


class ScalingTimeAugmenter(_BasePanelAugmenter):
    pass


class ChopAugmenter(_BasePanelAugmenter):
    pass


class JigsawAugmenter(_BasePanelAugmenter):
    pass


class PoolingAugmenter(_BasePanelAugmenter):
    pass


class BlendAugmenter(_BasePanelAugmenter):
    pass
"""

# Paper content, delete later
"""
from sklearn.model_selection import cross_validate


def get_score_over_aug_weight(X, y, aug, est, scoring,
                              weight_support=None,
                              aug_strategy="train_test",
                              n_jobs=1,
                              n_cv=5):
    if weight_support is None:
        if aug._param_desc is None:  # in case the augmenter has no parameter
            weight_support = 1
        else:
            weight_support = (aug._param_desc["min"], aug._param_desc["min"])
    # calculate score for each weight through n_cv-fold CV
    results = []
    pipe = Pipeline([['augmenter', aug], ['estimator', est]])
    for weight in weight_support:
        pipe.steps
        results.append(cross_validate(pipe, X, y=y,
                                      scoring=scoring,
                                      cv=n_cv,
                                      n_jobs=n_jobs,
                                      verbose=0,
                                      pre_dispatch='2*n_jobs',
                                      return_train_score=True,
                                      return_estimator=False,
                                      error_score=np.nan))
"""
