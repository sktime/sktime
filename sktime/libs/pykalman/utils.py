# New BSD License
#
# Copyright (c) 2007 - 2012 The scikit-learn developers.
# All rights reserved.
#
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#   a. Redistributions of source code must retain the above copyright notice,
#      this list of conditions and the following disclaimer.
#   b. Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#   c. Neither the name of the Scikit-learn Developers  nor the names of
#      its contributors may be used to endorse or promote products
#      derived from this software without specific prior written
#      permission.
#
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
# OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
# DAMAGE.
"""Utility functions taken from scikit-learn."""

import inspect

import numpy as np
from scipy import linalg


def array1d(X, dtype=None, order=None):
    """Return at least 1-d array with data from X."""
    return np.asarray(np.atleast_1d(X), dtype=dtype, order=order)


def array2d(X, dtype=None, order=None):
    """Return at least 2-d array with data from X."""
    return np.asarray(np.atleast_2d(X), dtype=dtype, order=order)


def log_multivariate_normal_density(X, means, covars, min_covar=1.0e-7):
    """Log probability for full covariance matrices."""
    n_samples, n_dim = X.shape
    nmix = len(means)
    log_prob = np.empty((n_samples, nmix))
    for c, (mu, cv) in enumerate(zip(means, covars)):
        try:
            cv_chol = linalg.cholesky(cv, lower=True)
        except linalg.LinAlgError:
            # The model is most probably stuck in a component with too
            # few observations, we need to reinitialize this components
            cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim), lower=True)
        cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
        cv_sol = np.linalg.solve(cv_chol, (X - mu).T).T
        log_prob[:, c] = -0.5 * (
            np.sum(cv_sol**2, axis=1) + n_dim * np.log(2 * np.pi) + cv_log_det
        )

    return log_prob


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.

    If seed is None, return the RandomState singleton used by np.random.
    If seed is an int, return a new RandomState instance seeded with seed.
    If seed is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (int, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "{0} cannot be used to seed a numpy.random.RandomState" + " instance"
    ).format(seed)


class Bunch(dict):
    """Container object for datasets.

    Dictionary-like object that exposes its keys as attributes.
    """

    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self


def get_params(obj):
    """Get names and values of all parameters in `obj`'s __init__."""
    try:
        # get names of every variable in the argument
        args = inspect.getfullargspec(obj.__init__)[0]
        args.pop(0)  # remove "self"

        # get values for each of the above in the object
        argdict = {arg: obj.__getattribute__(arg) for arg in args}
        return argdict
    except Exception:
        raise ValueError("object has no __init__ method")


def preprocess_arguments(argsets, converters):
    """Convert and collect arguments in order of priority.

    Parameters
    ----------
    argsets : [{argname: argval}]
        a list of argument sets, each with lower levels of priority
    converters : {argname: function}
        conversion functions for each argument

    Returns
    -------
    result : {argname: argval}
        processed arguments
    """
    result = {}
    for argset in argsets:
        for argname, argval in argset.items():
            # check that this argument is necessary
            if argname not in converters:
                raise ValueError(f"Unrecognized argument: {argname}")

            # potentially use this argument
            if argname not in result and argval is not None:
                # convert to right type
                argval = converters[argname](argval)

                # save
                result[argname] = argval

    # check that all arguments are covered
    if not len(converters.keys()) == len(result.keys()):
        missing = set(converters.keys()) - set(result.keys())
        s = f"The following arguments are missing: {list(missing)}"
        raise ValueError(s)

    return result
