# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Naive aligners, aligning starts/ends."""

import numpy as np
import pandas as pd

from sktime.alignment.base import BaseAligner


class AlignerNaive(BaseAligner):
    """Naive strategies for multiple alignment.

    Naive strategies supported by this estimator are:

    * start: aligns starts (lowest index), does no squeezing/stretching
    * end: aligns ends (highest index), no squeezing/stretching
    * start-end: aligns starts and ends, stretches linearly and rounds

    Parameters
    ----------
    strategy: str, one of "start", "end", "start-end" (default)
        start: aligns starts (lowest index), does no squeezing/stretching
        end: aligns ends (highest index), no squeezing/stretching
        start-end: aligns starts and ends, stretches linearly and rounds
    """

    _tags = {
        "capability:multiple-alignment": True,  # can align more than two sequences?
    }

    def __init__(self, strategy="start-end"):
        self.strategy = strategy

        super().__init__()

        if strategy in ["start", "end"]:
            self.set_tags(**{"alignment_type": "partial"})

    def _fit(self, X, Z=None):
        """Fit alignment given series/sequences to align.

            core logic

        Parameters
        ----------
        X: list of pd.DataFrame (sequence) of length n - panel of series to align
        Z: pd.DataFrame with n rows, optional; metadata, row correspond to indices of X
        """
        strategy = self.strategy

        alignlen = np.max([len(Xi) for Xi in X])
        align = []

        for i, Xi in enumerate(X):
            col = "ind" + str(i)
            nXi = len(Xi)

            if strategy == "start":
                # indices are consecutive and padded at the end
                padl = alignlen - nXi
                vals = np.arange(nXi, dtype="object")
                vals = np.pad(vals, (0, padl), constant_values=np.nan)
                vals = pd.array(vals, dtype="Int64")

            elif strategy == "end":
                # indices are consecutive and padded at the start
                padl = alignlen - nXi
                vals = np.arange(nXi, dtype="object")
                vals = np.pad(vals, (padl, 0), constant_values=np.nan)
                vals = pd.array(vals, dtype="Int64")

            elif strategy == "start-end":
                # indices are linearly spaced to fill entire length and rounded
                vals = np.linspace(start=0, stop=nXi - 1, num=alignlen)
                vals = np.round(vals).astype("int64")
            else:
                raise ValueError(
                    "strategy must be one of 'start', 'end', or 'start-end'"
                )

            align = align + [pd.DataFrame({col: vals})]

        align = pd.concat(align, axis=1)

        self.align = align

        return self

    def _get_alignment(self):
        """Return alignment for sequences/series passed in fit (iloc indices).

        Behaviour: returns an alignment for sequences in X passed to fit
            model should be in fitted state, fitted model parameters read from self

        Returns
        -------
        pd.DataFrame in alignment format, with columns 'ind'+str(i) for integer i
            cols contain iloc index of X[i] mapped to alignment coordinate for alignment
        """
        return self.align

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for aligners.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        params0 = {}
        params1 = {"strategy": "start"}
        params2 = {"strategy": "end"}

        return [params0, params1, params2]
