# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Naive aligners, aligning tops/bottoms."""

import numpy as np
import pandas as pd

from sktime.alignment.base import BaseAligner


class AlignerNaive(BaseAligner):
    """Naive strategies for multiple alignment.

    Parameters
    ----------
    strategy: str, one of "top", "bottom", "top-bottom" (default)
        top: aligns tops (lowest index), does no squeezing/stretching
        bottom: aligns bottoms (highest index), no squeezing/stretching
        top-bottom: aligns tops and bottoms, stretches linearly and rounds
    """

    def __init__(self, strategy="top-bottom"):

        self.strategy = strategy

    def _fit(self, X, Z=None):
        """Fit alignment given series/sequences to align.

            core logic

        Parameters
        ----------
        X: list of pd.DataFrame (sequence) of length n - panel of series to align
        Z: pd.DataFrame with n rows, optional; metadata, row correspond to indices of X
        """
        self.X = X

        strategy = self.strategy

        alignlen = np.max([len(Xi) for Xi in X])
        align = [pd.DataFrame({"ind_align": np.arange(alignlen)})]

        for i, Xi in enumerate(X):

            col = "ind" + str(i)
            nXi = len(Xi)

            if strategy == "top":
                # indices are consecutive and padded at the end
                padl = alignlen - nXi
                vals = np.arange(nXi, dtype="object")
                vals = np.pad(vals, (0, padl), constant_values=np.nan)
                vals = pd.array(vals, dtype="Int64")

            elif strategy == "bottom":
                # indices are consecutive and padded at the start
                padl = alignlen - nXi
                vals = np.arange(nXi, dtype="object")
                vals = np.pad(vals, (padl, 0), constant_values=np.nan)
                vals = pd.array(vals, dtype="Int64")

            elif strategy == "top-bottom":
                # indices are linearly spaced to fill entire length and rounded
                vals = np.linspace(start=0, stop=nXi-1, num=alignlen)
                vals = np.round(vals).astype("int64")
            else:
                raise ValueError(
                    "strategy must be one of 'top', 'bottom', or 'top-bottom'"
                )

            align = align + [pd.DataFrame({col: vals})]

        align = pd.concat(align, axis=1)

        self.align = align

        return self

    def _get_alignment(self):
        """Return alignment for sequences in X passed to fit.

        Returns
        -------
        pd.DataFrame in alignment format, as follows
        columns:
            ind_align: float, integer, or index, alignment coordinate
            multiple columns indexed by string 'ind'+str(i) for integer i:
                iloc index of X[i] mapped to alignment coordinate for alignment
        """
        return self.align
