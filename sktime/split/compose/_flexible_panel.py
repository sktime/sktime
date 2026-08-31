"""Splitter that makes a temporal splitter work with unequal-length panels."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from __future__ import annotations

__author__ = ["RobKuebler"]

__all__ = ["FlexiblePanelSplitter"]

import warnings

import numpy as np
import pandas as pd

from sktime.split.base import BaseSplitter
from sktime.utils.validation import is_int


class FlexiblePanelSplitter(BaseSplitter):
    """Wrap a temporal splitter to make it work with unequal-length panels.

    Folds (cutoffs) come from the longest series in the panel. For each
    fold, every other instance is included with as much of its own history
    as is available up to that cutoff, as long as it covers the full
    forecasting horizon and has enough training history. Instances that
    don't qualify for a fold are excluded from it, rather than truncated to
    the shortest instance the way ``base_cv`` would by default on a panel.

    Each split is itself a panel containing whichever instances qualify for
    that fold, so different folds can contain a different number of
    instances.

    This assumes all instances share a common, comparable time index, for
    example because they're all cut off at the same latest timestamp. If
    ``y`` is a single (non-panel) series, or every instance has equal length
    and shares the same time index, ``FlexiblePanelSplitter`` behaves
    exactly like ``base_cv``.

    Parameters
    ----------
    base_cv : sktime splitter, BaseSplitter descendant instance
        the underlying temporal splitter, e.g., ``SlidingWindowSplitter``
    min_length : int, optional, default=None
        minimum number of training observations an instance needs to be
        included in a fold. Must be > 0.
        If None (default), only instances with a full training window (as
        long as the longest instance's for that fold) are included.
        If set, instances with at least ``min_length`` observations before
        the cutoff are included too, with all the history they have.

    Notes
    -----
    Which instances qualify for a fold is decided from whatever panel is
    passed to ``split``/``split_series``. If this splitter is used as (part
    of) an explicit ``cv_X`` in ``evaluate``, wrap it with ``SameLocSplitter``
    around ``y`` (e.g. ``SameLocSplitter(TestPlusTrainSplitter(cv), y)``, the
    same composition ``evaluate`` uses by default) so ``X`` gets the same
    per-fold instance selection as ``y``, instead of ``X`` being split
    independently based on its own coverage. Without that, an ``X`` whose
    per-instance coverage differs from ``y`` (e.g. exogenous data starting
    later for some instance) silently yields folds where ``X`` and ``y``
    disagree on which instances are included.

    Examples
    --------
    >>> from sktime.split import SlidingWindowSplitter
    >>> from sktime.split.compose import FlexiblePanelSplitter
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> y = _make_hierarchical(hierarchy_levels=(2,), max_timepoints=10,
    ...     min_timepoints=7, random_state=42)
    >>> cv = FlexiblePanelSplitter(SlidingWindowSplitter(window_length=3, fh=1))
    >>> def train_ranges(train):
    ...     # per-series (start..end) of the train window, keyed by series id
    ...     idx = y.index[train]
    ...     times = idx.get_level_values(-1)
    ...     series = idx.droplevel(-1)
    ...     return ", ".join(
    ...         f"{s}: {times[series == s].min():%m-%d}"
    ...         f"..{times[series == s].max():%m-%d}"
    ...         for s in sorted(series.unique())
    ...     )
    >>> for i, (train, test) in enumerate(cv.split(y)):
    ...     print(f"fold {i}: {train_ranges(train)}")
    fold 0: h0_0: 01-02..01-04
    fold 1: h0_0: 01-03..01-05
    fold 2: h0_0: 01-04..01-06, h0_1: 01-04..01-06
    fold 3: h0_0: 01-05..01-07, h0_1: 01-05..01-07
    fold 4: h0_0: 01-06..01-08, h0_1: 01-06..01-08
    fold 5: h0_0: 01-07..01-09, h0_1: 01-07..01-09

    Every included series gets the same 3-day calendar window. ``base_cv``
    alone positions each series' window relative to its own local index
    instead, so windows of equal length land on different calendar dates
    per series (``h0_1`` starts 2 days later than ``h0_0``, and that offset
    carries through every fold):

    >>> for i, (train, test) in enumerate(cv.base_cv.split(y)):
    ...     print(f"fold {i}: {train_ranges(train)}")
    fold 0: h0_0: 01-02..01-04, h0_1: 01-04..01-06
    fold 1: h0_0: 01-03..01-05, h0_1: 01-05..01-07
    fold 2: h0_0: 01-04..01-06, h0_1: 01-06..01-08
    fold 3: h0_0: 01-05..01-07, h0_1: 01-07..01-09
    """

    _tags = {
        "split_hierarchical": True,
        # FlexiblePanelSplitter handles the hierarchical/panel case itself
    }

    def __init__(self, base_cv: BaseSplitter, min_length: int | None = None) -> None:
        self.base_cv = base_cv
        self.min_length = min_length

        super().__init__()

        self.window_length = base_cv.window_length

        tags_to_clone = ["split_series_uses"]
        self.clone_tags(base_cv, tags_to_clone)

        if min_length is not None:
            if min_length <= 0:
                raise ValueError(f"min_length must be > 0, but found {min_length}")

            window_length = base_cv.window_length
            if is_int(window_length) and min_length > window_length:
                warnings.warn(
                    f"min_length ({min_length}) is greater than base_cv's "
                    f"window_length ({window_length}), so it has no relaxing "
                    f"effect and every instance will be required to have a "
                    f"full training window.",
                    UserWarning,
                    stacklevel=2,
                )

    def _fh(self):
        """Forecasting horizon, in integer resp array of integer, relative to cutoff.

        Private method called by property ``fh``, overridden here to inherit
        the forecasting horizon from ``base_cv`` directly, since the splits
        are the same.
        """
        return self.base_cv.fh

    def _split(self, y: pd.Index):
        """Get iloc references to train/test splits of ``y``.

        private _split containing the core logic, called from split

        Parameters
        ----------
        y : pd.Index
            Index of time series to split

        Yields
        ------
        train : 1D np.ndarray of dtype int
            Training window indices, iloc references to training indices in y
        test : 1D np.ndarray of dtype int
            Test window indices, iloc references to test indices in y
        """
        if not isinstance(y, pd.MultiIndex):
            yield from self.base_cv.split(y)
            return

        level_cols = list(range(y.nlevels - 1))
        iloc_by_time = pd.Series(range(len(y)), index=y)

        # squeeze to a scalar level when there's only one, to avoid a pandas
        # FutureWarning about list-like level of length 1 changing group keys
        groupby_level = level_cols[0] if len(level_cols) == 1 else level_cols

        # per instance: its own time values and their global iloc positions in y
        instances = []
        for _, group in iloc_by_time.groupby(level=groupby_level):
            instances.append((group.index.get_level_values(-1), group.to_numpy()))
        longest_time_index, _ = max(instances, key=lambda inst: len(inst[0]))

        for train_pos, test_pos in self.base_cv.split(longest_time_index):
            train_vals = longest_time_index[train_pos]
            test_vals = longest_time_index[test_pos]
            cutoff = train_vals.max()
            full_train_len = len(train_vals)

            train_parts = []
            test_parts = []
            for time_index, iloc in instances:
                test_mask = time_index.isin(test_vals)
                if test_mask.sum() != len(test_vals):
                    continue  # does not cover the full forecasting horizon

                train_mask = time_index.isin(train_vals)
                if train_mask.sum() != full_train_len:
                    if self.min_length is None:
                        continue  # no full training window, no relaxation
                    train_mask = time_index <= cutoff
                    if train_mask.sum() < self.min_length:
                        continue  # not enough training history

                train_parts.append(iloc[train_mask])
                test_parts.append(iloc[test_mask])

            yield np.concatenate(train_parts), np.concatenate(test_parts)

    @classmethod
    def get_test_params(cls, parameter_set: str = "default") -> list[dict]:
        """Return testing parameter settings for the splitter.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        from sktime.split import ExpandingWindowSplitter, SlidingWindowSplitter

        params = [
            {"base_cv": SlidingWindowSplitter(window_length=3, fh=1)},
            {
                "base_cv": SlidingWindowSplitter(window_length=3, fh=1),
                "min_length": 1,
            },
            {"base_cv": ExpandingWindowSplitter(fh=[1, 2], initial_window=3)},
        ]

        return params
