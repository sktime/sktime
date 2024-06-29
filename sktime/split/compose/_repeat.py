"""Splitter obtained from repeating another splitter."""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]


from sktime.split.base import BaseSplitter


class Repeat(BaseSplitter):
    """Add repetitions to a splitter, element-wise or sequence-wise.

    Element-wise means: if the original splitter splits a series s into s1, s2, s3,
    then a 2-times repeat splits s into s1, s1, s2, s2, s3, s3.
    Sequence-wise means: if the original splitter splits a series s into s1, s2, s3,
    then a 2-times repeat splits s into s1, s2, s3, s1, s2, s3.

    This splitter also allows to control whether repetitions are exact
    or independent pseuo-random, for stochastic splitters.

    Parameters
    ----------
    splitter : sktime splitter object, BaseSplitter descendant instance
        splitter to repeat
    times : int, default=1
        number of times to repeat the splitter
    mode : str, one of "entry" and "sequence", default="entry"
        mode of repetition
        "entry" repeats each entry of the split ``times`` times
        "sequence" repeats the entire sequence of splits ``times`` times
    random_repeat : bool, default=False
        whether repetitions should be exact or independent pseudo-random
        If False, repetitions are exact (default)
        If True, repetitions are random, ``splitter`` is cloned for each repetition.
        Note: if a random seed is set in ``splitter``, the effect is the same
        as setting ``random_repeat`` to False, even if ``random_repeat`` is True.
    """

    _tags = {
        "split_hierarchical": True,
        "split_series_uses": "iloc",
    }

    def __init__(self, splitter, times=1, mode="entry", random_repeat=False):
        self.splitter = splitter
        self.times = times
        self.mode = mode
        self.random_repeat = random_repeat

        super().__init__()

        ALLOWED_MODES = ["entry", "sequence"]
        if mode not in ALLOWED_MODES:
            raise ValueError(
                f"Mode in Repeat splitter should be one of {ALLOWED_MODES}, "
                f"but found {mode}"
            )

        tags_to_clone = ["split_series_uses"]

        self.clone_tags(splitter, tags_to_clone)

    def _split(self, y):
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
        yield from self._repeat(y, method="split")

    def _split_loc(self, y):
        """Get loc references to train/test splits of ``y``.

        private _split containing the core logic, called from split_loc

        Default implements using split and y.index to look up the loc indices.
        Can be overridden for faster implementation.

        Parameters
        ----------
        y : pd.Index
            index of time series to split

        Yields
        ------
        train : pd.Index
            Training window indices, loc references to training indices in y
        test : pd.Index
            Test window indices, loc references to test indices in y
        """
        yield from self._repeat(y, method="split_loc")

    def _repeat(self, y, method="split"):
        """Repeat the splitter.

        Parameters
        ----------
        method : str, one of "split" and "split_loc", default="split"
            method to repeat

        Yields
        ------
        train : 1D np.ndarray of dtype int or pd.Index
            Training window indices, iloc or loc references to training indices in y
        test : 1D np.ndarray of dtype int or pd.Index
            Test window indices, iloc or loc references to test indices in y
        """
        random_repeat = self.random_repeat

        if random_repeat:
            spl_clones = [self.splitter.clone() for _ in range(self.times)]
            spl_gens = [getattr(spl, method)(y) for spl in spl_clones]
        else:
            one_clone = self.splitter.clone()
            one_gen = getattr(one_clone, method)(y)

        if self.mode == "entry" and not random_repeat:
            for train, test in one_gen:
                for _ in range(self.times):
                    yield train, test
        elif self.mode == "entry" and random_repeat:
            for _ in range(self.splitter.get_n_splits(y)):
                for spl_gen in spl_gens:
                    yield next(spl_gen)
        elif self.mode == "sequence" and not random_repeat:
            all_res = list(one_gen)
            for _ in range(self.times):
                for train, test in all_res:
                    yield train, test
        elif self.mode == "sequence" and random_repeat:
            for spl_gen in spl_gens:
                yield from spl_gen

    def get_n_splits(self, y) -> int:
        """Return the number of splits.

        Parameters
        ----------
        y : pd.Series or pd.Index, optional (default=None)
            Time series to split

        Returns
        -------
        n_splits : int
            The number of splits.
        """
        return self.splitter.get_n_splits(y) * self.times

    @classmethod
    def get_test_params(cls, parameter_set="default"):
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
        from sktime.split import ExpandingWindowSplitter

        modes = ["entry", "sequence"]
        repeats = [True, False]
        timess = [2, 3, 7]

        params = []
        for mode in modes:
            for random_repeat in repeats:
                for times in timess:
                    params.append(
                        {
                            "mode": mode,
                            "random_repeat": random_repeat,
                            "times": times,
                            "splitter": ExpandingWindowSplitter(),
                        }
                    )

        return params
