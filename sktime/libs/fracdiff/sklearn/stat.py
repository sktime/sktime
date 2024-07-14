import numpy as np

#import statsmodels.tsa.stattools as stattools


class StatTester:
    """Carry out stationarity test of time-series.

    Parameters
    ----------
    - method : {"ADF"}, default "ADF"
        If "ADF":
            Augmented Dickey-Fuller unit-root test.

    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)

    Stationary time-series:
    >>> x = np.random.randn(100)
    >>> tester = StatTester(method='ADF')
    >>> tester.pvalue(x) < 0.01
    True
    >>> tester.is_stat(x)
    True

    Non-stationary time-series:
    >>> x = np.cumsum(x)
    >>> tester.pvalue(x) < 0.01
    False
    >>> tester.is_stat(x)
    False
    """

    def __init__(self, method: str = "ADF"):
        self.method = method

    @property
    def null_hypothesis(self) -> str:
        if self.method == "ADF":
            return "unit-root"
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def pvalue(self, x: np.ndarray) -> float:
        """
        Return p-value of the stationarity test.

        Parameters
        ----------
        - x : array, shape (n_samples,)
            Time-series to evaluate p-value.

        Returns
        -------
        pvalue : float
            p-value of the stationarity test.
        """
        if self.method == "ADF":
            _, pvalue, _, _, _, _ = stattools.adfuller(x)
            return pvalue
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def is_stat(self, x: np.ndarray, pvalue: float = 0.05) -> bool:
        """
        Return whether stationarity test implies stationarity.

        Parameters
        ----------
        - x : array, shape (n_samples,)
            Time-series to evaluate p-value.
        - pvalue : float, default 0.05
            Threshold of p-value.

        Note
        ----
        The name 'is_stat' may be misleading.
        Strictly speaking, `is_stat = True` implies that the null-hypothesis of
        the presence of a unit-root has been rejected (ADF test) or the null-hypothesis
        of the absence of a unit-root has not been rejected (KPSS test).

        Returns
        -------
        is_stat : bool
            True may imply the stationarity.
        """
        assert self.null_hypothesis == "unit-root", "not supported"

        return self.pvalue(x) < pvalue
