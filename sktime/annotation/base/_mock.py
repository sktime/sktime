# -*- coding: utf-8 -*-
from sktime.annotation.base._base import BasePanelAnnotator, BaseStreamAnnotator


class MockPanelAnnotator(BasePanelAnnotator):
    """Mock stream annotator

    for testing purposes. This "silly" annotator annotates
        the n-th time point as positive, everything else negative

    Parameters
    ----------
    n : int, optional (default=1)
        the iloc index of the sequence(s) to annotate as positive
    """

    def __init__(self, n=1):

        self.n = n

        super().__init__()

    def _fit(self, X, Y=None, Z=None):

        return self

    def _predict(self, X, Y=None, Z=None):

        n = self.n

        Xt = X.copy()

        for Xi in Xt:
            Xi.iloc[:] = False
            if len(Xi) > n:
                Xi.iloc[n] = True

        return Xt


class MockStreamAnnotator(BaseStreamAnnotator):
    """Mock stream annotator

    for testing purposes. This "silly" annotator annotates
        the n-th time point as positive, everything else negative

    Parameters
    ----------
    n : int, optional (default=1)
        the iloc index of the sequence(s) to annotate as positive
    """

    def __init__(self, n=1):

        self.n = n

        super().__init__()

    def _fit(self, X, Y=None, Z=None):

        return self

    def _predict(self, X, Y=None):

        n = self.n

        Xt = X.copy()

        Xt.iloc[:] = False
        if len(Xt) > n:
            Xt.iloc[n] = True

        return Xt
