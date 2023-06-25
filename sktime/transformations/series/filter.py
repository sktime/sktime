"""Frequency filters."""

__author__ = ["sveameyer13"]
__all__ = ["Filter"]

import numpy as np

from sktime.transformations.base import BaseTransformer


class Filter(BaseTransformer):
    """Transformer that filters Series data.

    FIR, IIR, band pass filters.

    Provides a simple wrapper around ``mne.filter.filter_data``.

    Parameters
    ----------
    sfreq: int or float
        sampling frequency of the recorded data in Hz
    l_freq: float or None
        For FIR filters, the lower pass-band edge;
        for IIR filters, the lower cutoff frequency.
        If None the data are only low-passed.
    h_freq: float or None
        For FIR filters, the upper pass-band edge;
        for IIR filters, the upper cutoff frequency.
        If None the data are only high-passed.
    filter_kwargs: dict or None
        Additional parameters passed on to ``mne.filter.filter_data``.
        See ``mne.filter.filter_data``
        documentation for a detailed description of all options.
    """

    # default tag values for "Series-to-Series"
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["np.ndarray", "numpy3D"],
        "y_inner_mtype": "None",  # which mtypes do _fit/_predict support for X?
        "fit_is_empty": True,  # is fit empty and can be skipped? Yes = True
        "python_dependencies": "mne",
    }

    def __init__(
        self,
        sfreq,
        l_freq=None,
        h_freq=None,
        filter_kwargs=None,
    ):
        self.sfreq = sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.filter_kwargs = filter_kwargs
        if not (
            isinstance(sfreq, (int, float))
            & isinstance(l_freq, (int, float, type(None)))
            & isinstance(h_freq, (int, float, type(None)))
        ):
            raise TypeError
        elif (l_freq is not None) & (h_freq is not None):
            if not ((l_freq > 0) & (h_freq > 0)):
                raise ValueError("Negative values not supported")
            if l_freq > h_freq:
                raise ValueError("High frequency must be higher" " than low frequency")
        super().__init__()

    def _transform(self, X, y=None) -> np.ndarray:
        """Transform data.

        Returns a transformed version of X.

        Parameters
        ----------
        X : 2D or 3D numpy array, sktime format np.darray or numpy3D
        y : not used, passed only for interface conformance

        Returns
        -------
        Xt : 2D or 3D numpy array, same dimension as X
            Transformed time series.
        """
        from mne import filter

        # np.darray needs to be [anything, ..., time]
        # so 3D is ok, but we need to flip in 2D case
        if X.ndim == 2:
            X = X.transpose()

        sfreq = self.sfreq
        l_freq = self.l_freq
        h_freq = self.h_freq

        kwargs = self.filter_kwargs
        if kwargs is None:
            kwargs = {}

        # X is now of shape channels * timepoints or instance * channels * timepoints
        Xt = filter.filter_data(X, sfreq=sfreq, l_freq=l_freq, h_freq=h_freq, **kwargs)

        # Series, 2D: transpose back to have sktime shape again (timepoints*channels)
        if X.ndim == 2:
            Xt = Xt.transpose()
        # Panel, 3D: Xt comes out as 2D from filter_data and needs to be reshaped
        else:
            Xt = Xt.reshape(X.shape)
        return Xt

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return {"sfreq": 3}
