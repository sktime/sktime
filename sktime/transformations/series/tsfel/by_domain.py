# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements TSFEL transformer to get features by domain (statistical, temporal, spectral, fractal)."""

__author__ = ["Faakhir30"]


from sktime.transformations.base import BaseTransformer


class TSFELTransformer(BaseTransformer):
    """TSFEL transformer to get features by domain (statistical, temporal, spectral, fractal).

    Parameters
    ----------
    features_domain : str, optional (default=None)
        The domain of features to extract. Can be 'statistical', 'temporal',
        'spectral', 'fractal', or None. If None, all domains will be extracted.
    fs : float, optional (default=None)
        Sampling frequency of the input signal.
    window_size : int or None, optional (default=None)
        The size of the windows used to split the input signal, measured in
        the number of samples.
    overlap : float, optional (default=0)
        A value between 0 and 1 that defines the percentage of overlap between
        consecutive windows.
    verbose : int, optional (default=1)
        The verbosity mode. 0 means silent, and 1 means showing a progress bar.
    **kwargs : dict
        Additional keyword arguments passed to tsfel's feature extractor.
        See tsfel documentation for available options.
    """

    _tags = {
        "authors": ["Faakhir30"],
        "python_dependencies": ["tsfel"],
        # estimator type
        # --------------
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        #
        # behavioural tags: internal type
        # ----------------------------------
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "None",
        "capability:multivariate": True,
        "requires_y": False,
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        "fit_is_empty": True,
        "capability:inverse_transform": False,
        "capability:unequal_length": True,
        "capability:missing_values": False,
    }

    def __init__(
        self,
        features_domain=None,
        fs=None,
        window_size=None,
        overlap=0,
        verbose=1,
        **kwargs,
    ):
        self.features_domain = features_domain
        self.fs = fs
        self.window_size = window_size
        self.overlap = overlap
        self.verbose = verbose
        self.kwargs = kwargs

        super().__init__()

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series, Panel, or Hierarchical data, of mtype X_inner_mtype
            if X_inner_mtype is list, _transform must support all types in it
            Data to be transformed
        y : Series, Panel, or Hierarchical data, of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        X_transformed : Series of mtype pd.DataFrame
            transformed version of X containing extracted features
        """
        import tsfel

        # Get feature extraction config based on domain
        cfg_file = tsfel.get_features_by_domain(self.features_domain)

        # Extract features using tsfel
        features_df = (
            tsfel.feature_extraction.calc_features.time_series_features_extractor(
                cfg_file,
                X,
                fs=self.fs,
                window_size=self.window_size,
                overlap=self.overlap,
                verbose=self.verbose,
                **self.kwargs,
            )
        )

        return features_df
