"""Transfer Entropy calculation matrix between time series """

__authors__ = ["Spinachboul"]

__all__ = ["TransferEntropy"]
from sktime.dists_kernels.base import BasePairwiseTransformerPanel


class TransferEntropy(BasePairwiseTransformerPanel):
    """Calculate Transfer Entropy between pairs of time series.
    
    Transfer Entropy (TE) is an information-theoretic measure that quantifies the directed 
    transfer of information between time series. It detects both linear and non-linear 
    dependencies by measuring the reduction in uncertainty about a target time series 
    when considering the history of a source time series, beyond what can be explained 
    by the target's own history alone.
    
    Mathematically defined as:
    TE(X→Y) = H(Y(t)|Y(t-1:t-k)) - H(Y(t)|Y(t-1:t-k), X(t-1:t-l))
    
    where H represents Shannon Entropy.
    
    Parameters
    ----------
    lag_target : int, default=1
        The number of past values of the target time series to consider.
        Controls k in the formula above.
    
    lag_source : int, default=1
        The number of past values of the source time series to consider.
        Controls l in the formula above.
    
    estimator : str, default="binning"
        Method for probability estimation:
        - "binning": Equal-width binning of continuous values
        - "knn": K-nearest neighbors density estimation
        - "kernel": Kernel density estimation
    
    n_bins : int, default=10
        Number of bins used when estimator="binning".
        Higher values increase precision but require more data.
    
    k_neighbors : int, default=5
        Number of neighbors when estimator="knn".
    
    kernel_width : float, default=0.1
        Bandwidth parameter when estimator="kernel".
    
    significance_test : bool, default=False
        Whether to perform statistical significance testing.
    
    n_surrogates : int, default=100
        Number of surrogate time series to generate for significance testing.
        Only used when significance_test=True.
    
    significance_level : float, default=0.05
        P-value threshold for significance testing.
    
    normalize : bool, default=False
        If True, normalize TE values by the entropy of the target.
    
    missing_values : str, default="error"
        Strategy for handling missing values:
        - "error": Raise an error if missing values are found
        - "drop": Remove time points with missing values
        - "impute": Apply simple mean imputation
    
    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.distances import TransferEntropy
    >>> import pandas as pd
    >>> import numpy as np
    >>> X = pd.DataFrame({
    ...     "ts1": np.random.rand(100),
    ...     "ts2": np.random.rand(100),
    ... })
    >>> te = TransferEntropy(lag_source=2, lag_target=1, estimator="binning")
    >>> te_matrix = te.fit_transform(X)
    >>> print(te_matrix)
    
    Notes
    -----
    - Higher TE values indicate stronger directed information flow from source to target.
    - TE = 0 implies conditional independence (source doesn't improve prediction of target).
    - The measure is asymmetric: TE(X→Y) ≠ TE(Y→X).
    - Requires sufficient data points for reliable estimation (rule of thumb: at least 10^(lag_source + lag_target) points).
    - Computational complexity increases with lags and estimation precision.
    
    References
    ----------
    .. [1] Schreiber, T. (2000). Measuring information transfer. Physical Review Letters, 85(2), 461.
    .. [2] Vicente, R., Wibral, M., Lindner, M., & Pipa, G. (2011). Transfer entropy—a model-free 
           measure of effective connectivity for the neurosciences. Journal of Computational Neuroscience, 30, 45-67.
    """
    
    _tags = {
        "authors": ["Spinachboul"],
        "X_inner_mtype": "pd.DataFrame",
        "capability:missing_values": True,
        "capability:multivariate": True,
        "capability:unequal_length": False,
        
    }
    
    def __init__(
        self,
        lag_target=1,
        lag_source=1,
        estimator="binning",
        n_bins=10,
        k_neighbors=5,
        kernel_width=0.1,
        significance_test=False,
        n_surrogates=100,
        significance_level=0.05,
        normalize=False,
        missing_values="error",
    ):
        self.lag_target = lag_target
        self.lag_source = lag_source
        self.estimator = estimator
        self.n_bins = n_bins
        self.k_neighbors = k_neighbors
        self.kernel_width = kernel_width
        self.significance_test = significance_test
        self.n_surrogates = n_surrogates
        self.significance_level = significance_level
        self.normalize = normalize
        self.missing_values = missing_values
        super().__init__()
    
    def _transform(self, X, X2=None):
        """Calculate pairwise Transfer Entropy between all time series in X.
        
        Parameters
        ----------
        X : pd.DataFrame or nested pd.Series
            Panel data with multiple time series
        y : pd.Series, optional (default=None)
            Ignored, exists for API consistency
            
        Returns
        -------
        pd.DataFrame
            Matrix of pairwise Transfer Entropy values
        """
        import pandas as pd
        import numpy as np
        from sklearn.exceptions import NotFittedError
        
        # Check if X is a valid DataFrame
        # this can also be checked with the tag
        # if not isinstance(X, pd.DataFrame):
        #     raise TypeError("Input X must be a pandas DataFrame")
        
        # Handle missing values
        # we can use this manual calculation if the tag is False

        # convert explicitly to DataFrame
        if isinstance(X, list):
            X = pd.DataFrame(X)
        
        # Handle the missing values according to the specified strategy
        if self.missing_values == "error" and X.isnull().values.any():
            raise ValueError("Missing values found in input data")
        elif self.missing_values == "drop":
            X = X.dropna()
        elif self.missing_values == "impute":
            X = X.fillna(X.mean())
        
        # Get all column names for the DataFrame
        col_names = X.columns
        n_series = len(col_names)
        
        # Initialize an empty DataFrame to store pairwise Transfer Entropy values
        te_matrix = pd.DataFrame(
            np.zeros((n_series, n_series)),
            index=col_names,
            columns=col_names
        )
        
        # Compute Transfer Entropy for each pair of time series
        for i, source_name in enumerate(col_names):
            for j, target_name in enumerate(col_names):
                if i != j:  # Skip diagonal elements (self-entropy)
                    source = X[source_name].values
                    target = X[target_name].values
                    
                    # Calculate Transfer Entropy
                    te_value = self._get_transfer_entropy(source, target)
                    
                    # Perform significance testing if enabled
                    if self.significance_test:
                        p_value = self._significance_test(source, target)
                        if p_value > self.significance_level:
                            te_value = 0  # Zero out non-significant values
                    
                    te_matrix.iloc[i, j] = te_value
        
        return te_matrix
    
    def _get_transfer_entropy(self, source, target):
        """Calculate Transfer Entropy from source to target time series.
        
        Parameters
        ----------
        source : pd.Series or np.array
            Source time series
        target : pd.Series or np.array
            Target time series
            
        Returns
        -------
        float
            Transfer Entropy value from source to target
        """
        import numpy as np
        from scipy import stats
        from sklearn.neighbors import KernelDensity, NearestNeighbors
        
        # Ensure inputs are numpy arrays
        source = np.asarray(source)
        target = np.asarray(target)
        
        # Check if we have enough data points
        min_data_points = max(self.lag_source, self.lag_target) + 1
        if len(source) < min_data_points or len(target) < min_data_points:
            raise ValueError(
                f"Time series too short. Need at least {min_data_points} data points."
            )
        
        # Create lagged versions of the time series
        def create_lagged_vectors(series, lag):
            vectors = []
            for i in range(lag):
                vectors.append(series[lag - i - 1:-i - 1] if i < lag - 1 else series[lag - i - 1:])
            return np.column_stack(vectors) if vectors else np.empty((len(series) - lag + 1, 0))
        
        # Length of usable data after applying lags
        n = len(target) - max(self.lag_source, self.lag_target)
        
        # Create lagged variables for target and source
        y_target = target[max(self.lag_source, self.lag_target):]  # y_t
        y_target_hist = create_lagged_vectors(target, self.lag_target)[max(0, self.lag_source - self.lag_target):, :]  # y_(t-1:t-k)
        x_source_hist = create_lagged_vectors(source, self.lag_source)[max(0, self.lag_target - self.lag_source):, :]  # x_(t-1:t-l)
        
        # Calculate conditional entropies based on the chosen estimator
        if self.estimator == "binning":
            # Use binning for probability estimation
            return self._calculate_te_binning(y_target, y_target_hist, x_source_hist)
        elif self.estimator == "knn":
            # Use k-nearest neighbors for probability estimation
            return self._calculate_te_knn(y_target, y_target_hist, x_source_hist)
        elif self.estimator == "kernel":
            # Use kernel density estimation for probability estimation
            return self._calculate_te_kernel(y_target, y_target_hist, x_source_hist)
        else:
            raise ValueError(f"Unknown estimator: {self.estimator}")
    
    def _calculate_te_binning(self, y_target, y_target_hist, x_source_hist):
        """Calculate Transfer Entropy using binning method.
        
        Parameters
        ----------
        y_target : np.array
            Target time series values at time t
        y_target_hist : np.array
            Target time series history values
        x_source_hist : np.array
            Source time series history values
            
        Returns
        -------
        float
            Transfer Entropy value
        """
        import numpy as np
        from scipy import stats
        
        # Function to discretize continuous values into bins
        def discretize(data, n_bins):
            # For 1D data
            if data.ndim == 1:
                return np.digitize(data, np.linspace(min(data), max(data), n_bins + 1)[:-1])
            # For multi-dimensional data
            else:
                result = np.zeros_like(data, dtype=int)
                for i in range(data.shape[1]):
                    result[:, i] = np.digitize(
                        data[:, i], 
                        np.linspace(min(data[:, i]), max(data[:, i]), n_bins + 1)[:-1]
                    )
                return result
        
        # Discretize the data
        y_t_disc = discretize(y_target, self.n_bins)
        y_hist_disc = discretize(y_target_hist, self.n_bins)
        x_hist_disc = discretize(x_source_hist, self.n_bins)
        
        # For each unique discretized value, calculate probabilities
        def get_entropy_from_counts(counts):
            probs = counts / np.sum(counts)
            return -np.sum(probs * np.log2(probs + np.finfo(float).eps))
        
        # Calculate H(Y_t | Y_hist)
        joint_counts_y = np.zeros((self.n_bins, np.prod([self.n_bins] * y_hist_disc.shape[1])))
        
        # Create a unique index for each combination of y_hist values
        y_hist_idx = np.ravel_multi_index(
            [y_hist_disc[:, i] - 1 for i in range(y_hist_disc.shape[1])], 
            [self.n_bins] * y_hist_disc.shape[1]
        )
        
        # Count occurrences of each (y_t, y_hist) combination
        for i in range(len(y_t_disc)):
            joint_counts_y[y_t_disc[i] - 1, y_hist_idx[i]] += 1
        
        # Calculate H(Y_t | Y_hist)
        h_y_given_y_hist = 0
        for y_hist_val in range(joint_counts_y.shape[1]):
            if np.sum(joint_counts_y[:, y_hist_val]) > 0:
                p_y_hist = np.sum(joint_counts_y[:, y_hist_val]) / np.sum(joint_counts_y)
                h_y_given_y_hist_val = get_entropy_from_counts(joint_counts_y[:, y_hist_val])
                h_y_given_y_hist += p_y_hist * h_y_given_y_hist_val
        
        # Calculate H(Y_t | Y_hist, X_hist)
        # Create a unique index for the joint state of y_hist and x_hist
        joint_dim = y_hist_disc.shape[1] + x_hist_disc.shape[1]
        joint_counts_yx = np.zeros((self.n_bins, np.prod([self.n_bins] * joint_dim)))
        
        # Combine indices for y_hist and x_hist
        yx_hist_parts = [y_hist_disc[:, i] - 1 for i in range(y_hist_disc.shape[1])]
        yx_hist_parts.extend([x_hist_disc[:, i] - 1 for i in range(x_hist_disc.shape[1])])
        
        yx_hist_idx = np.ravel_multi_index(
            yx_hist_parts,
            [self.n_bins] * joint_dim
        )
        
        # Count occurrences of each (y_t, y_hist, x_hist) combination
        for i in range(len(y_t_disc)):
            joint_counts_yx[y_t_disc[i] - 1, yx_hist_idx[i]] += 1
        
        # Calculate H(Y_t | Y_hist, X_hist)
        h_y_given_yx_hist = 0
        for yx_hist_val in range(joint_counts_yx.shape[1]):
            if np.sum(joint_counts_yx[:, yx_hist_val]) > 0:
                p_yx_hist = np.sum(joint_counts_yx[:, yx_hist_val]) / np.sum(joint_counts_yx)
                h_y_given_yx_hist_val = get_entropy_from_counts(joint_counts_yx[:, yx_hist_val])
                h_y_given_yx_hist += p_yx_hist * h_y_given_yx_hist_val
        
        # Calculate Transfer Entropy: H(Y_t | Y_hist) - H(Y_t | Y_hist, X_hist)
        te = h_y_given_y_hist - h_y_given_yx_hist
        
        # Normalize if requested
        if self.normalize:
            # Calculate H(Y_t) for normalization
            y_t_counts = np.bincount(y_t_disc, minlength=self.n_bins + 1)[1:]
            h_y_t = get_entropy_from_counts(y_t_counts)
            
            if h_y_t > 0:
                te /= h_y_t
        
        return max(0, te)  # Transfer entropy should not be negative
    
    def _calculate_te_knn(self, y_target, y_target_hist, x_source_hist):
        """Calculate Transfer Entropy using k-nearest neighbors method.
        
        Parameters
        ----------
        y_target : np.array
            Target time series values at time t
        y_target_hist : np.array
            Target time series history values
        x_source_hist : np.array
            Source time series history values
            
        Returns
        -------
        float
            Transfer Entropy value
        """
        import numpy as np
        from scipy.special import digamma
        from sklearn.neighbors import NearestNeighbors
        
        # Implementation of the Kraskov-Stögbauer-Grassberger (KSG) estimator
        # Based on: Kraskov, A., Stögbauer, H., & Grassberger, P. (2004). 
        # "Estimating mutual information." Physical Review E, 69(6), 066138.
        
        k = self.k_neighbors
        
        # 1. Create joint spaces
        space_y_yhist = np.column_stack([y_target.reshape(-1, 1), y_target_hist])
        space_yhist = y_target_hist
        space_yhist_xhist = np.column_stack([y_target_hist, x_source_hist])
        
        # 2. Find k-nearest neighbors in joint space
        nn_y_yhist = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
        nn_y_yhist.fit(space_y_yhist)
        dists_y_yhist, _ = nn_y_yhist.kneighbors(space_y_yhist)
        
        # 3. Calculate the number of points within eps distance in subspaces
        eps = dists_y_yhist[:, k].reshape(-1, 1)  # use Chebyshev distance of k-th neighbor
        
        nn_yhist = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
        nn_yhist.fit(space_yhist)
        
        nn_yhist_xhist = NearestNeighbors(n_neighbors=k+1, metric='chebyshev')
        nn_yhist_xhist.fit(space_yhist_xhist)
        
        # Count points within eps
        n_y = np.array([np.sum(np.abs(y_target - y_target[i]) <= eps[i]) for i in range(len(y_target))])
        n_yhist = np.array([np.sum(np.all(np.abs(space_yhist - space_yhist[i]) <= eps[i], axis=1)) for i in range(len(space_yhist))])
        n_yhist_xhist = np.array([np.sum(np.all(np.abs(space_yhist_xhist - space_yhist_xhist[i]) <= eps[i], axis=1)) for i in range(len(space_yhist_xhist))])
        
        # Apply digamma function and calculate transfer entropy
        te = digamma(k) + np.mean(digamma(n_yhist) - digamma(n_yhist_xhist) - digamma(n_y))
        
        # Normalize if requested
        if self.normalize:
            h_y = digamma(len(y_target)) - np.mean(digamma(n_y))
            if h_y > 0:
                te /= h_y
        
        return max(0, te)  # Transfer entropy should not be negative
    
    def _calculate_te_kernel(self, y_target, y_target_hist, x_source_hist):
        """Calculate Transfer Entropy using kernel density estimation.
        
        Parameters
        ----------
        y_target : np.array
            Target time series values at time t
        y_target_hist : np.array
            Target time series history values
        x_source_hist : np.array
            Source time series history values
            
        Returns
        -------
        float
            Transfer Entropy value
        """
        import numpy as np
        from sklearn.neighbors import KernelDensity
        
        # Define a function to estimate entropy from samples and their log probabilities
        def entropy_from_logprobs(logprobs):
            return -np.mean(logprobs)
        
        # Set bandwidth for KDE
        bandwidth = self.kernel_width
        
        # Create joint spaces for different distributions
        joint_y_yhist = np.column_stack([y_target.reshape(-1, 1), y_target_hist])
        yhist = y_target_hist
        joint_yhist_xhist = np.column_stack([y_target_hist, x_source_hist])
        
        # Fit KDE to estimate p(y_t, y_hist)
        kde_y_yhist = KernelDensity(bandwidth=bandwidth).fit(joint_y_yhist)
        logprobs_y_yhist = kde_y_yhist.score_samples(joint_y_yhist)
        
        # Fit KDE to estimate p(y_hist)
        kde_yhist = KernelDensity(bandwidth=bandwidth).fit(yhist)
        logprobs_yhist = kde_yhist.score_samples(yhist)
        
        # Fit KDE to estimate p(y_hist, x_hist)
        kde_yhist_xhist = KernelDensity(bandwidth=bandwidth).fit(joint_yhist_xhist)
        logprobs_yhist_xhist = kde_yhist_xhist.score_samples(joint_yhist_xhist)
        
        # Calculate H(Y_t, Y_hist)
        h_y_yhist = entropy_from_logprobs(logprobs_y_yhist)
        
        # Calculate H(Y_hist)
        h_yhist = entropy_from_logprobs(logprobs_yhist)
        
        # Calculate H(Y_t | Y_hist) = H(Y_t, Y_hist) - H(Y_hist)
        h_y_given_yhist = h_y_yhist - h_yhist
        
        # For H(Y_t | Y_hist, X_hist), we need to fit additional KDEs
        joint_y_yhist_xhist = np.column_stack([y_target.reshape(-1, 1), y_target_hist, x_source_hist])
        
        # Fit KDE to estimate p(y_t, y_hist, x_hist)
        kde_y_yhist_xhist = KernelDensity(bandwidth=bandwidth).fit(joint_y_yhist_xhist)
        logprobs_y_yhist_xhist = kde_y_yhist_xhist.score_samples(joint_y_yhist_xhist)
        
        # Calculate H(Y_t, Y_hist, X_hist)
        h_y_yhist_xhist = entropy_from_logprobs(logprobs_y_yhist_xhist)
        
        # Calculate H(Y_hist, X_hist)
        h_yhist_xhist = entropy_from_logprobs(logprobs_yhist_xhist)
        
        # Calculate H(Y_t | Y_hist, X_hist) = H(Y_t, Y_hist, X_hist) - H(Y_hist, X_hist)
        h_y_given_yhist_xhist = h_y_yhist_xhist - h_yhist_xhist
        
        # Calculate Transfer Entropy: H(Y_t | Y_hist) - H(Y_t | Y_hist, X_hist)
        te = h_y_given_yhist - h_y_given_yhist_xhist
        
        # Normalize if requested
        if self.normalize:
            # Fit KDE to estimate p(y_t)
            kde_y = KernelDensity(bandwidth=bandwidth).fit(y_target.reshape(-1, 1))
            logprobs_y = kde_y.score_samples(y_target.reshape(-1, 1))
            
            # Calculate H(Y_t)
            h_y = entropy_from_logprobs(logprobs_y)
            
            if h_y > 0:
                te /= h_y
        
        return max(0, te)  # Transfer entropy should not be negative
    
    def _significance_test(self, source, target):
        """Perform significance testing using surrogate data.
        
        Parameters
        ----------
        source : np.array
            Source time series
        target : np.array
            Target time series
            
        Returns
        -------
        float
            p-value for the null hypothesis (no information transfer)
        """
        import numpy as np
        
        # Calculate original Transfer Entropy
        original_te = self._get_transfer_entropy(source, target)
        
        # Generate surrogate data and calculate TE values
        surrogate_te_values = []
        
        for _ in range(self.n_surrogates):
            # Generate a surrogate by shuffling the source time series
            # This breaks any potential causal link while preserving marginal statistics
            surrogate_source = np.random.permutation(source)
            surrogate_te = self._get_transfer_entropy(surrogate_source, target)
            surrogate_te_values.append(surrogate_te)
        
        # Calculate p-value as the proportion of surrogate TE values >= original TE
        p_value = np.mean(np.array(surrogate_te_values) >= original_te)
        
        return p_value
    
    @classmethod
    def get_test_params(cls):
        """Return testing parameter settings for the estimator.
    
        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            
        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            - `create_test_instance` creates instances for the testing class
            - `get_test_params` creates instances for individual parameters.
        """

        # Default parameters for testing
        params1 = {
            "lag_target": 1,
            "lag_source": 1,
            "estimator": "binning",
            "n_bins": 5,
            "missing_values": "drop"
        }
        
        # Alternative parameters for testing
        params2 = {
            "lag_target": 2,
            "lag_source": 2,
            "estimator": "binning",
            "n_bins": 3,
            "significance_test": True,
            "missing_values": "drop"

        }
        
        return [params1, params2]