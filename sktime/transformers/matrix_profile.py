import numpy as np
from sktime.transformers.base import BaseTransformer
from sktime.transformers.compose import Tabulariser

class MatrixProfile(BaseTransformer):
    """
        Takes as input a time series dataset and returns the matrix profile and
        index profile for each time series of the dataset.

        Example of use:
        # mp, ip = MatrixProfile().transform(X, m)
        where X, the dataset, and m, the desired subsequence length to be used,
        are the inputs and mp and ip, both ndarrays with the matrix profile and
        the index profile, respectively, of each time series, are the outputs.
    """

    def sliding_dot_products(q, t, q_len, t_len):
        """
            Computes the sliding dot products between a query and a time series.

            Parameters
            ----------
                q: numpy.array
                    Query.
                t: numpy.array
                    Time series.
                q_len: int
                    Length of the query.
                t_len: int
                    Length of the time series.

            Output
            ------
                dot_prod: numpy.array
                            Sliding dot products between q and t.
        """

        # Reversing query and padding both query and time series
        t_padded = np.pad(t, (0, t_len))
        q_reversed = np.flipud(q)
        q_reversed_padded = np.pad(q_reversed, (0, 2*t_len-q_len))

        # Applying FFT to both query and time series
        t_fft = np.fft.fft(t_padded)
        q_fft = np.fft.fft(q_reversed_padded)

        # Applying inverse FFT to obtain the convolution of the time series by the query
        element_wise_mult = np.multiply(t_fft, q_fft)
        inverse_fft = np.fft.ifft(element_wise_mult)

        # Returns only the valid dot products from inverse_fft
        dot_prod = inverse_fft[q_len-1:t_len].real

        return dot_prod


    def calculate_distance_profile(dot_prod, q_mean, q_std, t_mean, t_std, q_len, n_t_subs):
        """
            Calculates the distance profile for the given query.

            Parameters
            ----------
                dot_prod: numpy.array
                    Sliding dot products between the time series and the query.
                q_mean: float
                    Mean of the elements of the query.
                q_std: float
                    Standard deviation of elements of the query.
                t_mean: numpy.array
                    Array with the mean of the elements from each subsequence of length(query) from the time series.
                t_std: numpy.array
                    Array with the standard deviation of the elements from each subsequence of length(query) from the time series.
                q_len: int
                    Length of the query.
                n_t_subs: int
                    Number of subsequences in the time series.

            Output
            ------
                d: numpy.array
                    Distance profile of query q.
        """

        d = [2 * q_len * (1 - ((dot_prod[i] - q_len * q_mean * t_mean[i]) / (q_len * q_std * t_std[i]))) for i in range(0, n_t_subs)]
        d = np.absolute(d)
        d = np.sqrt(d)

        return d


    def minimum_distance(mp, ip, dp, i, m, dp_len):
        """
            Finds the minimum distance in the distance profile, considering the exclusion zone.

            Parameters
            ----------
                mp: numpy.array
                    Matrix profile.
                ip: numpy.array
                    Index profile.
                dp: numpy.array
                    Distance profile.
                i: int
                    Index of the element to be compared from the matrix profile.
                m: int
                    Length of the subsequences.
                dp_len: int
                    Length of the distance profile.

            Output
            ------
                mp: numpy.array
                    Array with the distance between every subsequence and its nearest neighbor from the same time series.
                ip: numpy.array
                    Array with the indexes of the nearest neighbors of each subsequence.
        """

        # Initialization
        min_value = float("inf")
        min_index = -1

        for k in range(0, dp_len):
            if dp[k] < min_value and (k < i-m/2 or k > i+m/2):
                min_value = dp[k]
                min_index = k
        mp[i] = min_value
        ip[i] = min_index

        return mp, ip


    def stomp_self(ts, m):
        """
            STOMP implementation for self-similarity join.

            Parameters
            ----------
                ts: numpy.array
                    Time series.
                m: int
                    Length of the subsequences.

            Output
            ------
                mp: numpy.array
                    Array with the distance between every subsequence from ts1 to the nearest subsequence with same length from ts2.
                ip: numpy.array
                    Array with the index of the nearest neighbor of ts1 in ts2.
        """

        ts_len = ts.size

        ts = ts.flatten()

        # Number of subsequences
        n_subs = ts_len-m+1

        # Compute the mean and standard deviation
        ts_mean = [np.mean(ts[i:i+m]) for i in range(0, n_subs)]
        ts_std = [np.std(ts[i:i+m]) for i in range(0, n_subs)]

        # Compute the dot products between the first subsequence and every other subsequence
        dot_prod = sliding_dot_products(ts[0:m], ts, m, ts_len)
        first_dot_prod = np.copy(dot_prod)

        # Initialization
        mp = np.full(n_subs, float('inf')) # matrix profile
        ip = np.zeros(n_subs) # index profile

        # Compute the distance profile for the first subsequence
        dp = calculate_distance_profile(dot_prod, ts_mean[0], ts_std[0], ts_mean, ts_std, m, n_subs)

        # Updates the matrix profile
        mp, ip = minimum_distance(mp, ip, dp, 0, m, n_subs)

        for i in range(1, n_subs):
            for j in range(n_subs-1, 0, -1):
                dot_prod[j] = dot_prod[j-1] - ts[j-1]*ts[i-1] + ts[j-1+m]*ts[i-1+m]  # compute the next dot products using the previous ones
            dot_prod[0] = first_dot_prod[i]
            dp = calculate_distance_profile(dot_prod, ts_mean[i], ts_std[i], ts_mean, ts_std, m, n_subs)
            mp, ip = minimum_distance(mp, ip, dp, i, m, n_subs)

        return mp, ip


    def transform(self, X, m):
        """
            Takes as input a time series dataset and returns the matrix profile and
            index profile for each time series of the dataset.

            Parameters
            ----------
                X: pandas.DataFrame
                    Time series dataset.
                m: int
                    Length of the subsequences.

            Output
            ------
                mp: numpy.ndarray
                    N-dimensional array with the matrix profile for each time series.
                ip: numpy.ndarray
                    N-dimensional array with the index profile for each time series.
        """

        x_size = X.size

        # Convert into tabular format
        tabulariser = Tabulariser()
        X = tabulariser.transform(X.iloc[:, :1])

        n_subs = X.shape[1]-m+1

        # Initialization
        mp = np.full((X.shape[0], n_subs), float('inf'))  # matrix profile ndarray
        ip = np.zeros((X.shape[0], n_subs))  # index profile ndarray

        for i in range(0, x_size):
            mp[i], ip[i] = stomp_self(np.array([X.iloc[i, :]]), m)

        return mp, ip
