"""fABBA - A Time Series Symbolic Representation.

Implentation derived from following library

{https://github.com/nla-group/fABBA/tree/master?tab=readme-ov-file}.

"""

# Copyright (c) 2025 poopsiclepooding. All rights reserved.
# This is a personal implementation that utilizes the fABBA library (https://github.com/nla-group/fABBA)
# for symbolic time series approximation and related tasks.
# This code is not a direct contribution to sktime.

__author__ = ["poopsiclepooding"]

# todo: add any necessary sktime external imports here
import os
import warnings
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
import pandas as pd
from joblib import parallel_backend
from scipy.sparse.linalg import svds
from sklearn.cluster import KMeans, MiniBatchKMeans

from sktime.transformations.base import BaseTransformer

# todo: add any necessary sktime internal imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class fABBA(BaseTransformer):
    """fABBA - An efficient symbolic aggregate approximation for temporal data.

    fABBA[1] is a dimensionality reduction technique that works by making
    symbolic approximation of temporal data, it well-suited for tasks such
    as compression, clustering, and classification. It converts time series
    data to sequence of tuples by adaptive polygonal chain approximation. It
    then uses clustering methods on these tuples and assigns symbols to the
    clusters.

    Parameters
    ----------
    alphabet_size : int, optional (default=5, greater equal 2)
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default=MyOtherEstimator(foo=42))
        descriptive explanation of paramc
    and so on

    References
    ----------
    .. [1] Chen, X., and Güttel, S.
    An efficient aggregation method for the symbolic representation of temporal data.
    arXiv preprint arXiv:2201.05697 (2022).
    https://arxiv.org/abs/2201.05697
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "scitype:transform-labels": "None",
        "X_inner_mtype": "df-list",
        "y_inner_mtype": "None",
        "univariate-only": True,
        "requires_X": True,
        "requires_y": False,
        "remember_data": False,
        "fit_is_empty": False,
        "X-y-must-have-same-index": False,
        "enforce_index_type": None,
        "transform-returns-same-time-index": False,
        "capability:inverse_transform": True,
        "capability:inverse_transform:range": None,
        "capability:inverse_transform:exact": False,
        "skip-inverse-transform": False,
        "capability:unequal_length": True,
        "capability:unequal_length:removes": False,
        "capability:missing_values": False,
        "capability:missing_values:removes": True,
        "capability:random_state": True,
        "property:randomness": "derandomized",
        "authors": ["poopsiclepooding"],
        "maintainers": ["poopsiclepooding"],
        "python_version": None,
        "python_dependencies": None,
    }

    def __init__(self,
        tolerance=0.2,
        method="agg",
        k=2,
        rate=0.5,
        sorting="norm",
        scl=1,
        max_iter=2,
        partition_rate=None,
        partition=None,
        max_len=np.inf,
        verbose=1,
        random_state=None,
        return_as_strings=False,
        return_start_values=False,
        fillna='ffill',
        auto_digitize=False,
        alpha=0.5,
        alphabet_set=0,
        n_jobs=-1
        ):

        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        # todo: write any hyper-parameters and components to self
        self.tolerance = tolerance
        self.method = method
        self.rate = rate
        self.sorting = sorting
        self.scl = scl
        self.max_iter = max_iter
        self.partition_rate = partition_rate
        self.return_as_strings = return_as_strings
        self.partition = partition
        self.max_len = max_len
        self.verbose = verbose
        self.random_state = random_state
        self.fillna = fillna
        self.alphabet_set = alphabet_set
        self.n_jobs = n_jobs
        self.k = k
        self.alpha = alpha
        self.auto_digitize = auto_digitize
        self.return_start_values = return_start_values

        self._k = k
        self._alpha = alpha
        self._auto_digitize = auto_digitize


        super().__init__()

        # todo: optional, parameter checking logic (if applicable) should happen here
        # if writes derived values to self, should *not* overwrite self.paramc etc
        # instead, write to self._paramc, self._newparam (starting with _)
        # example of handling conditional parameters or mutable defaults:
        if self._alpha is None:
            self._auto_digitize = True
        else:
            self._auto_digitize = auto_digitize

        # todo: if tags of estimator depend on component tags, set these here
        #  only needed if estimator is a composite
        #  tags set in the constructor apply to the object and override the class
        #
        # example 1: conditional setting of a tag
        # if est.foo == 42:
        #   self.set_tags(handles-missing-data=True)
        # example 2: cloning tags from component
        #   self.clone_tags(est2, ["enforce_index_type", "capability:missing_values"])

    def _custom_compress(self, ts):
        """Compress Using Polynomial Chain Approximation.

        Compress the numerical time series using polynomial chain approximation.

        Parameters
        ----------
            ts : ndarray
                single time series to be compressed
                size is (num_samples, num_features)

        Returns
        -------
            piece : list of tuples
                compressed time series in form of list of tuples
                each tuple is (length, slope, error)
        """
        ts = ts.squeeze(-1)
        if self.max_len < 0:
            max_len = len(ts)
        else:
            max_len = self.max_len

        start = 0
        end = 1
        pieces = list()
        x = np.arange(0, len(ts))
        epsilon =  np.finfo(float).eps

        # Polynomial Chain Approximation Algorithm
        while end < len(ts):
            inc = ts[end] - ts[start]
            err = ts[start] + (inc/(end-start))*x[0:end-start+1] - ts[start:end+1]
            err = np.inner(err, err)

            if (
                (err <= self.tolerance*(end-start-1) + epsilon)
                and (end-start-1 < max_len)
            ):
                (lastinc, lasterr) = (inc, err)
                end += 1
            else:
                pieces.append([end-start-1, lastinc, lasterr])
                start = end - 1

        pieces.append([end-start-1, lastinc, lasterr])
        return pieces

    def _custom_parallel_compress(self, X):
        """Parallel Compress.

        Compress the numerical series using polynomial chain approximation
        in a parallel manner.

        Parameters
        ----------
        X : Series or Panel of mtype df-list
            data to be transformed

        Returns
        -------
        pieces : list of list of tuples
            compressed time series in form of list of list of tuples
            each tuple is (length, slope, error)
        """
        assert isinstance(X, list), "X should be a list of Series or DataFrames"

        # Get number of processors to use
        n_jobs = self.n_jobs

        # Pool and partition doesn't take -1, need to count cpu
        if n_jobs==-1:
            n_jobs = os.cpu_count()

        # Check if only a single series or multiple
        if len(X) == 1:
            single_ts = True
        else:
            single_ts = False

        # If single series, process series in parts for speed
        if single_ts:
            single_series = np.asarray(X[0].values)

            # make sure the series is float64
            if single_series.dtype !=  'float64':
                single_series = np.asarray(single_series).astype('float64')

            # No point in more jobs than series len
            if n_jobs > len(single_series):
                n_jobs = 1

            # Partition the time series
            if self.partition is None:
                if self.partition_rate is None:
                    partition = n_jobs
                else:
                    partition = int(np.round(np.exp(1/self.partition_rate), 0))*n_jobs
                    if partition > len(single_series):
                        warnings.warn(
                            """Partition has exceed the maximum length of series."""
                            )
                        partition = len(single_series)
            else:
                if self.partition < len(single_series):
                    partition = self.partition
                    if n_jobs > partition: # to prevent useless processors
                        n_jobs = partition
                else:
                    warnings.warn("Partition has exceed the maximum length of series.")
                    partition = n_jobs

            # Interval of partition
            interval = int(len(single_series) / partition)
            # Get series in a list of 1D ndarrays
            series = [single_series[i*interval : (i+1)*interval]
                      for i in range(partition)]

        # if multiple series then convert them to list of ndarray
        else:
            # Get series in a list of 1D ndarrays
            series = [np.asarray(X[i].values) for i in range(len(X))]
            for i in range(len(series)):
                if series[i].dtype !=  'float64':
                    series[i] = np.asarray(series[i]).astype('float64')

            # No point in more jobs than no of series
            if n_jobs > len(series):
                n_jobs = len(series)

        # Compress the series using parallel processing
        pieces = list()
        self._start_set = list()

        p = Pool(n_jobs)

        self._start_set = [ts[0] for ts in series]

        # TODO : add fillna method
        result = [p.apply_async(func=self._custom_compress, args=(ts,))
                  for ts in series]

        p.close()
        p.join()
        pieces = [res.get() for res in result]

        return pieces

    def _custom_aggregate(self, data):
        """Aggregate/Cluster By Sorting.

        Aggregate/Clustering is done by sorting the data and choosing first
        data point as first group starting point. If the distance between the
        starting point of a group and another data point is less than or equal
        to the tolerance, the point is allocated to that group. First point
        outside the group becomes next group starting point.

        Parameters
        ----------
        data : np.ndarray
            the input that is array-like of shape (n_samples,).

        Returns
        -------
        labels (numpy.ndarray) :
            the group categories of the data after aggregation

        splist (list) :
            the list of the starting points

        nr_dist (int) :
            number of pairwise distance calculations
        """
        starting_point_list = list()
        num_pieces = data.shape[0]

        # Sorting using norm
        if self.sorting == "norm":
            # data shape will be (num_samples, num_features)
            cluster_data = data
            # norm data shape will be (num_samples,)
            norm_data = np.linalg.norm(cluster_data, ord=2, axis=1)
            # sorted indicies are returned
            sorted_indicies = np.argsort(norm_data)

        # Sorting using PCA
        elif self.sorting == "pca":
            # data shape will be (num_samples, num_features), need 0 centered for pca
            cluster_data = data - np.mean(data, axis=0)
            # need svd only if there more than 1 feature
            if cluster_data.shape[1] > 1:
                U1, s1, _ = svds(cluster_data, k=1, return_singular_vectors="u")
                norm_data = U1[:, 0] * s1[0]
            else:
                norm_data = norm_data[:, 0]
            # keep direction of sorting consistent
            norm_data = norm_data * np.sign(-norm_data[0])
            # sort this projected data
            sorted_indicies = np.argsort(norm_data)

        # No sorting
        else:
            norm_data = np.zeros(num_pieces)
            sorted_indicies = np.arange(num_pieces)

        # Labels to assign to clusters
        label = 0
        labels = [-1] * num_pieces

        # Do clustering
        for i in range(num_pieces):
            start = sorted_indicies[i]

            # if label already assigned then skip else its new start point
            if labels[start] >= 0:
                continue
            else:
                cluster_center = cluster_data[start, :]
                labels[start] = label
                num_points_in_group = 1

            # Move through all points after start to assign label till dist < tol
            for j in sorted_indicies[i:]:
                if labels[j] >= 0:
                    continue
                else:
                    # if distance from start point is more than tol then break
                    if (norm_data[j] - norm_data[start]) > self.alpha:
                        break

                    # above condition doesn't gurantee points are close in actual
                    # space since sorting might be done in a way that makes sorted
                    # distances different than acutal distances
                    # thus we need to confirm acutal distance is within tol
                    dist = cluster_data[j, :] - cluster_center
                    distance = np.inner(dist, dist)
                    if distance < self.alpha**2:
                        labels[j] = label
                        num_points_in_group += 1

            # update starting point list
            # it contains (start_index, label, num_points_in_group, center)
            starting_point_list.append([int(start), int(label)]
                                        + [int(num_points_in_group)]
                                        + data[start, :].tolist())
            # increase label
            label += 1

        return np.array(labels), starting_point_list

    def _custom_assign_symbols(self, labels):
        """Assign Symbols to Clusters.

        Assign symbols to each cluster label based on alphabet set.
        If no of unique labels exceed the alphabet set, use ASCII values.

        Parameters
        ----------
        labels : numpy.ndarray
            the group categories of the data after clustering, size is (num_samples,)

        Returns
        -------
        symbols : list of strings
            symbols correspoding to each label
        alphabets : list of strings
            the alphabets used for symbolization
        """
        # Get number of unique labels
        unique, unique_counts = np.unique(labels, return_counts=True)
        num_unique = len(unique)

        # Get alphabets to use
        if self.alphabet_set == 0:
            alphabets = ['A','a','B','b','C','c','D','d','E','e',
                        'F','f','G','g','H','h','I','i','J','j',
                        'K','k','L','l','M','m','N','n','O','o',
                        'P','p','Q','q','R','r','S','s','T','t',
                        'U','u','V','v','W','w','X','x','Y','y','Z','z']

        elif self.alphabet_set == 1:
            alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                        'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
                        'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                        'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v',
                        'w', 'x', 'y', 'z']

        elif isinstance(self.alphabet_set, list):
            if num_unique < len(self.alphabet_set):
                alphabets = self.alphabet_set
            else:
                raise ValueError("""Alphabet set provided is smaller than
                                 number of unique clusters.""")

        else:   # default alphabet set
            alphabets = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l',
                        'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 
                        'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                        'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
                        'W', 'X', 'Y', 'Z']

        # If more unique labels than alphabets in set, use ASCII values
        if num_unique > len(alphabets):
            warnings.warn("""Using ASCII values since num of unique clusters
                          more than alphabet set""")
            alphabets = [chr(i + 33) for i in range(0, num_unique)]
        # If less unique labels than alphabets in set, use only what is needed
        else:
            alphabets = alphabets[0:num_unique]

        # Assign symbols to labels
        alphabets = np.asarray(alphabets)
        symbols = alphabets[labels]

        return symbols.tolist(), alphabets.tolist()

    def _custom_strings_separation(self, strings, num_pieces):
        """Separates String Sequences.

        Separates strings of each time series based on num of pieces
        of each time series.

        Parameters
        ----------
        strings : list of strings
            symbolized time series in form of list of strings
            this list is for all time series concatenated
        num_pieces : list of int
            number of pieces for each time series

        Returns
        -------
        strings_sequences : list of strings
            symbolized time series in form of list of strings
            each string is for a single time series
        """
        strings_sequences = list()

        start = 0
        for len_ts in num_pieces:
            strings_sequences.append(strings[start : start + len_ts])
            start += len_ts

        return strings_sequences

    def _custom_labels_separation(self, labels, num_pieces):
        """Separates String Sequences.

        Separates strings of each time series based on num of pieces
        of each time series.

        Parameters
        ----------
        labels : ndarray of int
            symbolized time series in form of ndarray of int
            this array is for all time series concatenated

        num_pieces : list of int
            number of pieces for each time series

        Returns
        -------
        labels_sequence : list of ints
            symbolized time series in form of list of ints
            each ints is for a single time series
        """
        labels_sequences = list()

        start = 0
        for len_ts in num_pieces:
            labels_sequences.append(labels[start : start + len_ts])
            start += len_ts

        return labels_sequences

    def _custom_len_centers(self, len_pieces, labels):
        """Get Length Centers When scl is 0.

        Get centers of length when scl is 0 by taking mean of lengths

        Parameters
        ----------
        len_pieces : ndarry
            array of lengths of pieces, size is (num_samples,)
        labels : ndarray of int
            the group categories of the data after clustering, size is (num_samples,)

        Returns
        -------
        centers : ndarray of float
            centers of lengths for each cluster, size is (num_clusters,)
        """
        centers = np.zeros(self.k)

        # Take mean of lengths for same labels
        for label in np.unique(labels):
            centers[label] = np.mean(len_pieces[np.where(labels == label)])

        return centers

    def _custom_symbolize(self, X, pieces):
        """Symbolize Using Clustering.

        Symbolize the list of tuples using clustering methods.
        Assign symbols to each cluster.

        Parameters
        ----------
        X : Series or Panel of mtype df-list
            data to be transformed
        pieces : list of list of tuples

        Returns
        -------
        strings_sequences : list of strings
            symbolized time series in form of list of strings
            each string is for a single time series
        or
        labels_sequences : list of ints
            symbolized time series in form of list of ints
            each ints is for a single time series
        """
        # Get num of pieces for each ts
        num_pieces = list()
        for i in range(len(pieces)):
            num_pieces.append(len(pieces[i]))

        # Concat all pieces
        pieces = np.vstack([np.array(piece) for piece in pieces])[:, :2]

        # Find std of pieces
        self._std = np.std(pieces, axis=0)
        if self._std[0] == 0:   # prevent zero-division
            self._std[0] = 1
        if self._std[1] == 0:   # prevent zero-division
            self._std[1] = 1

        # If scaling of length is 0
        if self.scl == 0:
            len_pieces = pieces[:,0]

        # Scale pieces by scl to give more importance to len in clustering
        # if scl is 0 then all len will be same and not influence clustering
        pieces = pieces * np.array([self.scl, 1]) / self._std

        # Get amount of unique pieces, this will be maximum no of clusters
        max_k = np.unique(pieces, axis=0).shape[0]

        # Get sum of length of all time series
        if len(X) == 1:
            sum_of_length = len(X[0])
            self._eta = 0.01
        elif len(X) > 1:
            sum_of_length = [len(ts) for ts in X]
            self._eta = 0.000002
        else:
            raise ValueError("X found is not a list of dataframes")

        # If method is aggregate clustering
        if self.method == "agg":
            # If auto_digitize is true, calculate alpha
            if self._auto_digitize:
                self._alpha = pow(
                        60 * sum_of_length * (
                            np.abs(sum_of_length - max_k)) *
                            (self.tolerance**2)/(max_k*(self.eta**2) *
                            (3*(sum_of_length**4)+2-5*(max_k**2))
                            ), 1/4)

            # Run aggregate clustering
            labels, splist = self._custom_aggregate(pieces)
            splist = np.array(splist)
            centers = splist[:,3:5] * self._std / np.array([self.scl, 1])
            self._k = centers.shape[0]

        # If method is f-kmeans clustering
        elif self.method == "f-kmeans":
            # check if more unique tuples than clusters
            if self._k > max_k:
                self._k = max_k
                warnings.warn(f"""More clusters than unique tuples,
                              so k reduced to {self._k}""")

            with parallel_backend('threading', n_jobs=self.n_jobs):
                # TODO : should we use this old method or this minibatchKmeans?
                kmeans = MiniBatchKMeans(n_clusters=self._k,
                                        batch_size=self.r * len(pieces), # % of batches
                                        init='k-means++',
                                        n_init="auto",
                                        max_iter=self.max_iter,
                                        random_state=self.random_state)
                kmeans.fit(pieces)

            # get labels and centers of clusters
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])

        # default to k-means clustering
        else:
            # check if more unique tuples than clusters
            if self._k > max_k:
                self._k = max_k
                warnings.warn(f"""More clusters than unique tuples,
                              so k reduced to {self._k}""")

            # apply k means clustering
            # TODO : does it make sense to add tol?
            with parallel_backend('threading', n_jobs=self.n_jobs):
                kmeans = KMeans(n_clusters=self._k,
                                init='k-means++',
                                n_init="auto",
                                max_iter=self.max_iter,
                                random_state=self.random_state)
                kmeans.fit(pieces)

            # get labels and centers of clusters
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_ * self._std / np.array([self.scl, 1])

        # To get centers of length when scl is 0 we take cluster labels from clustering
        # using only inc, then taking mean of same labels to get len cluster centers.
        if self.scl == 0:
            centers[:, 0] = self._custom_len_centers(len_pieces, labels)

        # If returning strings assign symbols to the labels of clusters
        if self.return_as_strings:
            strings, alphabets = self._custom_assign_symbols(labels)
            self._parameter_alphabets = alphabets
        else:
            self._parameter_alphabets = None


        # Save params of centers and their corresponding symbol
        self._parameter_centers = centers

        # Save no of clusters formed
        self._parameter_num_grp = self._parameter_centers.shape[0]

        # Return strings if told else return labels(ints)
        if self.return_as_strings:
            strings_sequences = self._custom_string_separation(strings, num_pieces)
            return strings_sequences
        else:
            labels_sequences = self._custom_labels_separation(labels, num_pieces)
            return labels_sequences

    def _fit(self, X, y=None):
        """Fit transformer to X and y.

        private _fit containing the core logic, called from fit

        Parameters
        ----------
        X : Series or Panel of mtype df-list
        y : None
            unused

        Returns
        -------
        self: reference to self
        """
        self.pieces_ = self._custom_parallel_compress(X)
        self.strings_ = self._custom_symbolize(X, self.pieces_)

        return self

    def _custom_pieces_to_symbols(self, pieces):
        """Convert Pieces To Symbols.

        Comverts ndarray of pieces (len, inc) into symbols based on saved centers.

        Parameters
        ----------
        pieces : ndarray
            array of pieces, size is (num_samples, 2)

        Returns
        -------
        strings_sequences : list of strings
        or
        labels_sequences : list of ints
        """
        symbols = list()

        # convert each piece into symbol
        for piece in pieces:
            symbol = np.argmin(
                np.linalg.norm(self._parameter_centers - piece, ord=2, axis=1)
                )
            symbols.append(symbol)

        # if returning string then get alphabets
        if self.return_as_strings:
            symbols = [self._parameter_alphabets[symbol] for symbol in symbols]
            strings_sequences = symbols
            return strings_sequences
        # else return the label ints
        else:
            labels_sequences = symbols
            return labels_sequences

    def _transform_single_series(self, ts):
        """Convert single series to symbols.

        Converts a single time series into a string based on saved cluster
        centers from calling fit.

        Parameters
        ----------
            ts : 1D ndarray
                numerical time series to be compressed

        Returns
        -------
            symbols : list of ints or list of chars
                symbolized time series in form of list of ints or chars
        """
        # get pieces of the series
        pieces = self._custom_compress(ts)

        # convert to ndarray
        pieces = np.asarray(pieces)[:, :2]

        # convert pieces to symbols
        symbols = self._custom_pieces_to_symbols(pieces)

        return symbols

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        private _transform containing core logic, called from transform

        Parameters
        ----------
        X : Series or Panel of mtype df-list
        y : None
            unused

        Returns
        -------
        X_transformed : list of strings
        transformed version of X
        """
        assert isinstance(X, list), "X should be a list of Series or DataFrames"

        # Get number of processors to use
        n_jobs = self.n_jobs

        # Pool and partition doesn't take -1, need to count cpu
        if n_jobs==-1:
            n_jobs = os.cpu_count()

        # Check if only a single series or multiple
        if len(X) == 1:
            single_ts = True
        else:
            single_ts = False

        # If univariate, process series in parts for speed
        if single_ts:
            single_series = np.asarray(X[0].values)

            # make sure the series is float64
            if single_series.dtype !=  'float64':
                single_series = np.asarray(single_series).astype('float64')

            # No point in more jobs than series len
            if n_jobs > len(single_series):
                n_jobs = 1

            # Partition the time series
            if self.partition is None:
                if self.partition_rate is None:
                    partition = n_jobs
                else:
                    partition = int(np.round(np.exp(1/self.partition_rate), 0))*n_jobs
                    if partition > len(single_series):
                        warnings.warn("Partition has exceed the maximum length of series.")
                        partition = len(single_series)
            else:
                if self.partition < len(single_series):
                    partition = self.partition
                    if n_jobs > partition: # to prevent useless processors
                        n_jobs = partition
                else:
                    warnings.warn("Partition has exceed the maximum length of series.")
                    partition = n_jobs

            # Interval of partition
            interval = int(len(single_series) / partition)

            # Get series in a list of 1D ndarrays
            series = [single_series[i*interval : (i+1)*interval]
                      for i in range(partition)]
        else:
            # Get series in a list of 1D ndarrays
            series = [np.asarray(X[i].values) for i in range(len(X))]
            for i in range(len(series)):
                if series[i].dtype !=  'float64':
                    series[i] = np.asarray(series[i]).astype('float64')

        # Transform each of the series into symbols
        symbols = list()
        start_set = list()

        p = Pool(n_jobs)

        start_set = [ts[0][0] for ts in series]

        # TODO : add fillna method
        result = [p.apply_async(func=self._transform_single_series, args=(ts,))
                  for ts in series]

        p.close()
        p.join()
        symbols = [res.get() for res in result]

        # If return start values then append start value at start of symbols list
        if self.return_start_values:
            for i in range(len(start_set)):
                symbols[i].insert(0, start_set[i])

        # Return as list of df
        return_data = list()
        for i in range(len(symbols)):
            return_data.append(pd.DataFrame({"symbols": symbols[i]}))

        return return_data

    def _custom_inverse_symbolize(self, symbol_sequence):
        """Inverse Symbolization.

        Converts symbols back into cluster centers, which will be pieces to be used.

        Parameters
        ----------
            symbol_sequence : char or int
                symbolized time series in form of string

        Returns
        -------
            pieces : ndarray
                compressed version of time series
                each row is (length, slope)
        """
        pieces = list()

        # If string was returned then convert to labels and get pieces.
        if self.return_as_strings:
            for symbol in symbol_sequence:
                assert symbol in self._parameter_alphabets, "Symbol not in alphabet set"
                label = self._parameter_alphabets.index(symbol)
                pieces.append(self._parameter_centers[label, :])

        # else labels already given, just get pieces
        else:
            for label in symbol_sequence:
                assert label < self._parameter_num_grp, "Label not in range of cluster"
                pieces.append(self._parameter_centers[int(label), :])

        return np.vstack(pieces)

    def _custom_quantize_lengths(self, pieces):
        """Quantize Lengths.

        Converts lengths of pieces to nearest integer. 
        Since lengths can be floats during clustering, they need to be
        quantized back to integers.

        Parameters
        ----------
            pieces : ndarray
                compressed version of time series
                each row is (length, slope)

        Returns
        -------
            pieces : ndarray
                compressed version of time series
                each row is (length, slope)
        """
        # If only single piece, just round it
        if len(pieces) == 1:
            pieces[0,0] = round(pieces[0,0])
        # If many pieces, round individually such that sum of lengths remain same
        else:
            for p in range(len(pieces)-1):
                # Get correction to be applied 
                corr = round(pieces[p,0]) - pieces[p,0]

                # Add correction to current piece and subtract from next piece
                pieces[p,0] = round(pieces[p,0] + corr)
                pieces[p+1,0] = pieces[p+1,0] - corr

                # If piece length becomes 0, make it 1 and subtract 1 from next piece
                if pieces[p,0] == 0:
                    pieces[p,0] = 1
                    pieces[p+1,0] -= 1

            # Round the last piece
            pieces[-1,0] = round(pieces[-1,0],0)

        return pieces

    def _custom_inverse_compress(self, pieces, start_value):
        """Inverse Compress.

        Gets back the time series from pieces using start value.

        Parameters
        ----------
            pieces : ndarray
                compressed version of time series
                each row is (length, slope)
            start_value : float
                starting value of the time series

        Returns
        -------
            ts : 1D ndarray
                reconstructed time series
        """
        # Start with start value
        ts = [start_value]

        # Stick pieces one after another
        for j in range(0, len(pieces)):
            x = np.arange(0,pieces[j,0]+1)/(pieces[j,0])*pieces[j,1]
            y = ts[-1] + x
            ts = ts + y[1:].tolist()

        return ts

    def _inverse_transform_single_series(self, start_value, symbol_sequence):
        """Inverse Transform Single Series.

        Transforms a single symbol sequence back to time series using 
        start values. Cluster cneters saved are used to get back the pieces.
        These pieces are then joined to get back the time series.

        Parameters
        ----------
            start_value : float
                starting value of the time series
            symbol_sequence : char or int
                symbolized time series in form of string

        Returns
        -------
            ts : 1D ndarray
                reconstructed time series
        """
        # Get pieces from symbols
        pieces = self._custom_inverse_symbolize(symbol_sequence)

        #
        pieces = self._custom_quantize_lengths(pieces)

        # Get back time series from inverse compress
        ts = self._custom_inverse_compress(pieces, start_value)

        return ts

    def _inverse_transform(self, X, y=None):
        """Inverse transform, inverse operation to transform.

        private _inverse_transform containing core logic, called from inverse_transform

        Parameters
        ----------
        X : Series, Panel, or Hierarchical data, of mtype X_inner_mtype
            if X_inner_mtype is list, _inverse_transform must support all types in it
            Data to be inverse transformed
        y : Series, Panel, or Hierarchical data, of mtype y_inner_mtype, default=None
            Additional data, e.g., labels for transformation

        Returns
        -------
        inverse transformed version of X
        """
        if self.return_start_values is False:
            raise ValueError("Start values are needed for inverse transform")

        num_series = len(X)
        start_values = list()
        symbol_sequences = list()

        # Get symbol sequences and start values
        for item in X:
            if isinstance(item, pd.DataFrame):
                # Flatten the single column to a 1D array
                col_values = item.iloc[:, 0].values

                # First value
                start_values.append(col_values[0])

                # Remaining values
                symbol_sequences.append(col_values[1:])
            else:
                # TODO : Support other types
                raise ValueError("X should be a list of DataFrames")

        # Start parallel inverse transform
        n_jobs = self.n_jobs

        # Pool doesn't take -1, need to count cpu
        if n_jobs==-1:
            n_jobs = os.cpu_count()

        p = Pool(n_jobs)

        result = [p.apply_async(
                func=self._inverse_transform_single_series,
                args=(start_values[i], symbol_sequences[i],)
                ) for i in range(num_series)]

        p.close()
        p.join()

        # Get series fromr results
        series = [res.get() for res in result]

        # Return as a df-list
        return_data = list()
        for i in range(len(series)):
            return_data.append(pd.DataFrame({"ts": series[i]}))

        return return_data

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        fitted_params = {
            "cluster_centers": self._parameter_centers,
            "num_clusters": self._parameter_num_grp,
            "alphabets": self._parameter_alphabets,
        }
        return fitted_params
        # implement here
        #
        # when this function is reached, it is already guaranteed that self is fitted
        #   this does not need to be checked separately
        #
        # parameters of components should follow the sklearn convention:
        #   separate component name from parameter name by double-underscore
        #   e.g., componentname__paramname

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for transformers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        if parameter_set == "default":
                # single default-ish test config
                params = [
                    {
                        "tolerance": 0.1,
                        "method": "agg",
                        "k": 2,
                        "rate": 0.5,
                        "sorting": "norm",
                        "scl": 1,
                        "max_iter": 1,
                        "auto_digitize": False,
                        "alpha": 0.5,
                        "alphabet_set": 0,
                        "n_jobs": 1,
                    },
                    {
                        "tolerance": 0.05,
                        "method": "kmeans",
                        "k": 3,
                        "rate": 0.5,
                        "sorting": "norm",
                        "scl": 2,
                        "max_iter": 1,
                        "auto_digitize": False,
                        "alpha": 0.7,
                        "alphabet_set": 0,
                        "n_jobs": 1,
                    },
                ]
                return params
        return cls.get_test_params("default")
