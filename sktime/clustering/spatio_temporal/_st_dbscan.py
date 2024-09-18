"""
ST-DBSCAN - fast scalable implementation of ST DBSCAN
            scales also to memory by splitting into frames
            and merging the clusters together
"""

__author__ = ["VectorNd"]

from sktime.clustering import BaseClusterer

import numpy as np

from scipy.spatial.distance import pdist, squareform
from scipy.sparse import coo_matrix

from sklearn.cluster import DBSCAN
from sklearn.utils import check_array
from sklearn.neighbors import NearestNeighbors

import warnings

class ST_DBSCAN(BaseClusterer):
    """
    A class to perform the ST_DBSCAN clustering
    Parameters
    ----------
    eps1 : float, default=0.5
        The spatial density threshold (maximum spatial distance) between 
        two points to be considered related.
    eps2 : float, default=10
        The temporal threshold (maximum temporal distance) between two 
        points to be considered related.
    min_samples : int, default=5
        The number of samples required for a core point.
    metric : string default='euclidean'
        The used distance metric - more options are
        'braycurtis', 'canberra', 'chebyshev', 'cityblock', 'correlation',
        'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'jensenshannon',
        'kulsinski','manhattan'.
    n_jobs : int or None, default=-1
        The number of processes to start -1 means use all processors 
    Attributes
    ----------
    labels : array, shape = [n_samples]
        Cluster labels for the data - noise is defined as -1
    """

    _tags = {
        "authors": ["VectorNd","eren-ck"],  
        "python_dependencies": ["scipy","sklearn"],  
        "X_inner_mtype": "numpyflat",
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:multithreading": False,
        "capability:predict": True,  
        "capability:predict_proba": False, 
        "capability:out_of_sample": True, 
    }

    def __init__(self, eps1=0.5, eps2=10, min_samples=5, metric='euclidean',n_jobs=-1):
        self.eps = eps1
        self.eps1 = eps1
        self.eps2 = eps2
        self.min_samples = min_samples
        self.metric = metric
        self.n_jobs = n_jobs

        super().__init__()

    def _fit(self, X):
        """Apply the ST DBSCAN algorithm 
        ----------
        X : 2D numpy array with
            The first element of the array should be the time 
            attribute as float. The following positions in the array are 
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        Returns
        -------
        self
        """

        X = check_array(X)

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        n, m = X.shape

        if len(X) < 20000:
            time_dist = pdist(X[:, 0].reshape(n, 1), metric=self.metric)
            euc_dist = pdist(X[:, 1:], metric=self.metric)

            dist = np.where(time_dist <= self.eps2, euc_dist, 2 * self.eps1)

            db = DBSCAN(eps=self.eps1,
                        min_samples=self.min_samples,
                        metric='precomputed')
            db.fit(squareform(dist))

            self.labels = db.labels_

        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                nn_spatial = NearestNeighbors(metric=self.metric,
                                              radius=self.eps1)
                nn_spatial.fit(X[:, 1:])
                euc_sp = nn_spatial.radius_neighbors_graph(X[:, 1:],
                                                           mode='distance')

                nn_time = NearestNeighbors(metric=self.metric,
                                           radius=self.eps2)
                nn_time.fit(X[:, 0].reshape(n, 1))
                time_sp = nn_time.radius_neighbors_graph(X[:, 0].reshape(n, 1),
                                                         mode='distance')

                row = time_sp.nonzero()[0]
                column = time_sp.nonzero()[1]
                v = np.array(euc_sp[row, column])[0]

                dist_sp = coo_matrix((v, (row, column)), shape=(n, n))
                dist_sp = dist_sp.tocsc()
                dist_sp.eliminate_zeros()

                db = DBSCAN(eps=self.eps1,
                            min_samples=self.min_samples,
                            metric='precomputed')
                db.fit(dist_sp)

                self.labels = db.labels_

        return self
    
    def _fit_frame_split(self , X , frame_size , frame_overlap=None):
        """Apply the ST DBSCAN algorithm with splitting it into frames.
        ----------
        X : 2D numpy array with
            The first element of the array should be the time (sorted by time)
            attribute as float. The following positions in the array are 
            treated as spatial coordinates. The structure should look like this [[time_step1, x, y], [time_step2, x, y]..]
            For example 2D dataset:
            array([[0,0.45,0.43],
            [0,0.54,0.34],...])
        frame_size : float, default= None
            If not none the dataset is split into frames and merged aferwards
        frame_overlap : float, default=eps2
            If frame_size is set - there will be an overlap between the frames
            to merge the clusters afterwards 
        Returns
        -------
        self
        """
        X = check_array(X)

        if frame_overlap == None:
            frame_overlap = self.eps2

        if not self.eps1 > 0.0 or not self.eps2 > 0.0 or not self.min_samples > 0.0:
            raise ValueError('eps1, eps2, minPts must be positive')

        if not frame_size > 0.0 or not frame_overlap > 0.0 or frame_size < frame_overlap:
            raise ValueError(
                'frame_size, frame_overlap not correctly configured.')

        time = np.unique(X[:, 0])

        labels = None
        right_overlap = 0
        max_label = 0

        for i in range(0, len(time), (frame_size - frame_overlap + 1)):
            for period in [time[i:i + frame_size]]:
                frame = X[np.isin(X[:, 0], period)]

                self.fit(frame)

                if not type(labels) is np.ndarray:
                    labels = self.labels
                else:
                    frame_one_overlap_labels = labels[len(labels) -
                                                      right_overlap:]
                    frame_two_overlap_labels = self.labels[0:right_overlap]

                    mapper = {}
                    for i in list(
                            zip(frame_one_overlap_labels,
                                frame_two_overlap_labels)):
                        mapper[i[1]] = i[0]
                    mapper[
                        -1] = -1 

                    ignore_clusters = set(
                        self.labels) - set(frame_two_overlap_labels)
                    if -1 in labels:
                        labels_counter = len(set(labels)) - 1
                    else:
                        labels_counter = len(set(labels))
                    for j in ignore_clusters:
                        mapper[j] = labels_counter
                        labels_counter += 1

                    new_labels = np.array([mapper[j] for j in self.labels])

                    labels = labels[0:len(labels) - right_overlap]
                    labels = np.concatenate((labels, new_labels))

                right_overlap = len(X[np.isin(X[:, 0],
                                              period[-frame_overlap + 1:])])

        self.labels = labels
        return self

    def _predict(self, X):
        """Predict the closest cluster each sample in X belongs to.

        Parameters
        ----------
        X : data to cluster based on model formed in _fit, of type self.get_tag(
        "X_inner_mtype")
        y: ignored, exists for API consistency reasons.

        Returns
        -------
        np.ndarray (1d array of shape (n_instances,))
            Index of the cluster each time series in X belongs to.
        """

        X = check_array(X)

        dist_matrix = squareform(pdist(X, metric=self.dist))
        predicted_labels = self.model_.fit_predict(dist_matrix)

        return predicted_labels

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
        "eps1": self.eps1,
        "eps2": self.eps2,
        "min_samples": self.min_samples,
        "metric": self.metric,
        "n_jobs": self.n_jobs,
        "algorithm": self.algorithm,
        "leaf_size": self.leaf_size,
        "metric_params": self.metric_params,
        "p": self.p,
        "dist": self.dist,
        "labels_": self.labels_,
        "dbscan__core_sample_indices_": self.model_.core_sample_indices_,
        "dbscan__components_": self.model_.components_,
        "dbscan__n_features_in_": self.model_.n_features_in_
        }

        return fitted_params

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for clusterers.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        param1 = {"eps1" : 0.3 , "eps2" : 5 , "min_samples" : 3 , "metric" : "euclidean" , "n_jobs" : 1}
        param2 = {"eps1" : 0.5 , "eps2" : 10 , "min_samples" : 2 , "metric" : "manhattan" , "n_jobs" : 2}
        
        return [param1,param2]
