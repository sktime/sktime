# -*- coding: utf-8 -*-
""" multivariate shapelet transformations
"""
from sktime.classification.shapelet_based.dev.factories.shapelet_factory import ShapeletFactory
from sktime.classification.shapelet_based.dev.shapelets.shapelet_base import ShapeletBase
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy

__author__ = ["Alejandro Pasos Ruiz"]
__all__ = [
    "MultivariateShapeletTransform",
]

import heapq
import warnings

import numpy as np
import pandas as pd
import time

from sktime.transformations.base import _PanelToTabularTransformer

warnings.filterwarnings("ignore", category=FutureWarning)

class RandomFilter(_PanelToTabularTransformer):
    """Random Filter
    Attributes
    ----------
    """

    def __init__(
            self,
            shapelet_factory: ShapeletFactory,
            random_state,
            num_shapelets=50,
            min_shapelet_length=3,
            max_shapelet_length=1000,
            num_iterations=10000,
            remove_self_similar=True
    ):

        self.num_shapelets = num_shapelets
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.num_iterations = num_iterations
        self.remove_self_similar = remove_self_similar
        self.shapelet_factory = shapelet_factory
        self.random_state = random_state
        self.is_fitted_ = False
        self.shapelets = []
        super(RandomFilter, self).__init__()

    def fit(self, X, y=None):
        """A method to fit the shapelet transform to a specified X and y

        Parameters
        ----------
        X: pandas DataFrame
            The training input samples.
        y: array-like or list
            The class values for X

        Returns
        -------
        self : FullShapeletTransform
            This estimator
        """
        self.shapelets = []
        iteration = 0

        def getCandidate():
            shapelet_size = np.random.randint(self.min_shapelet_length, self.max_shapelet_length)
            instance_index = np.random.randint(0, len(X))

            candidate: ShapeletBase = self.shapelet_factory.get_random_shapelet(shapelet_size, instance_index,
                                                                                X.loc[instance_index],
                                                                                y[instance_index])
            self.set_quality(candidate, X, y)
            return candidate

        #start = time.process_time()
        #candidates = [getCandidate() for x in range(1000)]
        #print(time.process_time() - start)

        while True:
            shapelet_size = np.random.randint(self.min_shapelet_length, self.max_shapelet_length)
            instance_index = np.random.randint(0, len(X))

            candidate: ShapeletBase = self.shapelet_factory.get_random_shapelet(shapelet_size, instance_index,
                                                                                X.loc[instance_index],
                                                                                y[instance_index])
            self.set_quality(candidate, X, y)

            self.shapelets.append(candidate)
           # print(iteration)
            if iteration % 1000 == 0:
                self.shapelets.sort(key=lambda x: x.quality, reverse=True)
                self.shapelets = self.shapelets[:self.num_shapelets]
                print(iteration)

                if iteration > self.num_iterations:
                    self.shapelets.sort(key=lambda x: x.quality, reverse=True)
                    self.shapelets = self.shapelets[:self.num_shapelets]
                    self.is_fitted_ = True
                    return self

            iteration = iteration + 1

        return None

    def set_quality(self, candidate, X, y):

        def information_gain(Xs, split_point):
            '''
            Measures the reduction in entropy after the split
            :param v: Pandas Series of the members
            :param split:
            :return:
            '''
            split = pd.Series([1 if x >= split_point else 0 for x in Xs["Dist"]])
            members = Xs["Class"]
            split.name = 'split'
            members.name = 'members'
            # entropy_before = entropy(members.value_counts(), base=2)

            grouped_distrib = members.groupby(split) \
                .value_counts(normalize=True) \
                .reset_index(name='count') \
                .pivot_table(index='split', columns='members', values='count').fillna(0)
            entropy_after = entropy(grouped_distrib, base=2, axis=1)
            entropy_after *= split.value_counts(sort=False, normalize=True)
            return 1 - entropy_after.sum()


        Xs = pd.DataFrame(columns=['Dist', 'Class'])
        Xs["Dist"] = X.apply(lambda row: self.shapelet_factory.get_distance_it(candidate, row), axis=1)
        #Xs["Dist"] = np.apply_along_axis(lambda row: self.shapelet_factory.get_distance_it(candidate, row), 1, X)

        Xs["Class"] = np.vectorize(lambda yi: 1 if yi == candidate.class_index else 0)(y)
        Xs = Xs.sort_values(by=["Dist"])
        Xs["Splits"] = Xs["Class"].ne(Xs["Class"].shift())
        split_points = Xs[Xs["Splits"] == True]

        end = np.vectorize(lambda split: information_gain(Xs, split))(split_points["Dist"])

        candidate.set_quality(max(end))

    def transform(self, X):

        def set_row(x, columns):
            row = np.vectorize(lambda si: self.shapelet_factory.get_distance(si, x))(self.shapelets)
            return row

        cols = ['Shapelet_{0}'.format(x) for x in range(len(self.shapelets))]
        data = list(X.apply(lambda row: set_row(row, cols), axis=1))
        X_out = pd.DataFrame(data, columns=cols)

        return X_out

    @staticmethod
    def remove_self_similar_shapelets(shapelet_list):
        return None

    def get_shapelets(self):
        """An accessor method to return the extracted shapelets

        Returns
        -------
        shapelets: a list of Shapelet objects
        """
        return self.shapelets

    @staticmethod
    def zscore(a, axis=0, ddof=0):
        """A static method to return the normalised version of series.
        This mirrors the scipy implementation
        with a small difference - rather than allowing /0, the function
        returns output = np.zeroes(len(input)).
        This is to allow for sensible processing of candidate
        shapelets/comparison subseries that are a straight
        line. Original version:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats
        .zscore.html

        Parameters
        ----------
        a : array_like
            An array like object containing the sample data.

        axis : int or None, optional
            Axis along which to operate. Default is 0. If None, compute over
            the whole array a.

        ddof : int, optional
            Degrees of freedom correction in the calculation of the standard
            deviation. Default is 0.

        Returns
        -------
        zscore : array_like
            The z-scores, standardized by mean and standard deviation of
            input array a.
        """
        zscored = np.empty(a.shape)
        for i, j in enumerate(a):
            # j = np.asanyarray(j)
            sstd = j.std(axis=axis, ddof=ddof)

            # special case - if shapelet is a straight line (i.e. no
            # variance), zscore ver should be np.zeros(len(a))
            if sstd == 0:
                zscored[i] = np.zeros(len(j))
            else:
                mns = j.mean(axis=axis)
                if axis and mns.ndim < j.ndim:
                    zscored[i] = (j - np.expand_dims(mns, axis=axis)) / np.expand_dims(
                        sstd, axis=axis
                    )
                else:
                    zscored[i] = (j - mns) / sstd
        return zscored

    @staticmethod
    def euclidean_distance_early_abandon(u, v, min_dist):
        sum_dist = 0
        for i in range(0, len(u[0])):
            for j in range(0, len(u)):
                u_v = u[j][i] - v[j][i]
                sum_dist += np.dot(u_v, u_v)
                if sum_dist >= min_dist:
                    # The distance is higher, so early abandon.
                    return min_dist
        return sum_dist


class ShapeletPQ:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, shapelet):
        heapq.heappush(self._queue, (shapelet.info_gain, self._index, shapelet))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]

    def peek(self):
        return self._queue[0]

    def get_size(self):
        return len(self._queue)

    def get_array(self):
        return self._queue
