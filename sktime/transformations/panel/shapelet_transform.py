# -*- coding: utf-8 -*-
"""Random binary shapelet transformation.

A transformer from the time domain into the shapelet domain. Randomly samples shapelets
to use in the transformation, with capabilities for contracting.
"""

__author__ = ["MatthewMiddlehurst", "jasonlines", "dguijo"]
__all__ = ["RandomShapeletTransform"]

import heapq
import math
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import NumbaPendingDeprecationWarning, njit
from sklearn.utils import check_random_state

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.panel import check_X, check_X_y


class RandomShapeletTransform(_PanelToTabularTransformer):
    """Random Shapelet Transform.

    Implementation of the binary shapelet transform along the lines of [1]_[2]_, with
    randomly extracted shapelets.

    Overview: Input "n" series with "d" dimensions of length "m". Continuously extract
    candidate shapelets and filter them in batches.
        For each candidate shapelet
            - Extract a shapelet from an instance with random length, position and
              dimension
            - Using its distance to train cases, calculate the shapelets information
              gain
            - Abandon evaluating the shapelet if it is impossible to obtain a higher
              information gain than the current worst
        For each shapelet batch
            - Add each candidate to its classes shapelet heap, removing the lowest
              information gain shapelet if the max number of shapelets has been met
            - Remove self-similar shapelets from the heap
    Using the final set of filtered shapelets, transform the data into a vector of
    of distances from a series to each shapelet.

    Parameters
    ----------
    n_shapelet_samples : int, default=100000
        The number of candidate shapelets to be considered for the final transform.
        Filtered down to <= max_shapelets, keeping the shapelets with the most
        information gain.
    max_shapelets : int or None, default=None
        Max number of shapelets to keep for the final transform. Each class value will
        have its own max, set to n_classes / max_shapelets. If None uses the min between
        10 * n_instances and 1000
    min_shapelet_length : int, default=3
        Lower bound on candidate shapelet lengths.
    max_shapelet_length : int or None, default= None
        Upper bound on candidate shapelet lengths. If None no max length is used.
    remove_self_similar : boolean, default=True
        Remove overlapping "self-similar" shapelets when merging candidate shapelets.
    time_limit_in_minutes : int, default=0
        Time contract to limit build time in minutes, overriding n_shapelet_samples.
        Default of 0 means n_shapelet_samples is used.
    contract_max_n_shapelet_samples : int, default=np.inf
        Max number of shapelets to extract when time_limit_in_minutes is set.
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `transform`.
        ``-1`` means using all processors.
    batch_size : int or None, default=None
        Number of shapelet candidates processed before being merged into the set of best
        shapelets. If none the max of n_instances and max_shapelets is used.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    n_instances : int
        The number of train cases.
    n_dims : int
        The number of dimensions per case.
    series_length : int
        The length of each series.
    classes_ : list
        The classes labels.
    shapelets : list
        The stored shapelets and relating information after a dataset has been
        processed.
        Each item in the list is a tuple containing the following 7 items:
        (shapelet information gain, shapelet length, start position the shapelet was
        extracted from, shapelet dimension, index of the instance the shapelet was
        extracted from in fit, class value of the shapelet, The z-normalised shapelet
        array)

    See Also
    --------
    ShapeletTransformClassifier

    Notes
    -----
    For the Java version, see
    `TSML <https://github.com/uea-machine-learning/tsml/blob/master/src/main/
    java/tsml/transformers/ShapeletTransform.java>`_.

    References
    ----------
    .. [1] Jon Hills et al., "Classification of time series by shapelet transformation",
       Data Mining and Knowledge Discovery, 28(4), 851--881, 2014.
    .. [2] A. Bostrom and A. Bagnall, "Binary Shapelet Transform for Multiclass Time
       Series Classification", Transactions on Large-Scale Data and Knowledge Centered
       Systems, 32, 2017.

    Examples
    --------
    >>> from sktime.transformations.panel.shapelet_transform import (
    ...     RandomShapeletTransform
    ... )
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> t = RandomShapeletTransform(
    ...     n_shapelet_samples=500,
    ...     max_shapelets=10,
    ...     batch_size=100,
    ... )
    >>> t.fit(X_train, y_train)
    RandomShapeletTransform(...)
    >>> X_t = t.transform(X_train)
    """

    def __init__(
        self,
        n_shapelet_samples=100000,
        max_shapelets=None,
        min_shapelet_length=3,
        max_shapelet_length=None,
        remove_self_similar=True,
        time_limit_in_minutes=0.0,
        contract_max_n_shapelet_samples=np.inf,
        n_jobs=1,
        batch_size=None,
        random_state=None,
    ):
        self.n_shapelet_samples = n_shapelet_samples
        self.max_shapelets = max_shapelets
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.remove_self_similar = remove_self_similar

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelet_samples

        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.random_state = random_state

        # The following set in method fit
        self.n_classes = 0
        self.n_instances = 0
        self.n_dims = 0
        self.series_length = 0
        self.classes_ = []
        self.shapelets = []

        self._n_shapelet_samples = n_shapelet_samples
        self._max_shapelets = max_shapelets
        self._max_shapelet_length = max_shapelet_length
        self._n_jobs = n_jobs
        self._batch_size = batch_size
        self._class_counts = []
        self._class_dictionary = {}
        self._sorted_indicies = []

        super(RandomShapeletTransform, self).__init__()

    def fit(self, X, y):
        """Fit the shapelet transform to a specified X and y.

        Parameters
        ----------
        X: pandas DataFrame or np.ndarray
            The training input samples.
        y: array-like or list
            The class values for X.

        Returns
        -------
        self : RandomShapeletTransform
            This estimator.
        """
        X, y = check_X_y(X, y, coerce_to_numpy=True)

        # this is a few versions away currently, and heaps dont support the replacement
        warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

        self._n_jobs = check_n_jobs(self.n_jobs)

        self.classes_, self._class_counts = np.unique(y, return_counts=True)
        self.n_classes = self.classes_.shape[0]
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        self.n_instances, self.n_dims, self.series_length = X.shape

        # if self.n_shapelet_samples is None:
        #     self._n_shapelet_samples = ??? todo find good default

        if self.max_shapelets is None:
            self._max_shapelets = min(10 * self.n_instances, 1000)

        if self.batch_size is None:
            self._batch_size = max(self.n_instances, self._max_shapelets)

        if self.max_shapelet_length is None:
            self._max_shapelet_length = self.series_length

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        fit_time = 0

        max_shapelets_per_class = self._max_shapelets / self.n_classes
        shapelets = [[(-1.0, -1, -1, -1, -1, -1)] for _ in range(self.n_classes)]
        n_shapelets_extracted = 0

        if time_limit > 0:
            while (
                fit_time < time_limit
                and n_shapelets_extracted < self.contract_max_n_shapelet_samples
            ):
                candidate_shapelets = Parallel(n_jobs=self._n_jobs)(
                    delayed(self._extract_random_shapelet)(
                        X,
                        y,
                        n_shapelets_extracted + i,
                        shapelets,
                        max_shapelets_per_class,
                    )
                    for i in range(self._batch_size)
                )

                for i, heap in enumerate(shapelets):
                    RandomShapeletTransform._merge_shapelets(
                        heap,
                        candidate_shapelets,
                        max_shapelets_per_class,
                        i,
                    )

                if self.remove_self_similar:
                    for i, heap in enumerate(shapelets):
                        to_keep = (
                            RandomShapeletTransform._remove_self_similar_shapelets(heap)
                        )
                        shapelets[i] = [n for (n, b) in zip(heap, to_keep) if b]

                n_shapelets_extracted += self._batch_size
                fit_time = time.time() - start_time
        else:
            while n_shapelets_extracted < self._n_shapelet_samples:
                n_shapelets_to_extract = (
                    self._batch_size
                    if n_shapelets_extracted + self._batch_size
                    <= self._n_shapelet_samples
                    else self._n_shapelet_samples - n_shapelets_extracted
                )

                candidate_shapelets = Parallel(n_jobs=self._n_jobs)(
                    delayed(self._extract_random_shapelet)(
                        X,
                        y,
                        n_shapelets_extracted + i,
                        shapelets,
                        max_shapelets_per_class,
                    )
                    for i in range(n_shapelets_to_extract)
                )

                for i, heap in enumerate(shapelets):
                    RandomShapeletTransform._merge_shapelets(
                        heap,
                        candidate_shapelets,
                        max_shapelets_per_class,
                        i,
                    )

                if self.remove_self_similar:
                    for i, heap in enumerate(shapelets):
                        to_keep = (
                            RandomShapeletTransform._remove_self_similar_shapelets(heap)
                        )
                        shapelets[i] = [n for (n, b) in zip(heap, to_keep) if b]

                n_shapelets_extracted += n_shapelets_to_extract

        self.shapelets = [
            (
                s[0],
                s[1],
                s[2],
                s[3],
                s[4],
                self.classes_[s[5]],
                _z_norm(X[s[4], s[3], s[2] : s[2] + s[1]]),
            )
            for class_shapelets in shapelets
            for s in class_shapelets
            if s[0] > 0
        ]
        self.shapelets.sort(reverse=True, key=lambda s: (s[0], s[1], s[2], s[3], s[4]))

        to_keep = RandomShapeletTransform._remove_identical_shapelets(self.shapelets)
        self.shapelets = [n for (n, b) in zip(self.shapelets, to_keep) if b]

        self._sorted_indicies = []
        for s in self.shapelets:
            sabs = np.abs(s[6])
            self._sorted_indicies.append(
                sorted(range(s[1]), reverse=True, key=lambda i: sabs[i])
            )

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        """Transform X according to the extracted shapelets.

        Parameters
        ----------
        X : pandas DataFrame or np.ndarray
            The input data to transform.

        Returns
        -------
        output : pandas DataFrame
            The transformed dataframe in tabular format.
        """
        self.check_is_fitted()
        X = check_X(X, coerce_to_numpy=True)

        # this is a few versions away currently, and heaps dont support the replacement
        warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

        output = np.zeros((len(X), len(self.shapelets)))

        for i, series in enumerate(X):
            dists = Parallel(n_jobs=self._n_jobs)(
                delayed(_online_shapelet_distance)(
                    series[shapelet[3]],
                    shapelet[6],
                    self._sorted_indicies[n],
                    shapelet[2],
                    shapelet[1],
                )
                for n, shapelet in enumerate(self.shapelets)
            )

            output[i] = dists

        return pd.DataFrame(output)

    def _extract_random_shapelet(self, X, y, i, shapelets, max_shapelets_per_class):
        rs = 255 if self.random_state == 0 else self.random_state
        rs = (
            None
            if self.random_state is None
            else (rs * 37 * (i + 1)) % np.iinfo(np.int32).max
        )
        rng = check_random_state(rs)

        inst_idx = i % self.n_instances
        cls_idx = self._class_dictionary[y[inst_idx]]
        worst_quality = (
            shapelets[cls_idx][0][0] if shapelets == max_shapelets_per_class else -1
        )

        length = (
            rng.randint(0, self._max_shapelet_length - self.min_shapelet_length)
            + self.min_shapelet_length
        )
        position = rng.randint(0, self.series_length - length)
        dim = rng.randint(0, self.n_dims)

        shapelet = _z_norm(X[inst_idx, dim, position : position + length])
        sabs = np.abs(shapelet)
        sorted_indicies = sorted(range(length), reverse=True, key=lambda i: sabs[i])

        quality = RandomShapeletTransform._find_shapelet_quality(
            X,
            y,
            shapelet,
            sorted_indicies,
            position,
            length,
            dim,
            inst_idx,
            self._class_counts[cls_idx],
            self.n_instances - self._class_counts[cls_idx],
            worst_quality,
        )

        return quality, length, position, dim, inst_idx, cls_idx

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _find_shapelet_quality(
        X,
        y,
        shapelet,
        sorted_indicies,
        position,
        length,
        dim,
        inst_idx,
        this_cls_count,
        other_cls_count,
        worst_quality,
    ):
        orderline = []
        this_cls_traversed = 0
        other_cls_traversed = 0

        for i, series in enumerate(X):
            if i != inst_idx:
                distance = _online_shapelet_distance(
                    series[dim], shapelet, sorted_indicies, position, length
                )
            else:
                distance = 0

            if y[i] == y[inst_idx]:
                cls = 1
                this_cls_traversed += 1
            else:
                cls = -1
                other_cls_traversed += 1

            orderline.append((distance, cls))
            orderline.sort()

            if worst_quality > 0:
                quality = _calc_early_binary_ig(
                    orderline,
                    this_cls_traversed,
                    other_cls_traversed,
                    this_cls_count - this_cls_traversed,
                    other_cls_count - other_cls_traversed,
                    worst_quality,
                )

                if quality <= worst_quality:
                    return -1

        quality = _calc_binary_ig(orderline, this_cls_count, other_cls_count)

        return round(quality, 12)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _merge_shapelets(
        shapelet_heap, candidate_shapelets, max_shapelets_per_class, cls_idx
    ):
        for shapelet in candidate_shapelets:
            if shapelet[5] == cls_idx and shapelet[0] > 0:
                if (
                    len(shapelet_heap) == max_shapelets_per_class
                    and shapelet[0] < shapelet_heap[0][0]
                ):
                    continue

                heapq.heappush(shapelet_heap, shapelet)

                if len(shapelet_heap) > max_shapelets_per_class:
                    heapq.heappop(shapelet_heap)

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _remove_self_similar_shapelets(shapelet_heap):
        to_keep = [True] * len(shapelet_heap)

        for i in range(len(shapelet_heap)):
            if to_keep[i] is False:
                continue

            for n in range(i + 1, len(shapelet_heap)):
                if to_keep[n] and _is_self_similar(shapelet_heap[i], shapelet_heap[n]):
                    if (shapelet_heap[i][0], shapelet_heap[i][1]) >= (
                        shapelet_heap[n][0],
                        shapelet_heap[n][1],
                    ):
                        to_keep[n] = False
                    else:
                        to_keep[i] = False
                        break

        return to_keep

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _remove_identical_shapelets(shapelets):
        to_keep = [True] * len(shapelets)

        for i in range(len(shapelets)):
            for n in range(i + 1, len(shapelets)):
                if (
                    to_keep[n]
                    and shapelets[i][1] == shapelets[n][1]
                    and np.array_equal(shapelets[i][6], shapelets[n][6])
                ):
                    to_keep[n] = False

        return to_keep


@njit(fastmath=True, cache=True)
def _z_norm(shapelet):
    std = np.std(shapelet)
    if std > 0:
        shapelet = (shapelet - np.mean(shapelet)) / std
    else:
        shapelet = np.zeros(len(shapelet))

    return shapelet


@njit(fastmath=True, cache=True)
def _online_shapelet_distance(series, shapelet, sorted_indicies, position, length):
    subseq = series[position : position + length]

    sum = 0.0
    sum2 = 0.0
    for i in subseq:
        sum += i
        sum2 += i * i

    mean = sum / length
    std = (sum2 - mean * mean * length) / length
    if std > 0:
        subseq = (subseq - mean) / std
    else:
        subseq = np.zeros(length)

    best_dist = 0
    for i, n in zip(shapelet, subseq):
        temp = i - n
        best_dist += temp * temp

    i = 1
    traverse = [True, True]
    sums = [sum, sum]
    sums2 = [sum2, sum2]

    while traverse[0] or traverse[1]:
        for n in range(2):
            mod = -1 if n == 0 else 1
            pos = position + mod * i
            traverse[n] = pos >= 0 if n == 0 else pos <= len(series) - length

            if not traverse[n]:
                continue

            start = series[pos - n]
            end = series[pos - n + length]

            sums[n] += mod * end - mod * start
            sums2[n] += mod * end * end - mod * start * start

            mean = sums[n] / length
            std = math.sqrt((sums2[n] - mean * mean * length) / length)

            dist = 0
            use_std = std != 0
            for j in range(length):
                val = (series[pos + sorted_indicies[j]] - mean) / std if use_std else 0
                temp = shapelet[sorted_indicies[j]] - val
                dist += temp * temp

                if dist > best_dist:
                    break

            if dist < best_dist:
                best_dist = dist

        i += 1

    return best_dist if best_dist == 0 else 1 / length * best_dist


@njit(fastmath=True, cache=True)
def _calc_early_binary_ig(
    orderline,
    c1_traversed,
    c2_traversed,
    c1_to_add,
    c2_to_add,
    worst_quality,
):
    initial_ent = _binary_entropy(
        c1_traversed + c1_to_add,
        c2_traversed + c2_to_add,
    )

    total_all = c1_traversed + c2_traversed + c1_to_add + c2_to_add

    bsf_ig = 0
    # actual observations in orderline
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        # optimistically add this class to left side first and other to right
        left_prop = (split + 1 + c1_to_add) / total_all
        ent_left = _binary_entropy(c1_count + c1_to_add, c2_count)

        right_prop = 1 - left_prop  # because right side must
        # optimistically contain everything else
        ent_right = _binary_entropy(
            c1_traversed - c1_count,
            c2_traversed - c2_count + c2_to_add,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        # now optimistically add this class to right, other to left
        left_prop = (split + 1 + c2_to_add) / total_all
        ent_left = _binary_entropy(c1_count, c2_count + c2_to_add)

        right_prop = 1 - left_prop  # because right side must
        # optimistically contain everything else
        ent_right = _binary_entropy(
            c1_traversed - c1_count + c1_to_add,
            c2_traversed - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        if bsf_ig > worst_quality:
            return bsf_ig

    return bsf_ig


@njit(fastmath=True, cache=True)
def _calc_binary_ig(orderline, c1, c2):
    initial_ent = _binary_entropy(c1, c2)

    total_all = c1 + c2

    bsf_ig = 0
    c1_count = 0
    c2_count = 0

    # evaluate each split point
    for split in range(len(orderline)):
        next_class = orderline[split][1]  # +1 if this class, -1 if other
        if next_class > 0:
            c1_count += 1
        else:
            c2_count += 1

        left_prop = (split + 1) / total_all
        ent_left = _binary_entropy(c1_count, c2_count)

        right_prop = 1 - left_prop
        ent_right = _binary_entropy(
            c1 - c1_count,
            c2 - c2_count,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

    return bsf_ig


@njit(fastmath=True, cache=True)
def _binary_entropy(c1, c2):
    ent = 0
    if c1 != 0:
        ent -= c1 / (c1 + c2) * np.log2(c1 / (c1 + c2))
    if c2 != 0:
        ent -= c2 / (c1 + c2) * np.log2(c2 / (c1 + c2))
    return ent


@njit(fastmath=True, cache=True)
def _is_self_similar(s1, s2):
    # not self similar if from different series or dimension
    if s1[4] == s2[4] and s1[3] == s2[3]:
        if s1[2] >= s2[2] and s1[2] <= s2[2] + s2[1]:
            return True
        if s2[2] >= s1[2] and s2[2] <= s1[2] + s1[1]:
            return True

    return False
