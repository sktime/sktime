# -*- coding: utf-8 -*-

__author__ = ["MatthewMiddlehurst", "Jason Lines", "David Guijo"]
__all__ = ["ShapeletTransform"]

import heapq
import math
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit, NumbaPendingDeprecationWarning
from sklearn.utils import check_random_state

from sktime.transformations.base import _PanelToTabularTransformer
from sktime.utils.validation import check_n_jobs
from sktime.utils.validation.panel import check_X_y, check_X


class ShapeletTransform(_PanelToTabularTransformer):
    def __init__(
        self,
        n_shapelets_considered=50000,
        max_shapelets=1000,
        min_shapelet_length=3,
        max_shapelet_length=None,
        remove_self_similar=True,
        time_limit_in_minutes=0.0,
        contract_max_n_shapelets_considered=np.inf,
        n_jobs=1,
        batch_size=None,
        random_state=None,
    ):
        self.n_shapelet_samples = n_shapelets_considered
        self.max_shapelets = max_shapelets
        self.min_shapelet_length = min_shapelet_length
        self.max_shapelet_length = max_shapelet_length
        self.remove_self_similar = remove_self_similar

        self.time_limit_in_minutes = time_limit_in_minutes
        self.contract_max_n_shapelet_samples = contract_max_n_shapelets_considered

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

        self._max_shapelet_length = max_shapelet_length
        self._n_jobs = n_jobs
        self._batch_size = batch_size
        self._class_counts = []
        self._class_dictionary = {}
        self._sorted_indicies = []

        super(ShapeletTransform, self).__init__()

    def fit(self, X, y):
        X, y = check_X_y(X, y, enforce_univariate=False, coerce_to_numpy=True)

        self._n_jobs = check_n_jobs(self.n_jobs)

        self.classes_, self._class_counts = np.unique(y, return_counts=True)
        self.n_classes = self.classes_.shape[0]
        for index, classVal in enumerate(self.classes_):
            self._class_dictionary[classVal] = index

        self.n_instances, self.n_dims, self.series_length = X.shape

        if self.batch_size is None:
            self._batch_size = max(self.n_instances, self.max_shapelets)

        if self.max_shapelet_length is None:
            self._max_shapelet_length = self.series_length

        rng = check_random_state(self.random_state)
        indicies = np.arange(self.n_instances)
        rng.shuffle(indicies)
        X = X[indicies]
        y = y[indicies]

        time_limit = self.time_limit_in_minutes * 60
        start_time = time.time()
        fit_time = 0

        max_shapelets_per_class = self.max_shapelets / self.n_classes
        shapelets = [[(-1.0, -1, -1, -1, -1)] for _ in range(self.n_classes)]
        n_shapelets_extracted = 0

        # this is a few versions away currently, and heaps dont support the replacement
        warnings.simplefilter("ignore", category=NumbaPendingDeprecationWarning)

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
                    ShapeletTransform._merge_shapelets(
                        heap,
                        candidate_shapelets,
                        max_shapelets_per_class,
                        i,
                    )

                if self.remove_self_similar:
                    for i, heap in enumerate(shapelets):
                        to_keep = ShapeletTransform._remove_self_similar_shapelets(heap)
                        shapelets[i] = [n for (n, b) in zip(heap, to_keep) if b]

                n_shapelets_extracted += self._batch_size
                fit_time = time.time() - start_time
        else:
            while n_shapelets_extracted < self.n_shapelet_samples:
                n_shapelets_to_extract = (
                    self._batch_size
                    if n_shapelets_extracted + self._batch_size
                    <= self.n_shapelet_samples
                    else self.n_shapelet_samples - n_shapelets_extracted
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
                    ShapeletTransform._merge_shapelets(
                        heap,
                        candidate_shapelets,
                        max_shapelets_per_class,
                        i,
                    )

                if self.remove_self_similar:
                    for i, heap in enumerate(shapelets):
                        to_keep = ShapeletTransform._remove_self_similar_shapelets(heap)
                        shapelets[i] = [n for (n, b) in zip(heap, to_keep) if b]

                n_shapelets_extracted += n_shapelets_to_extract

        self.shapelets = [
            (
                s[0],
                s[1],
                s[2],
                s[3],
                self.classes_[s[4]],
                _z_norm(X[s[3], 0, s[2] : s[2] + s[1]]),
            )
            for class_shapelets in shapelets
            for s in class_shapelets
        ]
        self.shapelets = [shapelet for shapelet in self.shapelets if shapelet[0] > 0]
        self.shapelets.sort()

        self._sorted_indicies = []
        for s in self.shapelets:
            sabs = np.abs(s[5])
            self._sorted_indicies.append(
                sorted(range(s[1]), reverse=True, key=lambda i: sabs[i])
            )

        self._is_fitted = True
        return self

    def transform(self, X, y=None):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)

        output = np.zeros((len(X), len(self.shapelets)))

        for i, series in enumerate(X):
            for n, shapelet in enumerate(self.shapelets):
                output[i][n] = _online_shapelet_distance(
                    series[0],
                    shapelet[5],
                    self._sorted_indicies[n],
                    shapelet[2],
                    shapelet[1],
                )

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

        shapelet = _z_norm(X[inst_idx, 0, position : position + length])
        sabs = np.abs(shapelet)
        sorted_indicies = sorted(range(length), reverse=True, key=lambda i: sabs[i])

        quality = ShapeletTransform._find_shapelet_quality(
            X,
            y,
            shapelet,
            sorted_indicies,
            position,
            length,
            inst_idx,
            self._class_counts[cls_idx],
            self.n_instances - self._class_counts[cls_idx],
            worst_quality,
        )

        return quality, length, position, inst_idx, cls_idx

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _find_shapelet_quality(
        X,
        y,
        shapelet,
        sorted_indicies,
        position,
        length,
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
                    series[0], shapelet, sorted_indicies, position, length
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

        return quality

    @staticmethod
    @njit(fastmath=True, cache=True)
    def _merge_shapelets(
        shapelet_heap, candidate_shapelets, max_shapelets_per_class, cls_idx
    ):
        for shapelet in candidate_shapelets:
            if shapelet[4] == cls_idx and shapelet[0] > 0:
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
    # not self similar if from different series
    if s1[3] != s2[3]:
        return False

    if s1[2] >= s2[2] and s1[2] <= s2[2] + s2[1]:
        return True
    if s2[2] >= s1[2] and s2[2] <= s1[2] + s1[1]:
        return True

    return False
