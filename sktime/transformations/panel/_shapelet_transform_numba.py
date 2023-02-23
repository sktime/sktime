# -*- coding: utf-8 -*-
"""Isolated numba imports for shapelet."""

__author__ = ["MatthewMiddlehurst", "jasonlines", "dguijo"]

import heapq
import math

import numpy as np

from sktime.utils.numba.njit import njit


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

        # because right side must optimistically contain everything else
        right_prop = 1 - left_prop

        ent_right = _binary_entropy(
            c1_traversed - c1_count,
            c2_traversed - c2_count + c2_to_add,
        )

        ig = initial_ent - left_prop * ent_left - right_prop * ent_right
        bsf_ig = max(ig, bsf_ig)

        # now optimistically add this class to right, other to left
        left_prop = (split + 1 + c2_to_add) / total_all
        ent_left = _binary_entropy(c1_count, c2_count + c2_to_add)

        # because right side must optimistically contain everything else
        right_prop = 1 - left_prop

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
        if s2[2] <= s1[2] <= s2[2] + s2[1]:
            return True
        if s1[2] <= s2[2] <= s1[2] + s1[1]:
            return True

    return False


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
    # todo optimise this more, we spend 99% of time here
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
