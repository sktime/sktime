import numpy as np
import sys


def boss_distance(first, second, best_dist=sys.float_info.max):
    dist = 0

    if isinstance(first, dict):
        for word, val_a in first.items():
            val_b = second.get(word, 0)
            dist += (val_a - val_b) * (val_a - val_b)

            if dist > best_dist:
                return sys.float_info.max
    else:
        dist = np.sum([0 if first[n] == 0 else (first[n] - second[n]) * (first[n] - second[n])
                       for n in range(len(first))])

    return dist


def euclidean_distance(first, second, best_dist=sys.float_info.max):
    dist = 0

    if isinstance(first, dict):
        words = set(list(first) + list(second))
        for word in words:
            val_a = first.get(word, 0)
            val_b = second.get(word, 0)
            dist += (val_a - val_b) * (val_a - val_b)

            if dist > best_dist:
                return sys.float_info.max
    else:
        dist = np.sum([(first[n] - second[n]) * (first[n] - second[n]) for n in range(len(first))])

    return dist
