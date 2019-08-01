# find the index of the best value in the array
def arg_bests(array, comparator):
    indices = [0]
    best = array[0]
    for index in range(1, len(array)):
        value = array[index]
        comparison_result = comparator(value, best)
        if comparison_result >= 0:
            if comparison_result > 0:
                indices = []
                best = value
            indices.append(index)
    return indices


# pick values from array at given indices
def pick_from_indices(array, indices):
    picked = []
    for index in indices:
        picked.append(array[index])
    return picked


# find best values in array
def bests(array, comparator):
    indices = arg_bests(array, comparator)
    return pick_from_indices(array, indices)


# find min values in array
def mins(array, getter=None):
    indices = arg_mins(array, getter)
    return pick_from_indices(array, indices)


# find max values in array
def maxs(array, getter=None):
    indices = arg_maxs(array, getter)
    return pick_from_indices(array, indices)


# find min value in array, randomly breaking ties
def min(array, rand, getter=None):
    index = arg_min(array, rand, getter)
    return array[index]


# find max value in array, randomly breaking ties
def max(array, rand, getter=None):
    index = arg_max(array, rand, getter)
    return array[index]


# find best value in array, randomly breaking ties
def best(array, comparator, rand):
    return rand.choice(bests(array, comparator))


# find index of best value in array, randomly breaking ties
def arg_best(array, comparator, rand):
    return rand.choice(arg_bests(array, comparator))


# find index of min value in array, randomly breaking ties
def arg_min(array, rand, getter=None):
    return rand.choice(arg_mins(array, getter))


# find indices of best value in array, randomly breaking ties
def arg_mins(array, getter=None):
    return arg_bests(array, chain(less_than, getter))


# find index of max value in array, randomly breaking ties
def arg_max(array, rand, getter=None):
    return rand.choice(arg_maxs(array, getter))


# find indices of max value in array, randomly breaking ties
def arg_maxs(array, getter=None):
    return arg_bests(array, chain(more_than, getter))


# obtain a value before using in func
def chain(func, getter=None):
    if getter is None:
        return func
    else:
        return lambda a, b: func(getter(a), getter(b))


# test if a is more than b
def more_than(a, b):
    if a < b:
        return -1
    elif a > b:
        return 1
    else:
        return 0


# test if a is less than b
def less_than(a, b):
    if a < b:
        return 1
    elif a > b:
        return -1
    else:
        return 0
