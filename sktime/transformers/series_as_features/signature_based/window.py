"""
window.py
======================================
Introduces the Window module that is used when splitting the path over:
    - Global
    - Sliding
    - Expanding
    - Dyadic
window types.
"""
import collections as co


_Pair = co.namedtuple('Pair', ('start', 'end'))


def window_getter(string, **kwargs):
    """Gets the window method correspondent to the given string

    Args:
        string (str): String such that string.title() corresponds to a window method.
        *args: Arguments that will be supplied to the window method.

    Returns:
        Window: An initialised Window method.
    """
    return globals()[string](**kwargs)


class Window:
    """Abstract base class for windows.

    Each subclass should implement __call__, which returns a list of list of 2-tuples. Each 2-tuple specifies the start
    and end of a window. These are then grouped together into a list, and these lists are then grouped together again
    into another list. (Really for the sake of the Dyadic window, which considers windows at multiple scales, so the
    different scales of windows should be grouped together but not grouped with each other.)
    """
    def __init__(self):
        if self.__class__ is Window:
            raise NotImplementedError  # abstract base class

    def num_windows(self, length):
        """ Gets the total number of windows produced by the given window method.

        Args:
            length (int): Length of the time series.

        Returns:
            int: Number of windows.
        """
        all_windows = self(length)
        num_windows = 0
        for window_group in all_windows:
            num_windows += len(window_group)
        return num_windows


class Global(Window):
    """A single window over the whole data."""
    def __call__(self, length=None):
        return [[_Pair(None, None)]]


class _ExpandingSliding(Window):
    def __init__(self, initial_length, start_step, end_step):
        super(_ExpandingSliding, self).__init__()
        if self.__class__ is _ExpandingSliding:
            raise NotImplementedError  # abstract base class
        self.initial_length = initial_length
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, length):
        def _call():
            start = 0
            end = self.initial_length
            while end <= length:
                yield _Pair(start, end)
                start += self.start_step
                end += self.end_step
        return [list(_call())]


class Sliding(_ExpandingSliding):
    """A window starting at zero and going to some point that increases between windows."""
    def __init__(self, length, step):
        super(Sliding, self).__init__(initial_length=length, start_step=step, end_step=step)


class Expanding(_ExpandingSliding):
    """A window of fixed length, slid along the dataset."""
    def __init__(self, length, step):
        super(Expanding, self).__init__(initial_length=length, start_step=0, end_step=step)


class Dyadic(Window):
    """First the global window over the whole thing. Then the window of the first half and the second. Then the window
    over the first quarter, then the second quarter, then the third quarter, then the fourth quarter, etc. down to
    some specified depth. Make sure the depth isn't too high for the length of the dataset, lest we end up with trivial
    windows that we can't compute a signature over.
    """
    def __init__(self, depth):
        super(Dyadic, self).__init__()
        self.depth = depth

    def __call__(self, length):
        return self.call(float(length))

    def call(self, length, _offset=0.0, _depth=0, _out=None):
        if _out is None:
            _out = [[] for _ in range(self.depth + 1)]
        _out[_depth].append(_Pair(int(_offset), int(_offset + length)))

        if _depth < self.depth:
            left = Dyadic(self.depth)
            right = Dyadic(self.depth)
            half_length = length / 2
            # The order of left then right is important, so that they add their entries to _out in the correct order.
            left.call(half_length, _offset, _depth + 1, _out)
            right.call(half_length, _offset + half_length, _depth + 1, _out)

        return _out

