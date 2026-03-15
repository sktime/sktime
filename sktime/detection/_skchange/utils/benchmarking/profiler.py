"""Utilities to profile code."""

import cProfile
import io
import pstats
from pstats import SortKey


class Profiler:
    """Profiler class to profile code."""

    def __init__(self):
        self.pr = cProfile.Profile()
        pass

    def start(self):
        """Start profiling."""
        self.pr.enable()
        return self

    def stop(self):
        """Stop profiling and print sorted results."""
        self.pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(self.pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())  # noqa
