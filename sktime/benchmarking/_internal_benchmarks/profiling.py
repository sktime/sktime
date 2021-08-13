# -*- coding: utf-8 -*-
import cProfile
import pstats
from typing import Callable, Any
import timeit


def time_function_call(
    function_to_time: Callable, average_amount: int = 100, kwargs: Any = None
):
    def timeit_experiments():
        function_to_time(kwargs.get("x"), kwargs.get("y"))

    result = timeit.timeit(timeit_experiments, number=average_amount)

    return result / average_amount


def profile_a_function(
    function_to_profile: Callable,
    pstats_sort_by: pstats.SortKey = pstats.SortKey.TIME,
    output_file_path: str = None,
    print_stats: bool = False,
    kwargs: Any = None,
) -> pstats.Stats:
    """

    Parameters
    ----------
    function_to_profile: Callable
        function you want to profile

    pstats_sort_by: pstats.SortKey, defaults = pstats.SortKey.TIME
        what to sort that stats returned by

    output_file_path: str, defaults = None
        path to write file with profile results in. If not specified no file will be
        written

    print_stats: bool, defaults = False
        boolean when True will print the stats to console and when False will not print
        the stats to console.

    kwargs: Any
        kwargs should contain the arguments for your function your are parsing

    Returns
    -------
        pstats.Stats generated from running the function
    """

    with cProfile.Profile() as pr:
        function_to_profile(kwargs.get("x"), kwargs.get("y"))

    stats = pstats.Stats(pr)

    stats.sort_stats(pstats_sort_by)

    if print_stats:
        stats.print_stats()

    if output_file_path is not None:
        stats.dump_stats(filename=output_file_path)

    return stats
