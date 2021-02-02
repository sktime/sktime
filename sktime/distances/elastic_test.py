# content of test_time.py

import pytest
import csv

from sktime.distances.elastic import dtw_distance


def test_distance_measure():

    dist_func = dtw_distance
    with open("test_data/dtw_distance_gunpoint.csv") as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        reader.__next__() # discard header
        for row in reader:
            inst_index_a = int(row[0])
            inst_index_b = int(row[1])
            distance = int(row[2])
            params = row[3]


    pass

# from datetime import datetime, timedelta
#
# testdata = [
#     (datetime(2001, 12, 12), datetime(2001, 12, 11), timedelta(1)),
#     (datetime(2001, 12, 11), datetime(2001, 12, 12), timedelta(-1)),
# ]
#
# def idfn(val):
#     if isinstance(val, (datetime,)):
#         # note this wouldn't show any hours/minutes/seconds
#         return val.strftime("%Y%m%d")
#
#
# @pytest.mark.parametrize("a,b,expected", testdata, ids=idfn)
# def test_timedistance_v2(a, b, expected):
#     diff = a - b
#     assert diff == expected


