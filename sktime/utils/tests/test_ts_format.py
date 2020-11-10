import numpy as np
import pytest
import os
import tempfile
from sktime.utils.ts_format import csv_to_ts_format

def test_csv_to_ts_format():
    print("\n\n")
    # Testing a single dimension, no class label dataset
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, 'w') as tmp_file:
            file_contents = 'test1,test2,test3\n\
                    3,0,1\n\
                    3,1,2\n\
                    3,1,1\n\
                    8,0,1'
            tmp_file.write(file_contents)
            tmp_file.flush()
        csv_to_ts_format("test", path, "./tst", 'test1')
        f = open("./tst.ts", "r")
        result = f.read()
        f.close()
        assert result == "@problamName test\n\
@timestamps false\n\
@missing false\n\
@univariate false\n\
@equalLength false\n\
@seriesLength 3\n\
@dimensions 2\n\
@classLabel false\n\
@data\n\
0:011\n\
1:121"
        os.remove("./tst.ts")
    finally:
        os.remove(path)

    # Testing a 2 dimension, with class label dataset
    # fd, path = tempfile.mkstemp()
    # try:
    #     with os.fdopen(fd, 'w') as tmp_file:
    #         file_contents = 'test1,test2,test3\n\
    #                 3,0,1\n\
    #                 3,1,2\n\
    #                 3,1,1\n\
    #                 8,0,1'
    #         tmp_file.write(file_contents)
    #         tmp_file.flush()
    #     result = csv_to_ts_format("test", path, "./tst", 'test1', 'test1')
    #     assert result['classLabel'] == 'true 8 3'
    #     assert result['timestamps'] == 'false'
    #     assert result['missing'] == 'false'
    #     assert result['univariate'] == 'false'
    #     assert result['equalLength'] == 'false'
    #     assert result['seriesLength'] == 3
    #     assert result['dimensions'] == 2
    # finally:
    #     os.remove(path)

    # # Testing a single dimension, with class label dataset
    # fd, path = tempfile.mkstemp()
    # try:
    #     with os.fdopen(fd, 'w') as tmp_file:
    #         file_contents = 'test1,test2\n\
    #                 3,0\n\
    #                 3,1\n\
    #                 3,1\n\
    #                 8,0'
    #         tmp_file.write(file_contents)
    #         tmp_file.flush()
    #     result = csv_to_ts_format("test", path, "./tst", 'test1')
    #     assert result['classLabel'] == 'false'
    #     assert result['timestamps'] == 'false'
    #     assert result['missing'] == 'false'
    #     assert result['univariate'] == 'true'
    #     assert result['equalLength'] == 'false'
    #     assert result['seriesLength'] == 3
    #     assert result['dimensions'] == 1
    # finally:
    #     os.remove(path)