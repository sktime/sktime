# -*- coding: utf-8 -*-
import os
import tempfile
from sktime.utils.ts_format import csv_to_ts_format


def test_csv_to_ts_format():
    # Testing a single dimension, no class label
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            file_contents = "test1,test2,test3\n\
                    3,0,1\n\
                    3,1,2\n\
                    3,1,1\n\
                    8,0,1"
            tmp_file.write(file_contents)
            tmp_file.flush()
        csv_to_ts_format("test", path, "./tst", "test1")
        f = open("./tst.ts", "r")
        result = f.read()
        f.close()
        assert (
            result
            == "@problamName test\n\
@timestamps false\n\
@missing false\n\
@univariate false\n\
@equalLength false\n\
@seriesLength 3\n\
@dimensions 2\n\
@classLabel false\n\
@data\n\
0:1\n\
011:121"
        )
    finally:
        os.remove("./tst.ts")
        os.remove(path)

    # Testing a 2 dimension, with class label
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            file_contents = "test1,test2,test3\n\
                    3,0,1\n\
                    3,1,2\n\
                    3,1,1\n\
                    8,0,1"
            tmp_file.write(file_contents)
            tmp_file.flush()
        csv_to_ts_format("test", path, "./tst", "test1", "test1")
        f = open("./tst.ts", "r")
        result = f.read()
        f.close()
        assert (
            result
            == "@problamName test\n\
@timestamps false\n\
@missing false\n\
@univariate false\n\
@equalLength false\n\
@seriesLength 3\n\
@dimensions 2\n\
@classLabel true 8 3\n\
@data\n\
0:1:8\n\
011:121:3"
        )
    finally:
        os.remove("./tst.ts")
        os.remove(path)

    # Testing a single dimension (univariate), without class label
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            file_contents = "test1,test2\n\
                    3,0\n\
                    3,1\n\
                    3,1\n\
                    8,0"
            tmp_file.write(file_contents)
            tmp_file.flush()
        csv_to_ts_format("test", path, "./tst", "test1")
        f = open("./tst.ts", "r")
        result = f.read()
        f.close()
        assert (
            result
            == "@problamName test\n\
@timestamps false\n\
@missing false\n\
@univariate true\n\
@equalLength false\n\
@seriesLength 3\n\
@dimensions 1\n\
@classLabel false\n\
@data\n\
0\n\
011"
        )
    finally:
        os.remove("./tst.ts")
        os.remove(path)

    # Testing a single dimension (univariate), with class label
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            file_contents = "test1,test2\n\
                    3,0\n\
                    3,1\n\
                    3,1\n\
                    8,0"
            tmp_file.write(file_contents)
            tmp_file.flush()
        csv_to_ts_format("test", path, "./tst", "test1", "test1")
        f = open("./tst.ts", "r")
        result = f.read()
        f.close()
        assert (
            result
            == "@problamName test\n\
@timestamps false\n\
@missing false\n\
@univariate true\n\
@equalLength false\n\
@seriesLength 3\n\
@dimensions 1\n\
@classLabel true 8 3\n\
@data\n\
0:8\n\
011:3"
        )
    finally:
        os.remove("./tst.ts")
        os.remove(path)

    # Testing a multiple dimension of equal length, with class label
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            file_contents = "test1,test2,test3\n\
                    3,0,1\n\
                    3,1,2\n\
                    3,1,1\n\
                    8,0,1\n\
                    8,0,1\n\
                    8,0,1"
            tmp_file.write(file_contents)
            tmp_file.flush()
        csv_to_ts_format("test", path, "./tst", "test1", "test1")
        f = open("./tst.ts", "r")
        result = f.read()
        f.close()
        assert (
            result
            == "@problamName test\n\
@timestamps false\n\
@missing false\n\
@univariate false\n\
@equalLength true\n\
@seriesLength 3\n\
@dimensions 2\n\
@classLabel true 8 3\n\
@data\n\
000:111:8\n\
011:121:3"
        )
    finally:
        os.remove("./tst.ts")
        os.remove(path)

    # Testing a multiple dimension of equal length, with missing value and class label
    fd, path = tempfile.mkstemp()
    try:
        with os.fdopen(fd, "w") as tmp_file:
            file_contents = "test1,test2,test3\n\
                    3,0,1\n\
                    3,?,2\n\
                    3,?,1\n\
                    8,?,?\n\
                    8,0,1\n\
                    8,0,1"
            tmp_file.write(file_contents)
            tmp_file.flush()
        csv_to_ts_format("test", path, "./tst", "test1", "test1")
        f = open("./tst.ts", "r")
        result = f.read()
        f.close()
        assert (
            result
            == "@problamName test\n\
@timestamps false\n\
@missing true\n\
@univariate false\n\
@equalLength true\n\
@seriesLength 3\n\
@dimensions 2\n\
@classLabel true 8 3\n\
@data\n\
?00:?11:8\n\
0??:121:3"
        )
    finally:
        os.remove("./tst.ts")
        os.remove(path)


test_csv_to_ts_format()
