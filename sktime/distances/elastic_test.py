# -*- coding: utf-8 -*-
# content of test_time.py
# import ast
import os

# from datetime import datetime
import json

import pytest
import csv

from sklearn.utils import check_random_state
from sktime.distances.elastic_params import (
    build_dtw_distance_params,
    build_wdtw_distance_params,
    build_msm_distance_params,
    # build_erp_distance_params,
    # build_lcss_distance_params,
    build_twe_distance_params,
)

from sktime.datasets.base import _load_dataset

# don't have twed in python form
from sktime.distances.elastic import (
    dtw_distance,
    # derivative_dtw_distance,
    # ddtw_distance,
    wdtw_distance,
    wddtw_distance,
    # lcss_distance,
    msm_distance,
    # erp_distance,
    # e_distance,
)

# define several functions importing the cython versions of each distance measure.
# this is necessary because the python function names and cython function names clash.
# therefore these functions simply append "_cython" to the function as an alias for
# the cython based implementations.

from sktime.utils.data_processing import from_nested_to_3d_numpy


def dtw_distance_cython(a, b, **kwargs):
    from sktime.distances.elastic_cython import dtw_distance

    return dtw_distance(a, b, **kwargs)


def ddtw_distance_cython(a, b, **kwargs):
    from sktime.distances.elastic_cython import ddtw_distance

    return ddtw_distance(a, b, **kwargs)


def wdtw_distance_cython(a, b, **kwargs):
    from sktime.distances.elastic_cython import wdtw_distance

    return wdtw_distance(a, b, **kwargs)


def wddtw_distance_cython(a, b, **kwargs):
    from sktime.distances.elastic_cython import wddtw_distance

    return wddtw_distance(a, b, **kwargs)


def msm_distance_cython(a, b, **kwargs):
    from sktime.distances.elastic_cython import msm_distance

    return msm_distance(a, b, **kwargs)


def erp_distance_cython(a, b, **kwargs):
    from sktime.distances.elastic_cython import erp_distance

    return erp_distance(a, b, **kwargs)


def lcss_distance_cython(a, b, **kwargs):
    from sktime.distances.elastic_cython import lcss_distance

    return lcss_distance(a, b, **kwargs)


def twe_distance_cython(a, b, **kwargs):
    from sktime.distances.elastic_cython import twe_distance

    return twe_distance(a, b, **kwargs)


@pytest.mark.parametrize("dataset_name", ["GunPoint", "ItalyPowerDemand"])
@pytest.mark.parametrize("dist_func", [dtw_distance, dtw_distance_cython])
def test_dist_func(dist_func, dataset_name):
    dist_func_name = dist_func.__name__
    test_data_file_path = dataset_name + "/" + dist_func_name + ".csv"
    # load the data
    dataframe, y = _load_dataset(dataset_name, split=None, return_X_y=True)
    data = from_nested_to_3d_numpy(dataframe)
    # open the csv file
    path = __test_results_dir_name + "/" + test_data_file_path
    if not os.path.isfile(path):
        # todo make this fail tests in the future as we expect there to be test results
        #  for every distance measure
        # temporarily disabling this for now while developing distance tests
        assert True
        # assert False, " ".join([path, "does not exist"])
    else:
        with open(path) as csv_file:
            reader = csv.reader(csv_file, delimiter=",")
            reader.__next__()  # discard header
            for row in reader:
                # each row contains the inst index x2, the pre-computed distance and
                # any params for the distance measure
                inst_a_index = int(row[0])
                inst_b_index = int(row[1])
                pre_computed_distance = float(row[2])
                params = json.loads(row[3])
                # find the corresponding insts
                inst_a = data[inst_a_index]
                inst_b = data[inst_b_index]
                # compute the distance
                distance = dist_func(inst_a, inst_b, **params)
                assert pre_computed_distance == distance, " ".join(
                    [
                        "distance mismatch for",
                        dist_func_name,
                        "using",
                        str(params),
                        "on instance",
                        str(inst_a_index),
                        "and",
                        str(inst_b_index),
                        "of",
                        dataset_name,
                    ]
                )


__test_results_dir_name = "test_results"


def create_distance_measure_test_results(
    dist_func, param_space, dataset_name, rand=0, n_distances=100
):
    # start = datetime.now()
    rand = check_random_state(rand)
    dist_func_name = dist_func.__name__
    # print("computing distances for", dist_func_name, "on", dataset_name)
    # mkdir on the test results dir and sub folder for dataset
    os.makedirs(__test_results_dir_name + "/" + dataset_name, exist_ok=True)
    test_data_file_path = dataset_name + "/" + dist_func_name + ".csv"
    # load the data
    dataframe, y = _load_dataset(dataset_name, split=None, return_X_y=True)
    data = from_nested_to_3d_numpy(dataframe)
    # setup the param space. the param space should be a dict or a function which
    # accepts the data to produce a dict
    if callable(param_space):
        param_space = param_space(data)
    # open the csv file
    f = open(__test_results_dir_name + "/" + test_data_file_path, "w")
    # write the header
    f.write(",".join(["i", "j", "distance", "params", "\n"]))
    # for each distance calculation
    for _k in range(0, n_distances):
        # pick an inst
        inst_a_index = rand.choice(len(data))
        inst_a = data[inst_a_index]
        # pick another inst that is NOT the same as inst_a
        inst_b_index = inst_a_index
        while inst_a_index == inst_b_index:
            inst_b_index = rand.choice(len(data))
        inst_b = data[inst_b_index]
        # choose parameters for the distance measure
        param_set = choose_param_set(param_space, rand)
        # compute distance
        distance = dist_func(inst_a, inst_b, **param_set)
        # write the distance to file with corresponding info about which insts and
        # parameters were used
        line = ",".join(
            [str(inst_a_index), str(inst_b_index), str(distance), json.dumps(param_set)]
        )
        f.write(line + "\n")
        # print(line)
    # clean up
    f.close()
    # print(
    # f"computed distances for {dist_func_name}on  {dataset_name} in "
    # f"{(datetime.now() - start).total_seconds()} seconds"
    # )


def choose_param_set(param_space, rand):
    rand = check_random_state(rand)
    # randomly select a sub param space
    if isinstance(param_space, dict):
        sub_param_space = param_space
    elif hasattr(param_space, "__len__"):
        sub_param_space_index = rand.choice(len(param_space))
        sub_param_space = param_space[sub_param_space_index]
    else:
        raise ValueError("expected list or dict")
    # sub_param_space is a dict containing key value pairs. The values may be
    # further param spaces though! go through each key in the dict
    param_set = {}
    for key, value in sub_param_space.items():
        if hasattr(value, "rvs"):
            # value is a dict
            # pick a rand value for every key in the dict
            # each key currently maps to a distribution or list / array of values
            choice = value.rvs(random_state=rand)
        elif hasattr(value, "__len__"):
            # discrete list of possible values to be chosen from
            choice = value[rand.randint(len(value))]
            if isinstance(choice, dict):
                # the chosen value is a sub param space, so recurse in
                choice = choose_param_set(choice, rand)
        else:
            raise ValueError("param is neither list-like nor a distribution")
        # record the mapping of chosen param value
        param_set[key] = choice
    return param_set


if __name__ == "__main__":

    # erp_distance_cython(np.array([[1, 2, 3], [4, 5, 6]]), np.array([[7, 8, 9],
    # [10, 11, 12]]))

    for dataset in [
        "GunPoint",
        "ItalyPowerDemand",
        # "BasicMotions",
        # "Beef"
    ]:
        create_distance_measure_test_results(
            dist_func=dtw_distance,
            param_space=build_dtw_distance_params,
            dataset_name="GunPoint",
        )
        create_distance_measure_test_results(
            dist_func=dtw_distance_cython,
            param_space=build_dtw_distance_params,
            dataset_name=dataset,
        )
        create_distance_measure_test_results(
            dist_func=ddtw_distance_cython,
            param_space=build_dtw_distance_params,
            dataset_name=dataset,
        )
        create_distance_measure_test_results(
            dist_func=wdtw_distance,
            param_space=build_wdtw_distance_params,
            dataset_name=dataset,
        )
        create_distance_measure_test_results(
            dist_func=wdtw_distance_cython,
            param_space=build_wdtw_distance_params,
            dataset_name=dataset,
        )
        create_distance_measure_test_results(
            dist_func=wddtw_distance,
            param_space=build_wdtw_distance_params,
            dataset_name=dataset,
        )
        create_distance_measure_test_results(
            dist_func=wddtw_distance_cython,
            param_space=build_wdtw_distance_params,
            dataset_name=dataset,
        )
        create_distance_measure_test_results(
            dist_func=msm_distance,
            param_space=build_msm_distance_params,
            dataset_name=dataset,
        )
        create_distance_measure_test_results(
            dist_func=msm_distance_cython,
            param_space=build_msm_distance_params,
            dataset_name=dataset,
        )
        # don't have twed in python form
        create_distance_measure_test_results(
            dist_func=twe_distance_cython,
            param_space=build_twe_distance_params,
            dataset_name=dataset,
        )
