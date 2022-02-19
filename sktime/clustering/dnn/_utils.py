# -*- coding: utf-8 -*-
"""
tools used for the experiments
Based on Hassan Fawaz implementation https://github.com/hfawaz/dl-4-tsc
Author:
Baptiste Lafabregue 2019.25.04
"""

import os
import random
from functools import partial
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tensorflow.keras.backend as K
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import dtw
from tslearn.utils import check_dims, to_sklearn_dataset, to_time_series_dataset

from sktime.datasets.base import load_UCR_UEA_dataset


def generate_fake_samples(x, rate=0.2):
    fakes = []
    for i in range(len(x)):
        choices = random.sample(range(len(x[i])), k=int(len(x[i]) * rate))
        extract = x[i][choices]
        remaining = np.delete(x[i], choices, axis=0)

        for e in extract:
            choice = random.choice(range(len(remaining)))
            remaining = np.insert(remaining, choice, e, axis=0)
        fakes.append(remaining)

    return np.array(fakes)


def read_dataset(root_dir, archive_name, dataset_name, is_train=True):
    datasets_dict = {}

    if is_train:
        type = "train"
    else:
        type = "test"

    file_name = root_dir + "/archives/" + archive_name + "/" + dataset_name + "/"
    x = np.load(file_name + "x_" + type + ".npy", allow_pickle=True)
    y = np.load(file_name + "y_" + type + ".npy", allow_pickle=True)
    y = y.astype(int)
    if len(x.shape) == 2:
        x = np.reshape(x, (x.shape[0], -1, 1))
    #     x = np.reshape(x, (x.shape[0], 1, -1))
    # else:
    #     x = np.reshape(x, (x.shape[0], x.shape[2], x.shape[1]))

    datasets_dict[dataset_name] = (x.copy(), y.copy())
    datasets_dict["k"] = len(np.unique(y))

    return datasets_dict


def read_seeds(root_dir, archive_name, dataset_name, itr):
    file_name = (
        root_dir
        + "/seeds/"
        + archive_name
        + "/"
        + dataset_name
        + "/init_clusters_not_per_class/"
    )
    seeds = np.load(file_name + "init_clusters.npy", allow_pickle=True)
    seeds = seeds[itr]
    if len(seeds.shape) == 2:
        seeds = np.reshape(seeds, (seeds.shape[0], seeds.shape[1], 1))

    return seeds


def create_output_path(root_dir, itr, framework_name, dataset_name, type="ae_weights"):
    dir = (
        root_dir
        + "/"
        + type
        + "/"
        + str(itr)
        + "/"
        + framework_name
        + "/"
        + dataset_name
        + "/"
    )
    create_directory(dir)
    return dir


def create_directory(directory_path):
    if os.path.exists(directory_path):
        return directory_path
    else:
        try:
            os.makedirs(directory_path)
        except:
            # in case another machine created the path meanwhile !:(
            return None
        return directory_path


def create_path(root_dir, classifier_name, archive_name):
    output_directory = (
        root_dir + "/results/" + classifier_name + "/" + archive_name + "/"
    )
    if os.path.exists(output_directory):
        return None
    else:
        os.makedirs(output_directory)
        return output_directory


def construct_seeds(x_train, y_train, path, nb_itr=5):
    clusters = np.unique(y_train)
    seed_set = []
    for _ in range(nb_itr):
        seeds = []
        for c in clusters:
            seeds.append(random.choice(x_train[y_train == c]))
        seed_set.append(np.array(seeds))
    np.save(path, np.array(seed_set))


def noise(code, type="uniform"):
    noise_code = np.copy(code)
    if type == "uniform":
        noise_code = noise_code + np.random.uniform(size=noise_code.shape)
    if type == "gaussian":
        mu, sigma = 0, 0.1
        noise_code = noise_code + np.random.normal(mu, sigma, noise_code.shape)
    if type == "laplace":
        noise_code = noise_code + np.random.laplace(
            loc=0.0, scale=1.0, size=noise_code.shape
        )
    elif type == "drop":
        for i in range(len(noise_code)):
            idx = np.around(
                np.random.uniform(0.0, code.shape[1] - 1, size=int(code.shape[1] * 0.2))
            ).astype(np.int)
            noise_code[i, idx] = 0
    return noise_code


def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from scipy.optimize import linear_sum_assignment

    r, c = linear_sum_assignment(w.max() - w)
    sum_i = sum([w[r[i], c[i]] for i in range(len(r))])
    return sum_i * 1.0 / y_pred.size


def computes_dtw_square_error(x, labels):
    classes = np.unique(labels)
    total_error = 0

    for cl in classes:
        indexes = labels[labels == cl]
        sub_x = x[indexes]
        card = len(sub_x)
        barycenter = dtw_barycenter_averaging(sub_x)

        for ts in sub_x:
            total_error += dtw(barycenter, ts)

    return total_error


def computes_dtw_regularized_square_error(x, labels):
    classes = np.unique(labels)
    total_error = 0
    subs_x = []
    barycenters = []

    for cl in classes:
        indexes = labels[labels == cl]
        subs_x.append(x[indexes])
        barycenters.append(dtw_barycenter_averaging(subs_x[-1]))

    for i in range(len(subs_x)):
        sub_total = 0
        for ts in subs_x[i]:
            # compute the min dist to other barycenters
            min_dist = np.inf
            for j in range(len(barycenters)):
                if i != j:
                    dist = dtw(ts, barycenters[j])
                    if dist < min_dist:
                        min_dist = dist

            dist = dtw(barycenters[i], ts)
            dist = dist / min_dist
            sub_total += dist

        sub_total /= len(subs_x[i])
        total_error += sub_total

    return total_error


def computes_dtw_silhouette_score(dist_matrix, labels):
    return silhouette_score(dist_matrix, labels, metric="precomputed")


def cdist_dtw(
    dataset1,
    dataset2=None,
    global_constraint=None,
    sakoe_chiba_radius=None,
    itakura_max_slope=None,
    n_jobs=None,
    verbose=0,
):
    r"""Compute cross-similarity matrix using Dynamic Time Warping (DTW)
    similarity measure.
    DTW is computed as the Euclidean distance between aligned time series,
    i.e., if :math:`P` is the alignment path:
    .. math::
        DTW(X, Y) = \sqrt{\sum_{(i, j) \in P} (X_{i} - Y_{j})^2}
    DTW was originally presented in [1]_.
    Parameters
    ----------
    dataset1 : array-like
        A dataset of time series
    dataset2 : array-like (default: None)
        Another dataset of time series. If `None`, self-similarity of
        `dataset1` is returned.
    global_constraint : {"itakura", "sakoe_chiba"} or None (default: None)
        Global constraint to restrict admissible paths for DTW.
    sakoe_chiba_radius : int or None (default: None)
        Radius to be used for Sakoe-Chiba band global constraint.
        If None and `global_constraint` is set to "sakoe_chiba", a radius of
        1 is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    itakura_max_slope : float or None (default: None)
        Maximum slope for the Itakura parallelogram constraint.
        If None and `global_constraint` is set to "itakura", a maximum slope
        of 2. is used.
        If both `sakoe_chiba_radius` and `itakura_max_slope` are set,
        `global_constraint` is used to infer which constraint to use among the
        two. In this case, if `global_constraint` corresponds to no global
        constraint, a `RuntimeWarning` is raised and no global constraint is
        used.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See scikit-learns'
        `Glossary <https://scikit-learn.org/stable/glossary.html#term-n-jobs>`_
        for more details.
    verbose : int, optional (default=0)
        The verbosity level: if non zero, progress messages are printed.
        Above 50, the output is sent to stdout.
        The frequency of the messages increases with the verbosity level.
        If it more than 10, all iterations are reported.
        `Glossary <https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation>`__
        for more details.
    Returns
    -------
    cdist : numpy.ndarray
        Cross-similarity matrix
    Examples
    --------
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]])
    array([[0., 1.],
           [1., 0.]])
    >>> cdist_dtw([[1, 2, 2, 3], [1., 2., 3., 4.]], [[1, 2, 3], [2, 3, 4, 5]])
    array([[0.        , 2.44948974],
           [1.        , 1.41421356]])
    See Also
    --------
    dtw : Get DTW similarity score
    References
    ----------
    .. [1] H. Sakoe, S. Chiba, "Dynamic programming algorithm optimization for
           spoken word recognition," IEEE Transactions on Acoustics, Speech and
           Signal Processing, vol. 26(1), pp. 43--49, 1978.
    """  # noqa: E501
    for i in range(len(dataset1)):
        x = dataset1[i].astype(np.float)
    dataset1 = to_time_series_dataset(dataset1)

    if dataset2 is None:
        # Inspired from code by @GillesVandewiele:
        # https://github.com/rtavenar/tslearn/pull/128#discussion_r314978479
        matrix = np.zeros((len(dataset1), len(dataset1)))
        indices = np.triu_indices(len(dataset1), k=1, m=len(dataset1))
        matrix[indices] = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
            delayed(dtw)(
                dataset1[i],
                dataset1[j],
                global_constraint=global_constraint,
                sakoe_chiba_radius=sakoe_chiba_radius,
                itakura_max_slope=itakura_max_slope,
            )
            for i in range(len(dataset1))
            for j in range(i + 1, len(dataset1))
        )
        return matrix + matrix.T
    else:
        dataset2 = to_time_series_dataset(dataset2)
        print("len d2 " + str(len(dataset2)))
        count = 0
        # matrix = Parallel(n_jobs=n_jobs, prefer="threads", verbose=verbose)(
        #     delayed(dtw_row)(
        #         dataset1[i], dataset2, i,
        #         global_constraint=global_constraint,
        #         sakoe_chiba_radius=sakoe_chiba_radius,
        #         itakura_max_slope=itakura_max_slope)
        #     for i in range(len(dataset1))
        # )
        with Pool(processes=n_jobs) as pool:
            matrix = pool.map(
                partial(
                    dtw_row,
                    dataset=dataset2,
                    global_constraint=global_constraint,
                    sakoe_chiba_radius=sakoe_chiba_radius,
                    itakura_max_slope=itakura_max_slope,
                ),
                dataset1,
            )
        return np.array(matrix).reshape((len(dataset1), -1))


def dtw_row(
    ts, dataset, global_constraint=None, sakoe_chiba_radius=None, itakura_max_slope=None
):
    row = [
        dtw(
            ts,
            dataset[j],
            global_constraint=global_constraint,
            sakoe_chiba_radius=sakoe_chiba_radius,
            itakura_max_slope=itakura_max_slope,
        )
        for j in range(len(dataset))
    ]
    return row


def construct_knn_graph(features, label, save_path, k=3, n_jobs=3):
    fname = save_path
    num = len(label)

    dist = cdist_dtw(features, n_jobs=n_jobs)

    inds = []
    for i in range(dist.shape[0]):
        ind = np.argpartition(dist[i, :], (k + 1))[: (k + 1)]
        inds.append(ind)

    counter = 0
    indices = []
    for i, v in enumerate(inds):
        mutual_knn = False
        for vv in v:
            if vv == i:
                pass
            else:
                if label[vv] != label[i]:
                    counter += 1
                indices.append([i, vv])
    np.save(save_path, np.array(indices))
    print(save_path)
    print("error rate: {}".format(counter / (num * k)))


def load_graph(data_size, path):
    edges_unordered = np.load(path)
    # adj = tf.SparseTensor(indices=idx, values=np.ones(len(idx)), dense_shape=[data_size, data_size])
    idx = np.array([i for i in range(data_size)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(
        list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32
    ).reshape(edges_unordered.shape)
    if K.floatx() == "float64":
        dtype = np.float64
    else:
        dtype = np.float32
    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(data_size, data_size),
        dtype=dtype,
    )

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_tf_sparse_tensor(adj)
    return adj


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_tf_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    if K.floatx() == "float64":
        dtype = np.float64
    else:
        dtype = np.float32
    sparse_mx = sparse_mx.tocoo().astype(dtype)
    row = np.reshape(sparse_mx.row, (-1, 1)).astype(np.int64)
    col = np.reshape(sparse_mx.col, (-1, 1)).astype(np.int64)
    indices = np.concatenate((row, col), axis=1)
    values = sparse_mx.data
    shape = sparse_mx.shape
    return tf.SparseTensor(indices=indices, values=values, dense_shape=shape)


class CompatibilityException(Exception):
    """Exception raised for compatibility issues.
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


def transform_to_same_length(x, max_length):
    n = len(x)
    # n_var = x[0].shape[1]
    # pad = (0 for _ in range(n_var))

    # only use zero padding to follow univariate UCR method
    for i in range(n):
        if len(x[i].shape) == 1:
            x[i] = np.reshape(x[i], (1, x[i].shape[0]))
        x[i] = np.pad(x[i], ((0, max_length - x[i].shape[0]), (0, 0)), "constant")

    # # the new set in ucr form np array
    # ucr_x = np.zeros((n, max_length, n_var), dtype=np.float64)
    #
    # # loop through each time series
    # for i in range(n):
    #     mts = x[i]
    #     curr_length = mts.shape[1]
    #     idx = np.array(range(curr_length))
    #     idx_new = np.linspace(0, idx.max(), max_length)
    #     for j in range(n_var):
    #         ts = mts[j]
    #         # linear interpolation
    #         new_ts = ts + idx_new
    #         ucr_x[i, :, j] = new_ts
    #
    # return ucr_x
    return x


def get_func_length(x_train, x_test, func):
    if func == min:
        func_length = np.inf
    else:
        func_length = 0

    n = len(x_train)
    for i in range(n):
        func_length = func(func_length, x_train[i].shape[0])

    n = len(x_test)
    for i in range(n):
        func_length = func(func_length, x_test[i].shape[0])

    return func_length


def align(x, max_length):
    for j in range(len(x)):
        x[j] = np.pad(x[j], (0, max_length - len(x[j])), constant_values=(0, 0))


def transform_sktime_to_npy_format(mts_root_dir, mts_out_dir):
    # dataset_files = [name for name in os.listdir(mts_root_dir)]
    dataset_files = ["CharacterTrajectories"]

    for dataset_name in dataset_files:
        out_dir = mts_out_dir + dataset_name + "/"
        create_directory(out_dir)

        x_train_df, y_train = load_UCR_UEA_dataset(
            mts_root_dir + dataset_name + "/" + dataset_name + "_TRAIN.ts"
        )
        x_test_df, y_test = load_UCR_UEA_dataset(
            mts_root_dir + dataset_name + "/" + dataset_name + "_TEST.ts"
        )

        try:
            # ensure to handle string of floats
            y_train = y_train.astype(float)
            y_test = y_test.astype(float)
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        except:
            unique = np.unique(y_train)
            unique.sort()
            for i, val in enumerate(unique):
                y_train = np.where(y_train == val, i, y_train)
                y_test = np.where(y_test == val, i, y_test)
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)
        dims = x_train_df.columns
        x_train = []
        for i in range(x_train_df[dims[0]].size):
            x_channels = []
            t = -1
            realign = False
            for d in dims:
                x_channels.append(x_train_df.loc[i][d].values)
                new_t = len(x_channels[-1])
                if t < 0:
                    t = new_t
                elif t != new_t:
                    t = max((t, new_t))
                    realign = True
            if realign:
                align(x_channels, t)

            x_train.append(np.array(x_channels).T)

        x_test = []
        for i in range(x_test_df[dims[0]].size):
            x_channels = []
            for d in dims:
                x_channels.append(x_test_df.loc[i][d].values)
            x_test.append(np.array(x_channels).T)

        max_length = get_func_length(x_train, x_test, func=max)
        min_length = get_func_length(x_train, x_test, func=min)

        print(dataset_name, "max", max_length, "min", min_length)

        if min_length != max_length:
            x_train = transform_to_same_length(x_train, max_length)
            x_test = transform_to_same_length(x_test, max_length)

        x_train = np.array(x_train)
        x_test = np.array(x_test)

        # save them
        np.save(out_dir + "x_train.npy", x_train)
        np.save(out_dir + "y_train.npy", y_train)
        np.save(out_dir + "x_test.npy", x_test)
        np.save(out_dir + "y_test.npy", y_test)

        print("Done")


if __name__ == "__main__":
    mts_root_dir = "I:/Downloads/Multivariate2018_ts/"
    mts_out_dir = "./archives/Multivariate2018_ts/"
    transform_sktime_to_npy_format(mts_root_dir, mts_out_dir)
