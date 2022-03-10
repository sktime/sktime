# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from numpy.linalg import inv


class Reconciler:
    """Triple double quotes."""

    def __init__(self, method):
        self.method = method
        self._g_dispatch = {
            "BU": self.get_g_matrix_bu,
            "OLS": self.get_g_matrix_ols,
            "WLS_STR": self.get_g_matrix_wls_str,
        }

    def fit(self, hier_data):
        """Triple double quotes."""
        self.hier_data = hier_data
        # self.get_s_matrix()
        # TODO add checks for if method exists in our dict
        method_fn = self._g_dispatch[self.method]
        self.g_matrix = method_fn(hier_data)
        self.s_matrix = self._get_s_matrix(hier_data)
        return [self.g_matrix, self.s_matrix]

    def predict(self, hier_data):
        """Triple double quotes."""
        # check if multi step ahead (more than 1 predict)
        # if len(fh) == 1:
        #     return self._reconcile(hier_data, self.s_matrix, self.g_matrix)
        # if len(fh) > 1:
        # check hier_data for time points in multi index contains 1 column
        hier_data.groupby(level="timepoints")
        hier_data.transform(lambda x: self._reconcile(x, self.s_matrix, self.g_matrix))

    def _reconcile(base_fc, s_matrix, g_matrix):
        # return s_matrix.dot(g_matrix.dot(base_fc))
        return np.dot(s_matrix, np.dot(g_matrix, base_fc))

    # interna
    def _get_s_matrix(hier_data):
        # get bottom level indexes
        bl_inds = (
            hier_data.loc[
                ~(hier_data.index.get_level_values(level=-2).isin(["__total"]))
            ]
            .index.droplevel("timepoints")
            .unique()
        )

        # get all level indexes
        al_inds = hier_data.droplevel(level="timepoints").index.unique()

        s_matrix = pd.DataFrame(
            [[0.0 for i in range(len(bl_inds))] for i in range(len(al_inds))],
            index=al_inds,
        )

        #
        s_matrix.columns = list(bl_inds.get_level_values(level=-1))

        # now insert indicator for bottom level
        for i in s_matrix.columns:
            s_matrix.loc[s_matrix.index.get_level_values(-1) == i, i] = 1.0

        # now for each unique column
        for j in s_matrix.columns:

            # find bottom index id
            inds = list(
                s_matrix.index[s_matrix.index.get_level_values(level=-1).isin([j])]
            )

            # generate new tuples for the aggregate levels
            for i in range(len(inds[0])):
                tmp = list(inds[i])
                tmp[-(i + 1)] = "__total"
                inds.append(tuple(tmp))

            # insrt indicator for aggregates
            for i in inds:
                s_matrix.loc[i, j] = 1.0

        # drop new levels not present in orginal matrix
        s_matrix.dropna(inplace=True)

        return s_matrix

    def get_g_matrix_bu(hier_data):
        """Triple double quotes."""
        # get bottom level indexes
        bl_inds = (
            hier_data.loc[
                ~(hier_data.index.get_level_values(level=-2).isin(["__total"]))
            ]
            .index.droplevel("timepoints")
            .unique()
        )

        # get all level indexes
        al_inds = hier_data.droplevel(level="timepoints").index.unique()

        g_matrix = pd.DataFrame(
            [[0.0 for i in range(len(bl_inds))] for i in range(len(al_inds))],
            index=al_inds,
        )

        #
        g_matrix.columns = list(bl_inds.get_level_values(level=-1))

        # now insert indicator for bottom level
        for i in g_matrix.columns:
            g_matrix.loc[g_matrix.index.get_level_values(-1) == i, i] = 1.0

        return g_matrix.transpose()

    def get_g_matrix_ols(self):
        """Triple double quotes."""
        smat = self._s_matrix
        # get g
        g_ols = pd.DataFrame(
            np.dot(inv(np.dot(np.transpose(smat), smat)), np.transpose(smat))
        )

        g_ols = g_ols.transpose()
        g_ols = g_ols.set_index(smat.index)
        g_ols.columns = smat.columns
        g_ols = g_ols.transpose()

        return g_ols
