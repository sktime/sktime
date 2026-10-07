import numpy as np

from . import MatrixBuilderInterface
from . import ArrayHelper


class MatrixBuilder(MatrixBuilderInterface):

    def __init__(self, model_params):
        self.params = model_params

    def calculate_D_matrix(self):
        F = self.make_F_matrix()
        g = self.make_g_vector()
        w = self.make_w_vector()
        D = F - np.outer(g, w)
        return D

    def make_seasonal_components_for_w(self):
        raise NotImplementedError()

    def make_gamma_vector(self):
        raise NotImplementedError()

    def make_A_matrix(self):
        raise NotImplementedError()

    def seasonal_components_amount(self):
        return self.params.seasonal_components_amount()

    def make_w_vector(self):
        ar_coefs = self.params.ar_coefs
        ma_coefs = self.params.ma_coefs

        w = np.concatenate([
            [1.0],
            [self.params.phi],
            self.make_seasonal_components_for_w(),
            ar_coefs,
            ma_coefs
        ])
        w = w[w != np.array(None)]
        return np.asarray(w, dtype=float)

    def make_g_vector(self):
        g = np.concatenate([
            [self.params.alpha],
            [self.params.beta],
            self.make_gamma_vector(),
            ArrayHelper.make_one_and_zeroes_vector(self.params.components.p),
            ArrayHelper.make_one_and_zeroes_vector(self.params.components.q),
        ])
        g = g[g != np.array(None)]
        return np.asarray(g, dtype=float)

    def make_F_matrix(self):
        ar_coefs = self.params.ar_coefs
        ma_coefs = self.params.ma_coefs
        phi = ArrayHelper.to_array(self.params.phi)
        seasonal_periods = self.params.components.seasonal_periods

        tao = self.seasonal_components_amount()
        p = len(ar_coefs)
        q = len(ma_coefs)

        has_beta = self.params.components.use_trend
        has_phi = self.params.components.use_trend  # Even if use_damped_trend=False phi is used and fixed to 1.0
        has_seasonal = (tao > 0)

        F = []

        row = [np.ones((1, 1))]
        if has_phi:
            row.append(phi)
        if has_seasonal:
            row.append(np.zeros((1, tao)))
        if p > 0:
            row.append(self.params.alpha * ar_coefs)
        if q > 0:
            row.append(self.params.alpha * ma_coefs)
        F.append(row)

        if has_beta:
            row = [np.zeros((1, 1))]
            if has_phi:
                row.append(phi)
            if has_seasonal:
                row.append(np.zeros((1, tao)))
            if p > 0:
                row.append(self.params.beta * ar_coefs)
            if q > 0:
                row.append(self.params.beta * ma_coefs)
            F.append(row)

        if has_seasonal:
            row = [np.zeros((tao, 1))]
            if has_phi:
                row.append(np.zeros((tao, 1)))
            row.append(self.make_A_matrix())  # A matrix
            gamma_vectors = self.make_gamma_vector()
            if p > 0:
                row.append(np.outer(gamma_vectors, ar_coefs))  # B matrix
            if q > 0:
                row.append(np.outer(gamma_vectors, ma_coefs))  # C matrix
            F.append(row)

        if p > 0:
            row = [np.zeros((p, 1))]
            if has_phi:
                row.append(np.zeros((p, 1)))
            if has_seasonal:
                row.append(np.zeros((p, tao)))
            ar_matrix = np.block([[ar_coefs], [np.eye(p - 1 if p > 0 else 0, p)]])
            row.append(ar_matrix)
            if q > 0:
                ar_matrix_ma_part = np.block([[ma_coefs], [np.zeros((p - 1 if p > 0 else 0, q))]])
                row.append(ar_matrix_ma_part)
            F.append(row)

        if q > 0:
            row = [np.zeros((q, 1))]
            if has_phi:
                row.append(np.zeros((q, 1)))
            if has_seasonal:
                row.append(np.zeros((q, tao)))
            if p > 0:
                ma_matrix_ar_part = np.zeros((q, p))
                row.append(ma_matrix_ar_part)
            ma_matrix = np.eye(q, q, -1)
            row.append(ma_matrix)
            F.append(row)

        return np.block(F)
