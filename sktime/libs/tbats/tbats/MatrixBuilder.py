import numpy as np

from ..abstract import MatrixBuilder as AbstractMatrixBuilder
from . import ModelParams


class MatrixBuilder(AbstractMatrixBuilder):

    def __init(self, model_params):
        """

        :param ModelParams model_params:
        :return:
        """
        super().__init__(model_params)

    def make_seasonal_components_for_w(self):
        seasonal_components = []
        for harmonic in self.params.components.seasonal_harmonics:
            if harmonic > 0:
                seasonal_components.append([1.0] * harmonic)
                seasonal_components.append([0.0] * harmonic)
        if len(seasonal_components) > 0:
            seasonal_components = np.concatenate(seasonal_components)
        return seasonal_components

    def make_gamma_vector(self):
        gamma_1 = self.params.gamma_1()
        gamma_2 = self.params.gamma_2()
        seasonal_harmonics = self.params.components.seasonal_harmonics
        gamma_vectors = []
        for i in range(0, len(seasonal_harmonics)):  # assertion: seasonal_harmonics length equals amount of periods
            k = seasonal_harmonics[i]
            gamma_vectors.append([gamma_1[i]] * k)
            gamma_vectors.append([gamma_2[i]] * k)
        if len(gamma_vectors) > 0:
            gamma_vectors = np.concatenate(gamma_vectors)
        return np.asarray(gamma_vectors)

    def make_A_matrix(self):
        seasonal_periods = self.params.components.seasonal_periods
        seasonal_harmonics = self.params.components.seasonal_harmonics
        if len(seasonal_periods) == 0:
            return np.zeros((0, 0))
        A_array = []
        for period in range(0, len(seasonal_periods)):
            harmonics_amount = seasonal_harmonics[period]
            if harmonics_amount == 0:
                continue
            period_length = seasonal_periods[period]
            period_lambda = 2 * np.pi * np.asarray(range(1, harmonics_amount + 1)) / period_length
            cos_coefs = np.diag(np.cos(period_lambda))
            sin_coefs = np.diag(np.sin(period_lambda))
            Ai = np.block([
                [cos_coefs, sin_coefs],
                [-sin_coefs, cos_coefs],
            ])
            A_array.append(Ai)
        tao = int(2 * np.sum(seasonal_harmonics))
        dsum = np.zeros((tao, tao))
        offsetRow = 0
        offsetCol = 0
        for Ai in A_array:
            dsum[offsetRow:offsetRow + Ai.shape[0], offsetCol:offsetCol + Ai.shape[1]] = Ai
            offsetRow += Ai.shape[0]
            offsetCol += Ai.shape[1]
        return dsum
