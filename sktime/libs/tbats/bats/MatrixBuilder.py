import numpy as np

from ..abstract import ArrayHelper, MatrixBuilder as AbstractMatrixBuilder


class MatrixBuilder(AbstractMatrixBuilder):

    def __init(self, model_params):
        super().__init__(model_params)

    def make_seasonal_components_for_w(self):
        seasonal_components = []
        for period in self.params.components.seasonal_periods:
            seasonal_components.append(ArrayHelper.make_one_and_zeroes_vector(period, one_position='end'))
        if len(seasonal_components) > 0:
            seasonal_components = np.concatenate(seasonal_components)
        return seasonal_components

    def make_gamma_vector(self):
        gamma_params = self.params.gamma_params
        seasonal_periods = self.params.components.seasonal_periods
        gamma_vectors = []
        i = 0
        for gamma in gamma_params:
            gamma_vectors.append(ArrayHelper.make_one_and_zeroes_vector(seasonal_periods[i], one=gamma))
            i += 1
        if len(gamma_vectors) > 0:
            gamma_vectors = np.concatenate(gamma_vectors)
        return np.asarray(gamma_vectors)

    def make_A_matrix(self):
        seasonal_periods = self.params.components.seasonal_periods
        if len(seasonal_periods) == 0:
            return np.zeros((0, 0))
        A_array = []
        for period in seasonal_periods:
            Ai = np.block([
                [np.zeros((1, period - 1)), 1],
                [np.identity(period - 1), np.zeros((period - 1, 1))]
            ])
            A_array.append(Ai)
        tao = int(np.sum(seasonal_periods))
        dsum = np.zeros((tao, tao))
        offsetRow = 0
        offsetCol = 0
        for Ai in A_array:
            dsum[offsetRow:offsetRow + Ai.shape[0], offsetCol:offsetCol + Ai.shape[1]] = Ai
            offsetRow += Ai.shape[0]
            offsetCol += Ai.shape[1]
        return dsum
