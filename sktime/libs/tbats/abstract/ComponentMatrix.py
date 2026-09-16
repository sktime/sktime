import numpy as np


class ComponentMatrix(object):

    def __init__(self, matrix, model_components, append_arma_coefs=False):
        self.components = model_components
        self.matrix = np.atleast_2d(matrix)
        if append_arma_coefs and self.components.arma_length() > 0:
            self.matrix = np.append(self.matrix, np.zeros((len(self.matrix), self.components.arma_length())), axis=1)

    def without_arma(self):
        return __class__(self.matrix[:, :(self.arma_offset())], self.components.without_arma())

    def with_replaced_seasonal(self, new_seasonal_part, new_seasonal_periods):
        if len(new_seasonal_part) == 0 or new_seasonal_part.shape[0] == 0:
            return __class__(
                np.append(self.alpha_beta_part(), self.arma_part(), axis=1),
                self.components.with_seasonal_periods(new_seasonal_periods)
            )
        new_seasonal_part = np.atleast_2d(new_seasonal_part)
        return __class__(
            np.block([self.alpha_beta_part(), new_seasonal_part, self.arma_part()]),
            self.components.with_seasonal_periods(new_seasonal_periods)
        )

    def as_matrix(self):
        return self.matrix

    def as_vector(self):
        """
        :return: First row as flat vector
        """
        return np.asarray(self.matrix.T[:, 0])

    def seasonal_part(self):
        return self.matrix[:, self.seasonal_offset():self.arma_offset()]

    def alpha_beta_part(self):
        return self.matrix[:, :self.seasonal_offset()]

    def arma_part(self):
        return self.matrix[:, self.arma_offset():]

    def break_into_seasons(self):
        w_seasons = []
        offset = 0
        matrix_seasonal_part = self.seasonal_part()
        season_lengths = self.components.seasonal_components_amount()
        for s in season_lengths:
            w_seasons.append(matrix_seasonal_part[:, offset:(offset + s)])
            offset += s
        return w_seasons

    def arma_offset(self):
        return self.matrix.shape[1] - self.components.arma_length()

    def seasonal_offset(self):
        offset = 1
        if self.components.use_trend:
            offset += 1
        return offset
