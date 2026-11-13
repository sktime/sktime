import numpy as np
from sklearn.linear_model import LinearRegression

from . import Components


class SeedFinder(object):

    def __init__(self, components):
        """
        :param Components components:
        :return:
        """
        self.components = components

    def to_matrix_for_linear_regression(self, w_tilda):
        # abstract method
        raise NotImplementedError()

    def from_linear_regression_coefs_to_x0(self, linear_regression_coefs):
        # abstract method
        raise NotImplementedError()

    def find(self, w_tilda, residuals):
        w_for_lr = self.to_matrix_for_linear_regression(w_tilda)

        # this makes sure that coefficient for all zeroes dimension will be zero
        # without this calculated coefficient may be very large and unrealistic
        for i in range(w_for_lr.shape[1]):
            if np.allclose(w_for_lr[:, i], 0):
                w_for_lr[:, i] = [0] * len(w_for_lr)

        linear_regression = LinearRegression(fit_intercept=False)
        coefs = np.asarray(linear_regression.fit(w_for_lr, residuals).coef_)
        return self.from_linear_regression_coefs_to_x0(coefs)
