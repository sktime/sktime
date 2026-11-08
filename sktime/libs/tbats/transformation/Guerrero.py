import numpy as np
from scipy.optimize import minimize_scalar


# Guerrero, V.M. (1993) Time-series analysis supported by power
# transformations. \emph{Journal of Forecasting}, \bold{12}, 37--48.
class Guerrero(object):

    def __init__(self):
        self.y = None
        self.season_length = None

    def find_lambda(self, y, seasonal_periods=None, bounds=(-1, 2)):
        # assertion seasonal_periods are sorted and at least 2
        if seasonal_periods is None:
            seasonal_periods = []
        seasonal_periods = np.asarray(seasonal_periods)

        self.season_length = self.find_longest_season_with_2x_observations(y, seasonal_periods)
        if len(y) < 2 * self.season_length:
            return 1.0

        if np.any(y <= 0):
            bounds[0] = np.max([bounds[0], 0])

        if bounds[0] >= bounds[1]:
            return bounds[0]

        return self.minimize(y, bounds)

    @staticmethod
    def find_longest_season_with_2x_observations(y, seasonal_periods):
        season_length = 2
        for length in seasonal_periods:
            if len(y) < int(length) * 2:
                break
            # there are at least 2 seasons of observations
            season_length = length
        return int(season_length)

    def minimize(self, y, bounds):
        self.y = y
        result = minimize_scalar(
            self.guerrero_coefficient_of_variation,
            method='bounded',
            bounds=bounds,
            options=dict(
                xatol=1e-8,
            )

        )
        return result.x

    def guerrero_coefficient_of_variation(self, lam):
        # assertion y is long enough to have 2 full seasons of observations
        full_seasons = int(len(self.y) / self.season_length)
        non_full_season_offset = int(len(self.y) - full_seasons * self.season_length)
        # nth seasonal moment - nth observation in each season
        y_seasonal_moments = np.reshape(self.y[non_full_season_offset:], (full_seasons, self.season_length))
        y_moment_means = np.mean(y_seasonal_moments, axis=1)
        y_moment_stds = np.std(y_seasonal_moments, axis=1, ddof=1)
        y_moment_ratios = y_moment_stds / y_moment_means ** (1 - lam)
        return np.std(y_moment_ratios, ddof=1) / np.mean(y_moment_ratios)
