import math
import numpy as np

from ..abstract import Components, ComponentMatrix, SeedFinder as AbstractSeedFinder


class SeedFinder(AbstractSeedFinder):

    def __init__(self, components):
        """
        :param Components components:
        :return:
        """
        super().__init__(components)
        self.mask = None

    def to_matrix_for_linear_regression(self, w_tilda):
        w_tilda_obj = ComponentMatrix(w_tilda, self.components).without_arma()
        v, cuts = self.prepare_seasonal_params(w_tilda_obj)
        return w_tilda_obj.with_replaced_seasonal(v, cuts).as_matrix()

    def from_linear_regression_coefs_to_x0(self, linear_regression_coefs):
        cuts = self.mask_to_seasonal_cuts(self.get_mask())
        lr_coefs = ComponentMatrix(
            linear_regression_coefs,
            model_components=self.components.with_seasonal_periods(cuts),
            append_arma_coefs=True  # ARMA is not part of linear_regression so we need to re-add it
        )
        seasonal_part = self.transform_to_seasonal_vector(lr_coefs)
        x0 = lr_coefs.with_replaced_seasonal(seasonal_part, self.components.seasonal_periods).as_vector()
        return x0

    def prepare_seasonal_params(self, w_tilda_obj):
        w_tilda_seasons = w_tilda_obj.break_into_seasons()
        new_seasonal_params = []
        season = 0
        cuts = self.mask_to_seasonal_cuts(self.get_mask())
        for length in cuts:
            new_seasonal_params.append(w_tilda_seasons[season][:, 0:length])
            season += 1
        if len(new_seasonal_params) == 0:
            return np.empty((0, w_tilda_obj.matrix.shape[1])), cuts
        return np.block(new_seasonal_params), cuts

    def transform_to_seasonal_vector(self, lr_coefs):
        cuts = lr_coefs.components.seasonal_periods
        seasons = lr_coefs.break_into_seasons()
        new_seasonal_part = []
        for s in range(0, len(self.components.seasonal_periods)):
            season_length = self.components.seasonal_periods[s]
            season_coefs = seasons[s]
            cut_for_season = cuts[s]
            vec = np.asarray([0.0] * season_length)
            vec[:cut_for_season] = season_coefs
            vec = vec - np.mean(vec)  # remove mean so that it sums to 0
            new_seasonal_part.append(vec)
        if len(new_seasonal_part) == 0:
            return np.zeros((1, 0))
        return np.concatenate(new_seasonal_part)

    def get_mask(self):
        if self.mask is None:
            self.mask = self.prepare_mask(self.components.seasonal_periods)
        return self.mask

    @classmethod
    def prepare_mask(cls, seasonal_periods):
        periods = len(seasonal_periods)
        mask = [0] * periods
        for smaller in range(0, periods):  # for all seasonal periods
            for larger in range(smaller + 1, periods):  # for larger > smaller
                if (seasonal_periods[larger] % seasonal_periods[smaller]) == 0:
                    mask[smaller] = 1

        for larger in reversed(range(1, periods)):  # for larger in decreasing order
            for smaller in reversed(range(0, larger)):  # for smaller < larger in decreasing order
                if mask[smaller] == 1 or mask[larger] == 1:
                    continue
                hcf = math.gcd(seasonal_periods[smaller], seasonal_periods[larger])
                if hcf > 1:
                    mask[larger] = -hcf
        return mask

    def mask_to_seasonal_cuts(self, mask):
        seasonal_periods = self.components.seasonal_periods
        cuts = []
        # offset = 0
        for s in range(0, len(seasonal_periods)):
            period_length = seasonal_periods[s]
            if mask[s] < 0:  # there is a common divisor between seasons, cut out whole divisor
                length = period_length + mask[s]  # note that mask is negative
            elif mask[s] == 0:  # nothing in common, cut out last parameter
                length = period_length - 1
            else:  # when mask[s] == 1 cut whole season out as it is a sub-season of some longer season
                length = 0
            cuts.append(length)
            # offset += length

        return cuts
