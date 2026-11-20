from ..abstract import ArrayHelper, Components as AbstractComponents


class Components(AbstractComponents):
    """Contains information necessary to determine amount of parameters of the model.

    See parent class description for details.
    """
    def __init__(self, use_box_cox=False, use_trend=False, use_damped_trend=False,
                 seasonal_periods=None, use_arma_errors=False, p=0, q=0,
                 box_cox_bounds=(0, 1)):
        super().__init__(
            use_box_cox=use_box_cox, use_trend=use_trend, use_damped_trend=use_damped_trend,
            seasonal_periods=seasonal_periods,
            use_arma_errors=use_arma_errors, p=p, q=q,
            box_cox_bounds=box_cox_bounds)

    @classmethod
    def create_constant_components(cls):
        return cls(use_box_cox=False, use_trend=False, seasonal_periods=None, use_arma_errors=False)

    def seasonal_components_amount(self):
        """Returns amount of seasonal seed states per season"""
        return self.seasonal_periods

    def gamma_params_amount(self):
        """BATS model contains one gamma parameter for each season"""
        return len(self.seasonal_periods)

    def _normalize_seasons(self, seasonal_periods):
        """Ensures seasons are integer values"""
        return ArrayHelper.to_array(seasonal_periods, int)

    def _seasonal_summary(self):
        return ""
