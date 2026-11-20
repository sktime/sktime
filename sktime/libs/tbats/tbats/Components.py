import numpy as np
import copy

from ..abstract import ArrayHelper, Components as AbstractComponents


class Components(AbstractComponents):
    """Contains information necessary to determine amount of parameters of the model.

    See parent class description for details.
    """

    def __init__(self, use_box_cox=False, use_trend=False, use_damped_trend=False,
                 seasonal_periods=None, seasonal_harmonics=None,
                 use_arma_errors=False, p=0, q=0,
                 box_cox_bounds=(0, 1)):
        """
        Components class contains all the information necessary to determine the amount and the structure
        of parameters used in the model but not parameters' values.

        Parameters
        ----------
        seasonal_harmonics: array-like of positive int values
            Amount of harmonics used to model each of the seasons.
            Length of this array must equal amount of seasons (length of seasonal_periods)
        """
        super().__init__(
            use_box_cox=use_box_cox, use_trend=use_trend, use_damped_trend=use_damped_trend,
            seasonal_periods=seasonal_periods,
            use_arma_errors=use_arma_errors, p=p, q=q,
            box_cox_bounds=box_cox_bounds)

        self._init_seasonal_harmonics(seasonal_harmonics)
        # assertion: all provided harmonics are positive, 0 should not be provided

    @classmethod
    def create_constant_components(cls):
        return cls(use_box_cox=False, use_trend=False, seasonal_periods=None, use_arma_errors=False)

    def gamma_params_amount(self):
        """TBATS model contains two gamma parameters for each seasonality"""
        return 2 * len(self.seasonal_periods)

    def seasonal_components_amount(self):
        """TBATS model contains two seed values for each seasonality being modelled"""
        return self.seasonal_harmonics * 2

    def with_harmonics_as_ones(self):
        """Creates copy of itself but with all harmonics set to 1

        Returns
        -------
        Components
            copy of components with all harmonics set to 1
        """
        me = copy.deepcopy(self)
        me._init_seasonal_harmonics()  # sets them to a vector of ones
        return me

    def with_harmonic_for_season(self, season_index, new_harmonic):
        """Creates copy of itself with provided harmonics amount for chosen season.

        Parameters
        ----------
        season_index: int
            Index of the season to change harmonic for.
            Index for first season is 0.
        new_harmonic: int
            New amount of harmonics for this season.
            Only positive values are allowed.

        Returns
        -------
        Components
            copy of components with provided harmonics amount for chosen season.
        """
        me = copy.deepcopy(self)
        me.seasonal_harmonics[season_index] = new_harmonic
        return me

    def with_seasonal_periods(self, seasonal_periods):
        """Creates copy of itself but with new seasonal periods

        All harmonics will be set to 1

        Parameters
        ----------
        seasonal_periods: array-like
            New season lengths

        Returns
        -------
        Components
            copy of components with new seasonal periods
        """
        me = super().with_seasonal_periods(seasonal_periods)
        me._init_seasonal_harmonics()
        return me

    def without_seasonal_periods(self):
        """Creates copy itself without seasonality

        Returns
        -------
        Components
            copy of components without seasonal periods
        """
        me = super().without_seasonal_periods()
        me._init_seasonal_harmonics()
        return me

    def _normalize_seasons(self, seasonal_periods):
        """Ensures all seasons are float values"""
        return ArrayHelper.to_array(seasonal_periods, float)

    def _init_seasonal_harmonics(self, seasonal_harmonics=None):
        self.seasonal_harmonics = ArrayHelper.to_array(seasonal_harmonics, int)
        if len(self.seasonal_harmonics) != len(self.seasonal_periods):
            self.seasonal_harmonics = np.asarray([1] * len(self.seasonal_periods), int)

    def _seasonal_summary(self):
        return "Seasonal harmonics %s\n" % self.seasonal_harmonics
