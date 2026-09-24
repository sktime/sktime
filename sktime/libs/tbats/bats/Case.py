from ..abstract import Case as AbstractCase


class Case(AbstractCase):

    def fit_initial_model(self, y):
        """Optimizes seasonal and non-seasonal models with no ARMA and returns the better one

        Parameters
        ----------
        y: array-like of floats
            Time series

        Returns
        -------
        Model
            The best model with no ARMA
        """
        model = self.fit_case(y, self.components.without_arma())
        if len(self.components.seasonal_periods) > 0:
            # Try non-seasonal model without ARMA
            model_candidate = self.fit_case(y, self.components.without_seasonal_periods().without_arma())
            if model_candidate.aic < model.aic:
                model = model_candidate
        return model
