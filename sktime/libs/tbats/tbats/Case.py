from ..abstract import Case as AbstractCase


class Case(AbstractCase):

    def fit_initial_model(self, y):
        """Fits model with no ARMA and returns it

        TBATS, unlike BATS, does not check for non-seasonal model,
        non-seasonal models for TBATS are checked when seasonal harmonics are being determined

        Parameters
        ----------
        y: array-like
            Time series

        Returns
        -------
        Model
            Fitted model with no ARMA
        """
        return self.fit_case(y, self.components.without_arma())
