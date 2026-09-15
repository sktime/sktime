from pmdarima.arima import auto_arima


class Case(object):
    """Provides means to choose best model for provided components setup

    Attributes
    ----------
    components: Components
        Components model should use
    model: Model or None
        Best model found.
        None when no fitting occured yet.

    Methods
    """

    def __init__(self, components, context):
        """Creates new BATS/TBATS case

        Don't use this constructor directly. See ContextInterface.create_case

        Parameters
        ----------
        components: Components
            Components the case applies to
        context: ContextInterface
            Used to build models
        """
        self.components = components
        self.context = context
        self.model = None  # not fitted yet

    def fit(self, y):
        """ Fits and chooses best model with chosen components

        Parameters
        ----------
        y: array-like
            Time series to fit the model to and rate the model against

        Returns
        -------
        Model
            Best model by AIC
        """
        best_model = self.fit_initial_model(y)

        if self.components.use_arma_errors:
            # Try adding ARMA to the model
            arma_model = auto_arima(best_model.resid, stationary=True, trend='n',
                                    suppress_warnings=True, error_action='ignore')
            p = arma_model.order[0]
            q = arma_model.order[2]
            if p > 0 or q > 0:  # Found ARMA components
                # Fit model with ARMA errors modelling
                model_candidate = self.fit_case(y, best_model.params.components.with_arma(p, q))
                if model_candidate.aic < best_model.aic:
                    best_model = model_candidate

        self.model = best_model
        return self.model

    def fit_initial_model(self, y):
        # abstract method
        raise NotImplementedError()

    def fit_case(self, y, case_definition):
        """Fits model with provided components to time series

        Parameters
        ----------
        y: array-like
            Time series
        case_definition: Components
            Components for which model should be fit

        Returns
        -------
        Model
            Fitted model
        """
        starting_params = self.context.create_default_starting_params(y, components=case_definition)
        return self.fit_with_starting_params(y, starting_params)

    def fit_with_starting_params(self, y, model_params):
        """ Fits (optimizes) model with provided starting parameters

        Parameters
        ----------
        y: array-like
            Time series
        model_params: ModelParams
            Model parameters

        Returns
        -------
        Model
            optimal model
        """
        optimization = self.context.create_params_optimizer()
        return optimization.optimize(y, model_params).optimal_model()
