#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Kutay Koralturk", "Martin Walter"]
__all__ = ["BSTS"]

import numpy as np
import pandas as pd
import scipy.stats as st
import operator
from sktime.forecasting.base._sktime import _SktimeForecaster
from sktime.forecasting.base._sktime import DEFAULT_ALPHA
from sktime.forecasting.base._sktime import _OptionalForecastingHorizonMixin
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("tensorflow_probability")
# _check_soft_dependencies("tensorflow")


class BSTS(_OptionalForecastingHorizonMixin, _SktimeForecaster):
    """
     Bayesian Structural Time Series forecaster by wrapping tensorflow_probability.sts.
     Parameters
     ----------
     Args for tfp.sts.Sum()
     ------------------------------------------
         constant_offset:
             optional float Tensor of shape broadcasting to
             concat([batch_shape, [num_timesteps]]) specifying a
             constant value added to the sum of outputs
             from the component models.
             This allows the components to model the shifted
             series y - constant_offset.
             If None, this is set to the mean of the
             provided y-data.
             Default value: None.
         observation_noise_scale_prior:
             optional tfd.Distribution instance specifying a
             prior on observation_noise_scale.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         name:
             Python str name of this model component;
             used as name_scope for ops created by this class.
             Default value: 'Sum'.
         sample_size:
             Python list of Tensors representing posterior samples of
             model parameters, with shapes
             [concat([[num_posterior_draws],param.prior.batch_shape,
             param.prior.event_shape]) for param in model.parameters].
             This may optionally also be a map (Python dict)
             of parameter names to Tensor values.
         seed:
             Python integer to seed the random number generator.


     Args for tfp.sts.Autoregressive()
     ------------------------------------------
         order__Autoregressive:
             scalar Python positive int specifying the
             number of past timesteps to regress on.
         coefficients_prior__Autoregressive:
             optional tfd.Distribution instance specifying
             a prior on the coefficients parameter.
             If None, a default standard normal
             (tfd.MultivariateNormalDiag(scale_diag=tf.ones([order])))
             prior is used.
             Default value: None.
         level_scale_prior__Autoregressive:
             optional tfd.Distribution instance specifying
             a prior on the level_scale parameter.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         initial_state_prior__Autoregressive:
             optional tfd.Distribution instance
             specifying a prior on the initial state,
             corresponding to the values of the process
             at a set of size order of
             imagined timesteps before the initial step.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         coefficient_constraining_bijector__Autoregressive:
             optional tfb.Bijector instance representing a
             constraining mapping for the autoregressive
             coefficients. For example, tfb.Tanh() constrains
             the coefficients to lie in (-1, 1),
             while tfb.Softplus() constrains them to be positive,
             and tfb.Identity() implies no constraint.
             If None, the default behavior constrains the coefficients
             to lie in (-1, 1) using a Tanh bijector.
             Default value: None.
         name__Autoregressive:
             the name of this model component.
             Default value: 'Autoregressive'.
     Args for tfp.sts.LinearRegression()
     ------------------------------------------

         design_matrix__LinearRegression:
             float Tensor of shape concat([batch_shape,
             [num_timesteps, num_features]]).This may also
             optionally be an instance of tf.linalg.LinearOperator.
         weights_prior__LinearRegression:
             tfd.Distribution representing a prior
             over the regression weights.
             Must have event shape [num_features]
             and batch shape broadcastable
             to the design matrix's batch_shape.
             If None, defaults to
             Sample(StudentT(df=5, loc=0., scale=10.), num_features]),
             a weakly-informative prior loosely inspired
             by the Stan prior choice recommendations.
             Default value: None.
         name__LinearRegression:
             the name of this model component.
             Default value: 'LinearRegression'.
     Args for tfp.sts.DynamicLinearRegression()
     ------------------------------------------
         design_matrix__DynamicLinearRegression:
             float Tensor of shape
             concat([batch_shape,[num_timesteps, num_features]]).
         drift_scale_prior__DynamicLinearRegression:
             instance of tfd.Distribution specifying
             a prior on the drift_scale parameter.
             If None, a heuristic default prior is
             constructed based on the provided
             y-data.
             Default value: None.
         initial_weights_prior__DynamicLinearRegression:
             instance of tfd.MultivariateNormal representing
             the prior distribution on the
             latent states (the regression weights).
             Must have event shape [num_features].
             If None, a weakly-informative
             Normal(0., 10.) prior is used.
             Default value: None.
         name__DynamicLinearRegression:
             Python str for the name of this component.
             Default value: 'DynamicLinearRegression'.

     Args for tfp.sts.SparseLinearRegression()
     ------------------------------------------
         design_matrix__SparseLinearRegression:
             float Tensor of shape
             concat([batch_shape,[num_timesteps, num_features]]).
             This may also optionally be an
             instance of tf.linalg.LinearOperator.
         weights_prior_scale__SparseLinearRegression:
             float Tensor defining the scale of the
             Horseshoe prior on regression weights.
             Small values encourage the weights to be sparse.
             The shape must broadcast with weights_batch_shape.
             Default value: 0.1.
         weights_batch_shape__SparseLinearRegression:
             if None, defaults to design_matrix.batch_shape_tensor().
             Must broadcast with the batch shape of design_matrix.
             Default value: None.
         name__SparseLinearRegression:
             the name of this model component.
             Default value: 'SparseLinearRegression'.

     Args for tfp.sts.LocalLevel()
     ------------------------------------------

         level_scale_prior__LocalLevel:
             optional tfd.Distribution instance specifying
             a prior on the level_scale parameter.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         name__LocalLevel:
             the name of this model component.
             Default value: 'LocalLevel'

     Args for tfp.sts.LocalLinearTrend()
     ------------------------------------------
         level_scale_prior__LocalLinearTrend:
             optional tfd.Distribution instance specifying
             a prior on the level_scale parameter.
             If None, a heuristic default prior is
             constructed based on the provided y-data.
             Default value: None.
         slope_scale_prior__LocalLinearTrend:
             optional tfd.Distribution instance specifying
             a prior on the slope_scale parameter.
             If None, a heuristic default prior is
             constructed based on the provided y-data.
             Default value: None.
         initial_level_prior__LocalLinearTrend:
             optional tfd.Distribution instance
             specifying a prior on the initial level.
             If None, a heuristic default prior is
             constructed based on the provided y-data.
             Default value: None.
         initial_slope_prior__LocalLinearTrend:
             optional tfd.Distribution instance specifying
             a prior on the initial slope.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         name__LocalLinearTrend:
             the name of this model component.
             Default value: 'LocalLevel'

     Args for tfp.sts.LocalLinearTrend()
     ------------------------------------------
         level_scale_prior__SemiLocalLinearTrend:
             optional tfd.Distribution instance specifying a
             prior on the level_scale parameter.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         slope_mean_prior__SemiLocalLinearTrend:
             optional tfd.Distribution instance specifying a
             prior on the slope_mean parameter.
             If None, a heuristic default prior is constructed based
             on the provided y-data.
             Default value: None.
         slope_scale_prior__SemiLocalLinearTrend:
             optional tfd.Distribution instance specifying a
             prior on the slope_scale parameter.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         autoregressive_coef_prior__SemiLocalLinearTrend:
             optional tfd.Distribution instance specifying a
             prior on the autoregressive_coef parameter.
             If None, the default prior is a standard Normal(0., 1.).
             Note that the prior may be implicitly truncated
             by constrain_ar_coef_stationary and/or
             constrain_ar_coef_positive.
             Default value: None.
         initial_level_prior__SemiLocalLinearTrend:
             optional tfd.Distribution instance specifying a
             prior on the initial level.
             If None, a heuristic default
             prior is constructed based on the
             provided y-data.
             Default value: None.
         initial_slope_prior__SemiLocalLinearTrend:
             optional tfd.Distribution instance
             specifying a prior on the initial slope.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         constrain_ar_coef_stationary__SemiLocalLinearTrend:
             if True, perform inference using a
             parameterization that restricts
             autoregressive_coef to the interval (-1, 1), or (0, 1)
             if force_positive_ar_coef is also True,
             corresponding to stationary processes.
             This will implicitly truncates the
             support of autoregressive_coef_prior.
             Default value: True.
         constrain_ar_coef_positive__SemiLocalLinearTrend:
             if True, perform inference using a parameterization
             that restricts autoregressive_coef to be positive,
             or in (0, 1) if constrain_ar_coef_stationary is also True.
             This will implicitly truncate the support of
             autoregressive_coef_prior.
             Default value: False.
         name__SemiLocalLinearTrend:
             the name of this model component.
             Default value: 'SemiLocalLinearTrend'.

    Args for tfp.sts.Seasonal()
     ------------------------------------------
         num_seasons__Seasonal:
             Scalar Python int number of seasons.
         num_steps_per_season__Seasonal:
             Python int number of steps in each season.
             This may be either a scalar (shape []),
             in which case all seasons have the same length, or
             a NumPy array of shape [num_seasons],
             in which seasons have different length, but remain
             constant around different cycles,
             or a NumPy array of shape [num_cycles, num_seasons],
             in which num_steps_per_season
             for each season also varies in different cycle
             (e.g., a 4 years cycle with leap day).
             Default value: 1.
         allow_drift__Seasonal:
             optional Python bool specifying whether the seasonal
             effects can drift over time.
             Setting this to False removes the drift_scale
             parameter from the model.
             This is mathematically equivalent to
             drift_scale_prior = tfd.Deterministic(0.),
             but removing drift directly is preferred
             because it avoids the use of a degenerate prior.
             Default value: True.
         drift_scale_prior__Seasonal:
             optional tfd.Distribution instance specifying a
             prior on the drift_scale parameter.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         initial_effect_prior__Seasonal:
             optional tfd.Distribution instance specifying a
             normal prior on the initial effect of each season.
             This may be either a scalar tfd.Normal prior,
             in which case it applies
             independently to every season, or it may be
             multivariate normal (e.g., tfd.MultivariateNormalDiag)
             with event shape [num_seasons], in which case it
             specifies a joint prior across all seasons.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         constrain_mean_effect_to_zero__Seasonal:
             if True, use a model parameterization that
             constrains the mean effect across all seasons to be zero.
             This constraint is generally helpful in identifying the
             contributions of different model
             components and can lead to more
             interpretable posterior decompositions.
             It may be undesirable
             if you plan to directly examine the latent
             space of the underlying state space model.
             Default value: True.
         name__Seasonal:
             the name of this model component.
             Default value: 'Seasonal'.

     Args for tfp.sts.SmoothSeasonal()
     ------------------------------------------
         period__SmoothSeasonal:
             positive scalar float Tensor giving the number of timesteps
             required for the longest cyclic effect to repeat.
         frequency_multipliers__SmoothSeasonal:
             One-dimensional float Tensor listing the frequencies
             (cyclic components) included in the model,
             as multipliers of the base/fundamental frequency 2.
             * pi / period. Each component is specified by the number
             of times it repeats per period, and
             adds two latent dimensions to the model.
             A smooth seasonal model that can represent
             any periodic function is given
             by frequency_multipliers = [1,2, ..., floor(period / 2)].
             However, it is often desirable to enforce a smoothness
             assumption (and reduce the computational burden)
             by dropping some of the higher frequencies.
         allow_drift__SmoothSeasonal:
             optional Python bool specifying whether the seasonal
             effects can drift over time.
             Setting this to False removes the
             drift_scale parameter from the model.
             This is mathematically equivalent to
             drift_scale_prior = tfd.Deterministic(0.),
             but removing drift directly is preferred because
             it avoids the use of a degenerate prior.
             Default value: True.
         drift_scale_prior__SmoothSeasonal:
             optional tfd.Distribution instance specifying a
             prior on the drift_scale parameter.
             If None, a heuristic default prior is constructed
             based on the provided y-data.
             Default value: None.
         initial_state_prior__SmoothSeasonal:
             instance of tfd.MultivariateNormal representing the
             prior distribution on the latent states.
             Must have event shape [2 * len(frequency_multipliers)].
             If None, a heuristic default prior is constructed based on
             the provided y-data.
         name__SmoothSeasonal:
             the name of this model component.
             Default value: 'SmoothSeasonal'.


     References
     ----------
     https://www.tensorflow.org/probability/api_docs/python/tfp/sts
    """

    def __init__(
        self,
        sample_size=200,
        constant_offset=None,
        observation_noise_scale_prior=None,
        name="STS",
        random_state=0,
        **kwargs
    ):

        self.sample_size = sample_size
        self.constant_offset = constant_offset
        self.observation_noise_scale_prior = observation_noise_scale_prior
        self.name = name
        self.random_state = random_state
        self.__dict__.update(kwargs)

        self._time_series_components = {}
        self.time_series_components = []

        self.valid_components = [
            "Autoregressive",
            "DynamicLinearRegression",
            "LinearRegression",
            "SparseLinearRegression",
            "LocalLevel",
            "LocalLinearTrend",
            "SemiLocalLinearTrend",
            "Seasonal",
            "SmoothSeasonal",
        ]

        self._forecaster = None
        self._fitted_forecaster = None

        # import inside method to avoid hard dependency
        from tensorflow_probability import sts as _sts
        import tensorflow as tf

        self.tf = tf
        tf.random.set_seed(self.random_state)

        self._ModelClass = _sts

        super(BSTS, self).__init__()

    def _check_components(self, component_name):
        if component_name not in self.valid_components:
            raise Exception(
                "Please provide a valid component name.  \
                Valid component names are: {}".format(
                    self.valid_components
                )
            )

    def _process_kwargs(self):
        for key, value in self.__dict__.items():
            splitted_key = key.split("__")

            if len(splitted_key) > 1:
                if len(splitted_key) > 3:
                    raise Exception(
                        "Component arguments in `{}` is provided in a wrong way.  \
                        Please try to follow the `__` convention to denote different  \
                        arguments of different components.".format(
                            key
                        )
                    )

                self._check_components(splitted_key[1])
                identifier = "__".join(splitted_key[1:])
                if identifier not in self._time_series_components:
                    self._time_series_components[str(identifier)] = {}

                self._time_series_components[str(identifier)][splitted_key[0]] = value

    def _instantiate_model(self, y, X=None):

        y = y.astype("float64")

        if X is not None:
            X = X.astype("float64")

        self.process_kwargs()

        self.time_series_components = []

        for key, conf in self._time_series_components.items():
            splitted_key = key.split("__")

            class_name = splitted_key[0]

            if "name" not in conf:
                conf["name"] = key

            if "Regression" in class_name:
                self._check_design_matrix(design_matrix=X)
                self.time_series_components.append(
                    operator.attrgetter(class_name)(self._ModelClass)(
                        design_matrix=X, **conf
                    )
                )
            else:
                self.time_series_components.append(
                    operator.attrgetter(class_name)(self._ModelClass)(
                        observed_time_series=y, **conf
                    )
                )

        self._forecaster = self._ModelClass.Sum(
            self.time_series_components,
            observed_time_series=y,
            constant_offset=self.constant_offset,
            observation_noise_scale_prior=self.observation_noise_scale_prior,
            name=self.name,
        )

        return self

    def fit(self, y, X=None, fh=None):
        """Fit to training data.
        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """
        self.tf.random.set_seed(self.random_state)
        self._set_y_X(y, X)
        self._type_check_y_X(self._y, self._X)
        self._set_fh(fh)
        self._instantiate_model(y=y, X=X)
        self._fitted_forecaster = self._ModelClass.build_factored_surrogate_posterior(
            model=self._forecaster
        )

        self._parameter_samples = self._fitted_forecaster.sample(self.sample_size)
        self._is_fitted = True
        return self

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Predict

        Parameters
        ----------
        fh : array-like
            The forecasters horizon with the steps ahead to to predict.
            Default is
            one-step ahead forecast, i.e. np.array([1]).
        X : pd.DataFrame, optional
            Exogenous data, by default None
        return_pred_int : bool, optional
            Returns a pd.DataFrame with confidence intervalls, by default False
        alpha : float, optional
            Alpha level for confidence intervalls, by default DEFAULT_ALPHA
        Returns
        -------
        y_pred : pandas.Series
            Returns series of predicted values.
        Raises
        ------
        Exception
            Error when merging data
        """

        self._type_check_y_X(X=X)
        fh = fh.to_relative(cutoff=self.cutoff)

        # Outsample
        if not fh.is_all_in_sample(cutoff=self.cutoff):
            fh_out = fh.to_out_of_sample(cutoff=self.cutoff)
            steps = fh_out.to_pandas().max().astype("int32")
            self._forecast_dist = self._ModelClass.forecast(
                model=self._forecaster,
                observed_time_series=self._y,
                parameter_samples=self._parameter_samples,
                num_steps_forecast=steps,
            )

            y_out_sample = self._forecast_dist.mean().numpy()[..., 0]
            standard_deviation = self._forecast_dist.stddev().numpy()[..., 0]
            p_value = alpha / 2
            z_score = st.norm.ppf(1 - p_value)
            lower = y_out_sample - standard_deviation * z_score
            upper = y_out_sample + standard_deviation * z_score
            pred_int = self._get_pred_int(lower=lower, upper=upper)

        else:
            y_out_sample = np.array([])

        # Insample
        demand_one_step_dist = self._ModelClass.one_step_predictive(
            model=self._forecaster,
            observed_time_series=self._y,
            parameter_samples=self._parameter_samples,
        )

        y_in_sample, _ = (
            demand_one_step_dist.mean().numpy(),
            demand_one_step_dist.stddev().numpy(),
        )

        y_in_sample = pd.Series(y_in_sample)
        y_out_sample = pd.Series(y_out_sample)
        y_pred = self._get_y_pred(y_in_sample=y_in_sample, y_out_sample=y_out_sample)

        if return_pred_int:
            return y_pred, pred_int
        else:
            return y_pred

    def get_fitted_params(self):
        """Get fitted parameters
        Returns
        -------
        fitted_params : dict
        References
        ----------
        """

        self.check_is_fitted()
        fitted_params = {}
        should_brake = False

        for param in self._forecaster.parameters:
            if param.name is None:
                should_brake = True
                break
            fitted_params[param.name] = "{} +- {}".format(
                np.mean(self._parameter_samples[param.name], axis=0),
                np.std(self._parameter_samples[param.name], axis=0),
            )

        # To enforce consistency in notations
        if should_brake:
            fitted_params = self._forecaster.parameters

        return fitted_params

    def _check_conf(self, dictionary):
        """Raise error when key "observed_time_series" in dictionary
        to avoid cryptic exceptions. observed_time_series is given in fit()
        by means of "y".

        :param dictionary: A dictionary to configure a BSTS component.
        :type dictionary: dict
        """
        if "observed_time_series" in dictionary:
            raise ValueError(
                """Do not provide "observed_time_series" as a key
                in a component, it is taken automatically by the \"y\"
                argument in the sktime.BSTS.fit() function."""
            )

    def _check_design_matrix(self, design_matrix):
        if design_matrix is None:
            raise ValueError(
                """Design matrix (X) has to be given to add a
                linear regression/dynamic regression/sparse
                linear regression component"""
            )

    def _type_check_y_X(self, y=None, X=None):
        if y is not None:
            self._y = y.astype("float64")
        if X is not None:
            self._X = X.astype("float64")
