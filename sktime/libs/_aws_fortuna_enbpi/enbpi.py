"""EnbPI implementation from aws fortuna."""

import numpy as np


class EnbPI:
    """
    Ensemble Batch Prediction Intervals (EnbPI).

    Ensemble Batch Prediction Intervals (EnbPI) is a conformal
    prediction algorithm for time series regression.
    By bootstrapping the data and training the model on
    each of the bootstrap samples, EnbPI is able to compute
    conformal intervals that satisfy an approximate marginal
        guarantee on each test data point. Furthermore,
    EnbPI can incorporate online feedback from incoming
    batches of data, and improve the conformal intervals
    without having to retrain the model.

    Parameters
    ----------
    aggregation_fun: str
        either mean or median, default mean
    """

    def __init__(self, aggregation_fun="mean"):
        if aggregation_fun == "mean":
            self.aggregation_fun = lambda x: np.mean(x, 0)
        elif aggregation_fun == "median":
            self.aggregation_fun = lambda x: np.median(x, 0)

    def conformal_interval(
        self,
        bootstrap_indices,
        bootstrap_train_preds,
        bootstrap_test_preds,
        train_targets,
        error: float,
        return_residuals: bool = False,
    ):
        """
        Compute a coverage interval for each of the test inputs.

        Compute a coverage interval for each of the test inputs
        of the time series, at the desired coverage error.
        This is supported only for one-dimensional target
        variables, and for one times series at the time.

        Parameters
        ----------
        bootstrap_indices
            The indices, randomly sampled with replacement, of the
            training data in the time series used to train the
            model. The first dimension is over the different
            samples of indices. The second dimension contain the
            data points for each sample, which must be as many
            as thnumber of data points in the time series used for
            training. A simple way of obtaining indices randomly
            sampled with replacement is
            :code:`numpy.random.choice(T, size=(B, T))`,
            where :code:`T` is the number of training points
            in the time series,
            and :code:`B` is the number of bootstrap samples.
            It is the user job to make sure that the models ar
            trained upon the data corresponding to the random
            indices.
        bootstrap_train_preds
            Model predictions for each of the bootstrap samples
            of data of the time series used for training the model,
            evaluated at each of the training data inputs of
            the time series. The first dimension is over the
            different bootstrap samples. The second dimensions
            is over the training inputs. There may be a third dimension,
            corresponding to the dimensionality of the predictions,
            but if so this must be one.
        bootstrap_test_preds
            Model predictions for each of the bootstrap samples of
            data of the time series used for training the model,
            evaluated at each of the test data inputs of the time
            series. The first dimension is over the
            different bootstrap samples. The second dimensions
            is over the test inputs. There may be a third dimension,
            corresponding to the dimensionality of the predictions,
            but if so this must be one.
        train_targets
            The target variables of the training data points in the time series.
        error: float
            The desired coverage error. This must be a scalar
            between 0 and 1, extremes included.
        return_residuals: bool
            If True, return the residual errors computed over
            the training data. These are used in

        Returns
        -------
        Union[Array, Tuple[Array, Array]]
            The conformal intervals. The two components of the
            second dimension correspond to the left and right
            interval bounds. If :code:`return_residuals` is set
            to True, then it returns also the residuals computed on
            the training set.
        """
        n_bootstraps, n_train_times = bootstrap_indices.shape

        in_bootstrap_indices = np.zeros((n_bootstraps, n_train_times), dtype=bool)
        np.put_along_axis(in_bootstrap_indices, bootstrap_indices, values=1, axis=1)
        aggr_bootstrap_test_preds = np.zeros(
            (n_train_times,) + bootstrap_test_preds.shape[1:]
        )
        train_residuals = np.zeros((n_train_times,) + train_targets.shape[1:])

        for t in range(n_train_times):
            which_bootstraps = np.where(~(in_bootstrap_indices[:, t]))[0]
            if len(which_bootstraps) > 0:
                aggr_bootstrap_train_pred = self.aggregation_fun(
                    bootstrap_train_preds[which_bootstraps, t]
                )
                train_residuals[t] = np.abs(
                    train_targets[t] - aggr_bootstrap_train_pred
                )
                aggr_bootstrap_test_preds[t] = self.aggregation_fun(
                    bootstrap_test_preds[which_bootstraps]
                )
            else:
                train_residuals[t] = np.abs(train_targets[t])

        test_quantiles = np.quantile(aggr_bootstrap_test_preds, q=1 - error, axis=0)
        residuals_quantile = np.quantile(train_residuals, q=1 - error, axis=0)

        left = test_quantiles - residuals_quantile
        right = test_quantiles + residuals_quantile

        conformal_intervals = np.array(list(zip(left, right)))
        if not return_residuals:
            return conformal_intervals
        return conformal_intervals, train_residuals
