# -*- coding: utf-8 -*-
"""Evaluator class for analyzing results of a machine learning experiment."""
__author__ = ["viktorkaz", "mloning", "Aaron Bostrom"]
__all__ = ["Evaluator"]

import itertools

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import ranksums, ttest_ind

from sktime.benchmarking.base import BaseResults
from sktime.exceptions import NotEvaluatedError
from sktime.utils.validation._dependencies import _check_soft_dependencies

_check_soft_dependencies("matplotlib", "scikit_posthocs", severity="warning")


class Evaluator:
    """Analyze results of machine learning experiments."""

    def __init__(self, results):

        if not isinstance(results, BaseResults):
            raise ValueError("`results` must inherit from BaseResults")
        self.results = results
        self._metric_dicts = []

        # preallocate dataframe for metrics
        self._metrics = pd.DataFrame(columns=["dataset", "strategy", "cv_fold"])
        self._metrics_by_strategy_dataset = pd.DataFrame(
            columns=["dataset", "strategy"]
        )
        self._metrics_by_strategy = pd.DataFrame(columns=["strategy"])

        # keep track of metric names
        self._metric_names = []

    @property
    def metric_names(self):
        """Return metric names."""
        return self._metric_names

    @property
    def metrics(self):
        """Return metrics."""
        self._check_is_evaluated()
        return self._metrics

    @property
    def metrics_by_strategy(self):
        """Return metric by strategy."""
        self._check_is_evaluated()
        return self._metrics_by_strategy

    @property
    def metrics_by_strategy_dataset(self):
        """Return metrics by strategy and dataset."""
        self._check_is_evaluated()
        return self._metrics_by_strategy_dataset

    def evaluate(self, metric, train_or_test="test", cv_fold="all"):
        """Evaluate estimator performance.

        Calculates the average prediction error per estimator as well as the
        prediction error achieved by each
        estimator on individual datasets.
        """
        # check input
        if isinstance(cv_fold, int) and cv_fold >= 0:
            cv_folds = [cv_fold]  # if single fold, make iterable
        elif cv_fold == "all":
            cv_folds = np.arange(self.results.cv.get_n_splits())
            if len(cv_folds) == 0:
                raise ValueError()
        else:
            raise ValueError(
                f"`cv_fold` must be either positive integer (>=0) or 'all', "
                f"but found: {type(cv_fold)}"
            )

        # load all predictions
        for cv_fold in cv_folds:
            for result in self.results.load_predictions(
                cv_fold=cv_fold, train_or_test=train_or_test
            ):
                # unwrap result object
                strategy_name = result.strategy_name
                dataset_name = result.dataset_name
                # index = result.index
                y_true = result.y_true
                y_pred = result.y_pred
                # y_proba = result.y_proba

                # compute metric
                mean, stderr = metric.compute(y_true, y_pred)

                # store results
                metric_dict = {
                    "dataset": dataset_name,
                    "strategy": strategy_name,
                    "cv_fold": cv_fold,
                    self._get_column_name(metric.name, suffix="mean"): mean,
                    self._get_column_name(metric.name, suffix="stderr"): stderr,
                }
                self._metric_dicts.append(metric_dict)

        # update metrics dataframe with computed metrics
        metrics = pd.DataFrame(self._metric_dicts)
        self._metrics = self._metrics.merge(metrics, how="outer")

        # aggregate results
        # aggregate over cv folds
        metrics_by_strategy_dataset = (
            self._metrics.groupby(["dataset", "strategy"], as_index=False)
            .agg(np.mean)
            .drop(columns="cv_fold")
        )
        self._metrics_by_strategy_dataset = self._metrics_by_strategy_dataset.merge(
            metrics_by_strategy_dataset, how="outer"
        )
        # aggregate over cv folds and datasets
        metrics_by_strategy = metrics_by_strategy_dataset.groupby(
            ["strategy"], as_index=False
        ).agg(np.mean)
        self._metrics_by_strategy = self._metrics_by_strategy.merge(
            metrics_by_strategy, how="outer"
        )

        # append metric names
        self._metric_names.append(metric.name)

        # return aggregated results
        return self._metrics_by_strategy

    def plot_boxplots(self, metric_name=None, **kwargs):
        """Box plot of metric."""
        _check_soft_dependencies("matplotlib")

        import matplotlib.pyplot as plt  # noqa: E402

        plt.style.use("seaborn-ticks")

        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        column = self._get_column_name(metric_name, suffix="mean")

        fig, ax = plt.subplots(1)
        self.metrics_by_strategy_dataset.boxplot(
            by="strategy", column=column, grid=False, ax=ax, **kwargs
        )
        ax.set(
            title=f"{metric_name} by strategy", xlabel="strategies", ylabel=metric_name
        )
        fig.suptitle(None)
        plt.tight_layout()
        return fig, ax

    def rank(self, metric_name=None, ascending=False):
        """Determine estimator ranking.

        Calculates the average ranks based on the performance of each
        estimator on each dataset
        """
        self._check_is_evaluated()
        if not isinstance(ascending, bool):
            raise ValueError(
                f"`ascending` must be boolean, but found: {type(ascending)}"
            )

        metric_name = self._validate_metric_name(metric_name)
        column = self._get_column_name(metric_name, suffix="mean")

        ranked = (
            self.metrics_by_strategy_dataset.loc[:, ["dataset", "strategy", column]]
            .set_index("strategy")
            .groupby("dataset")
            .rank(ascending=ascending)
            .reset_index()
            .groupby("strategy")
            .mean()
            .rename(columns={column: f"{metric_name}_mean_rank"})
            .reset_index()
        )
        return ranked

    def t_test(self, metric_name=None):
        """T-test on all possible combinations between the estimators."""
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(
            metric_name
        )

        t_df = pd.DataFrame()
        perms = itertools.product(metrics_per_estimator_dataset.keys(), repeat=2)
        values = np.array([])
        for perm in perms:
            x = np.array(metrics_per_estimator_dataset[perm[0]])
            y = np.array(metrics_per_estimator_dataset[perm[1]])
            t_stat, p_val = ttest_ind(x, y)

            t_test = {
                "estimator_1": perm[0],
                "estimator_2": perm[1],
                "t_stat": t_stat,
                "p_val": p_val,
            }

            t_df = pd.concat([t_df, pd.DataFrame(t_test, index=[0])], ignore_index=True)
            values = np.append(values, t_stat)
            values = np.append(values, p_val)

        index = t_df["estimator_1"].unique()
        values_names = ["t_stat", "p_val"]
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(
            values_reshaped, index=index, columns=col_idx
        )

        return t_df, values_df_multiindex

    def sign_test(self, metric_name=None):
        """Non-parametric test for consistent differences between observation pairs.

        See `<https://en.wikipedia.org/wiki/Sign_test>`_ for details about
        the test and
        `<https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy
        .stats.binom_test.html>`_
        for details about the scipy implementation.
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(
            metric_name
        )

        sign_df = pd.DataFrame()
        perms = itertools.product(metrics_per_estimator_dataset.keys(), repeat=2)

        for perm in perms:
            x = np.array(metrics_per_estimator_dataset[perm[0]])
            y = np.array(metrics_per_estimator_dataset[perm[1]])
            signs = np.sum([i[0] > i[1] for i in zip(x, y)])
            n = len(x)
            p_val = stats.binom_test(signs, n)
            sign_test = {"estimator_1": perm[0], "estimator_2": perm[1], "p_val": p_val}

            sign_df = pd.concat(
                [sign_df, pd.DataFrame(sign_test, index=[0])], ignore_index=True
            )
            sign_df_pivot = sign_df.pivot(
                index="estimator_1", columns="estimator_2", values="p_val"
            )

        return sign_df, sign_df_pivot

    def ranksum_test(self, metric_name=None):
        """Non-parametric test of consistent differences between observation pairs.

        The test counts the number of observations that are greater, smaller
        and equal to the mean
        `<http://en.wikipedia.org/wiki/Wilcoxon_rank-sum_test>`_.
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(
            metric_name
        )

        ranksum_df = pd.DataFrame()
        perms = itertools.product(metrics_per_estimator_dataset.keys(), repeat=2)
        values = np.array([])
        for perm in perms:
            x = metrics_per_estimator_dataset[perm[0]]
            y = metrics_per_estimator_dataset[perm[1]]
            t_stat, p_val = ranksums(x, y)
            ranksum = {
                "estimator_1": perm[0],
                "estimator_2": perm[1],
                "t_stat": t_stat,
                "p_val": p_val,
            }
            ranksum_df = pd.concat(
                [ranksum_df, pd.DataFrame(ranksum, index=[0])], ignore_index=True
            )
            values = np.append(values, t_stat)
            values = np.append(values, p_val)

        index = ranksum_df["estimator_1"].unique()
        values_names = ["t_stat", "p_val"]
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(
            values_reshaped, index=index, columns=col_idx
        )

        return ranksum_df, values_df_multiindex

    def t_test_with_bonferroni_correction(self, metric_name=None, alpha=0.05):
        """T-test with correction used to counteract multiple comparisons.

        https://en.wikipedia.org/wiki/Bonferroni_correction
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)

        df_t_test, _ = self.t_test(metric_name=metric_name)
        idx_estim_1 = df_t_test["estimator_1"].unique()
        idx_estim_2 = df_t_test["estimator_2"].unique()
        estim_1 = len(idx_estim_1)
        estim_2 = len(idx_estim_2)
        critical_value = alpha / (estim_1 * estim_2)

        bonfer_test = df_t_test["p_val"] <= critical_value

        bonfer_test_reshaped = bonfer_test.values.reshape(estim_1, estim_2)

        bonfer_df = pd.DataFrame(
            bonfer_test_reshaped, index=idx_estim_1, columns=idx_estim_2
        )

        return bonfer_df

    def wilcoxon_test(self, metric_name=None):
        """Wilcoxon signed-rank test.

        http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
        `Wilcoxon signed-rank test
        <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`_.
        Tests whether two  related paired samples come from the same
        distribution. In particular, it tests whether the distribution of the
        differences x-y is symmetric about zero
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(
            metric_name
        )

        wilcoxon_df = pd.DataFrame()
        prod = itertools.combinations(metrics_per_estimator_dataset.keys(), 2)
        for p in prod:
            estim_1 = p[0]
            estim_2 = p[1]
            w, p_val = stats.wilcoxon(
                metrics_per_estimator_dataset[p[0]], metrics_per_estimator_dataset[p[1]]
            )

            w_test = {
                "estimator_1": estim_1,
                "estimator_2": estim_2,
                "statistic": w,
                "p_val": p_val,
            }

            wilcoxon_df = pd.concat(
                [wilcoxon_df, pd.DataFrame(w_test, index=[0])], ignore_index=True
            )

        return wilcoxon_df

    def friedman_test(self, metric_name=None):
        """Friedman test.

        The Friedman test is a non-parametric statistical test used to
        detect differences
        in treatments across multiple test attempts. The procedure involves
        ranking each row (or block) together,
        then considering the values of ranks by columns.
        Implementation used:
        `scipy.stats <https://docs.scipy.org/doc/scipy-0.15.1/reference
        /generated/scipy.stats.friedmanchisquare.html>`_.
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(
            metric_name
        )

        friedman_test = stats.friedmanchisquare(
            *[
                metrics_per_estimator_dataset[k]
                for k in metrics_per_estimator_dataset.keys()
            ]
        )
        values = [friedman_test[0], friedman_test[1]]
        values_df = pd.DataFrame([values], columns=["statistic", "p_value"])

        return friedman_test, values_df

    def nemenyi(self, metric_name=None):
        """Nemenyi test.

        Post-hoc test run if the `friedman_test` reveals statistical
        significance.
        For more information see `Nemenyi test
        <https://en.wikipedia.org/wiki/Nemenyi_test>`_.
        Implementation used `scikit-posthocs
        <https://github.com/maximtrp/scikit-posthocs>`_.
        """
        _check_soft_dependencies("scikit_posthocs")

        # lazy import to avoid hard dependency
        from scikit_posthocs import posthoc_nemenyi

        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(
            metric_name
        )

        strategy_dict = pd.DataFrame(metrics_per_estimator_dataset)
        strategy_dict = strategy_dict.melt(var_name="groups", value_name="values")
        nemenyi = posthoc_nemenyi(strategy_dict, val_col="values", group_col="groups")
        return nemenyi

    def fit_runtime(self, unit="s", train_or_test="test", cv_fold="all"):
        """Calculate the average time for fitting the strategy.

        Parameters
        ----------
        unit : string (must be either 's' for seconds, 'm' for minutes or 'h' for hours)
            the unit in which the run time will be calculated

        Returns
        -------
        run_times: Pandas DataFrame
            average run times per estimator and strategy
        """
        # check input
        if isinstance(cv_fold, int) and cv_fold >= 0:
            cv_folds = [cv_fold]  # if single fold, make iterable
        elif cv_fold == "all":
            cv_folds = np.arange(self.results.cv.get_n_splits())
            if len(cv_folds) == 0:
                raise ValueError()
        else:
            raise ValueError(
                f"`cv_fold` must be either positive integer (>=0) or 'all', "
                f"but found: {type(cv_fold)}"
            )

        # load all predictions
        run_times = pd.DataFrame(
            columns=[
                "strategy_name",
                "dataset_name",
                "fit_estimator_start_time",
                "fit_estimator_end_time",
                "cv_fold",
            ]
        )
        for cv_fold in cv_folds:
            for result in self.results.load_predictions(
                cv_fold=cv_fold, train_or_test=train_or_test
            ):
                # unwrap result object
                strategy_name = result.strategy_name
                dataset_name = result.dataset_name
                fit_estimator_start_time = result.fit_estimator_start_time
                fit_estimator_end_time = result.fit_estimator_end_time
                predict_estimator_start_time = result.predict_estimator_start_time
                predict_estimator_end_time = result.predict_estimator_end_time
                unwrapped = pd.DataFrame(
                    {
                        "strategy_name": [strategy_name],
                        "dataset_name": [dataset_name],
                        "fit_estimator_start_time": [fit_estimator_start_time],
                        "fit_estimator_end_time": [fit_estimator_end_time],
                        "predict_estimator_start_time": [predict_estimator_start_time],
                        "predict_estimator_end_time": [predict_estimator_end_time],
                        "cv_fold": [cv_fold],
                    }
                )
                run_times = pd.concat([run_times, unwrapped], ignore_index=True)

        # calculate run time difference
        run_times["fit_runtime"] = (
            run_times["fit_estimator_end_time"] - run_times["fit_estimator_start_time"]
        ) / np.timedelta64(1, unit)
        run_times["predict_runtime"] = (
            run_times["predict_estimator_end_time"]
            - run_times["predict_estimator_start_time"]
        ) / np.timedelta64(1, unit)

        return pd.pivot_table(
            run_times,
            index=["strategy_name", "dataset_name"],
            values=["fit_runtime", "predict_runtime"],
            aggfunc={"fit_runtime": np.average, "predict_runtime": np.average},
        )
        #         # compute metric
        #         mean, stderr = metric.compute(y_true, y_pred)

        #         # store results
        #         metric_dict = {
        #             "dataset": dataset_name,
        #             "strategy": strategy_name,
        #             "cv_fold": cv_fold,
        #             self._get_column_name(metric.name, suffix="mean"): mean,
        #             self._get_column_name(metric.name, suffix="stderr"): stderr,
        #         }
        #         self._metric_dicts.append(metric_dict)

        # # update metrics dataframe with computed metrics
        # metrics = pd.DataFrame(self._metric_dicts)
        # self._metrics = self._metrics.merge(metrics, how="outer")

        # # aggregate results
        # # aggregate over cv folds
        # metrics_by_strategy_dataset = (
        #     self._metrics.groupby(["dataset", "strategy"], as_index=False)
        #     .agg(np.mean)
        #     .drop(columns="cv_fold")
        # )
        # self._metrics_by_strategy_dataset = self._metrics_by_strategy_dataset.merge(
        #     metrics_by_strategy_dataset, how="outer"
        # )
        # # aggregate over cv folds and datasets
        # metrics_by_strategy = metrics_by_strategy_dataset.groupby(
        #     ["strategy"], as_index=False
        # ).agg(np.mean)
        # self._metrics_by_strategy = self._metrics_by_strategy.merge(
        #     metrics_by_strategy, how="outer"
        # )

        # # append metric names
        # self._metric_names.append(metric.name)

        # # return aggregated results
        # return self._metrics_by_strategy

    def plot_critical_difference_diagram(self, metric_name=None, alpha=0.1):
        """Plot critical difference diagrams.

        References
        ----------
        original implementation by Aaron Bostrom, modified by Markus Löning.
        """
        _check_soft_dependencies("matplotlib")

        import matplotlib.pyplot as plt  # noqa: E402

        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        column = self._get_column_name(metric_name, suffix="mean")
        data = (
            self.metrics_by_strategy_dataset.copy()
            .loc[:, ["dataset", "strategy", column]]
            .pivot(index="strategy", columns="dataset", values=column)
            .values
        )

        n_strategies, n_datasets = data.shape  # [N,k] = size(s); correct
        labels = self.results.strategy_names

        r = np.argsort(data, axis=0)
        S = np.sort(data, axis=0)
        idx = n_strategies * np.tile(np.arange(n_datasets), (n_strategies, 1)).T + r.T
        R = np.asfarray(np.tile(np.arange(n_strategies) + 1, (n_datasets, 1)))
        S = S.T

        for i in range(n_datasets):
            for j in range(n_strategies):
                index = S[i, j] == S[i, :]
                R[i, index] = np.mean(R[i, index], dtype=np.float64)

        r = np.asfarray(r)
        r.T.flat[idx] = R
        r = r.T

        if alpha == 0.01:
            # fmt: off
            qalpha = [0.000, 2.576, 2.913, 3.113, 3.255, 3.364, 3.452, 3.526,
                      3.590, 3.646, 3.696, 3.741, 3.781, 3.818,
                      3.853, 3.884, 3.914, 3.941, 3.967, 3.992, 4.015, 4.037,
                      4.057, 4.077, 4.096, 4.114, 4.132, 4.148,
                      4.164, 4.179, 4.194, 4.208, 4.222, 4.236, 4.249, 4.261,
                      4.273, 4.285, 4.296, 4.307, 4.318, 4.329,
                      4.339, 4.349, 4.359, 4.368, 4.378, 4.387, 4.395, 4.404,
                      4.412, 4.420, 4.428, 4.435, 4.442, 4.449,
                      4.456]
            # fmt: on
        elif alpha == 0.05:
            # fmt: off
            qalpha = [0.000, 1.960, 2.344, 2.569, 2.728, 2.850, 2.948, 3.031,
                      3.102, 3.164, 3.219, 3.268, 3.313, 3.354,
                      3.391, 3.426, 3.458, 3.489, 3.517, 3.544, 3.569, 3.593,
                      3.616, 3.637, 3.658, 3.678, 3.696, 3.714,
                      3.732, 3.749, 3.765, 3.780, 3.795, 3.810, 3.824, 3.837,
                      3.850, 3.863, 3.876, 3.888, 3.899, 3.911,
                      3.922, 3.933, 3.943, 3.954, 3.964, 3.973, 3.983, 3.992,
                      4.001, 4.009, 4.017, 4.025, 4.032, 4.040,
                      4.046]
            # fmt: on
        elif alpha == 0.1:
            # fmt: off
            qalpha = [0.000, 1.645, 2.052, 2.291, 2.460, 2.589, 2.693, 2.780,
                      2.855, 2.920, 2.978, 3.030, 3.077, 3.120,
                      3.159, 3.196, 3.230, 3.261, 3.291, 3.319, 3.346, 3.371,
                      3.394, 3.417, 3.439, 3.459, 3.479, 3.498,
                      3.516, 3.533, 3.550, 3.567, 3.582, 3.597, 3.612, 3.626,
                      3.640, 3.653, 3.666, 3.679, 3.691, 3.703,
                      3.714, 3.726, 3.737, 3.747, 3.758, 3.768, 3.778, 3.788,
                      3.797, 3.806, 3.814, 3.823, 3.831, 3.838,
                      3.846]
            # fmt: on
        else:
            raise Exception("alpha must be 0.01, 0.05 or 0.1")

        cd = qalpha[n_strategies - 1] * np.sqrt(
            n_strategies * (n_strategies + 1) / (6 * n_datasets)
        )

        # set up plot
        fig, ax = plt.subplots(1)
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(0, 140)
        ax.set_axis_off()

        tics = np.tile(np.array(np.arange(n_strategies)) / (n_strategies - 1), (3, 1))
        plt.plot(
            tics.flatten("F"),
            np.tile([100, 105, 100], (1, n_strategies)).flatten(),
            linewidth=2,
            color="black",
        )
        tics = np.tile(
            (np.array(range(0, n_strategies - 1)) / (n_strategies - 1))
            + 0.5 / (n_strategies - 1),
            (3, 1),
        )
        plt.plot(
            tics.flatten("F"),
            np.tile([100, 102.5, 100], (1, n_strategies - 1)).flatten(),
            linewidth=1,
            color="black",
        )
        plt.plot(
            [
                0,
                0,
                0,
                cd / (n_strategies - 1),
                cd / (n_strategies - 1),
                cd / (n_strategies - 1),
            ],
            [127, 123, 125, 125, 123, 127],
            linewidth=1,
            color="black",
        )
        plt.text(
            0.5 * cd / (n_strategies - 1),
            130,
            "CD",
            fontsize=12,
            horizontalalignment="center",
        )

        for i in range(n_strategies):
            plt.text(
                i / (n_strategies - 1),
                110,
                str(n_strategies - i),
                fontsize=12,
                horizontalalignment="center",
            )

        # compute average ranks
        r = np.mean(r, axis=0)
        idx = np.argsort(r, axis=0)
        r = np.sort(r, axis=0)

        # compute statistically similar cliques
        clique = np.tile(r, (n_strategies, 1)) - np.tile(
            np.vstack(r.T), (1, n_strategies)
        )
        clique[clique < 0] = np.inf
        clique = clique < cd

        for i in range(n_strategies - 1, 0, -1):
            if np.all(clique[i - 1, clique[i, :]] == clique[i, clique[i, :]]):
                clique[i, :] = 0

        n = np.sum(clique, 1)
        clique = clique[n > 1, :]
        n = np.size(clique, 0)

        for i in range(int(np.ceil(n_strategies / 2))):
            plt.plot(
                [
                    (n_strategies - r[i]) / (n_strategies - 1),
                    (n_strategies - r[i]) / (n_strategies - 1),
                    1.2,
                ],
                [
                    100,
                    100 - 5 * (n + 1) - 10 * (i + 1),
                    100 - 5 * (n + 1) - 10 * (i + 1),
                ],
                color="black",
            )
            plt.text(
                1.2,
                100 - 5 * (n + 1) - 10 * (i + 1) + 2,
                "%.2f" % r[i],
                fontsize=10,
                horizontalalignment="right",
            )
            plt.text(
                1.25,
                100 - 5 * (n + 1) - 10 * (i + 1),
                labels[idx[i]],
                fontsize=12,
                verticalalignment="center",
                horizontalalignment="left",
            )

        # labels displayed on the left
        for i in range(int(np.ceil(n_strategies / 2)), n_strategies):
            plt.plot(
                [
                    (n_strategies - r[i]) / (n_strategies - 1),
                    (n_strategies - r[i]) / (n_strategies - 1),
                    -0.2,
                ],
                [
                    100,
                    100 - 5 * (n + 1) - 10 * (n_strategies - i),
                    100 - 5 * (n + 1) - 10 * (n_strategies - i),
                ],
                color="black",
            )
            plt.text(
                -0.2,
                100 - 5 * (n + 1) - 10 * (n_strategies - i) + 2,
                "%.2f" % r[i],
                fontsize=10,
                horizontalalignment="left",
            )
            plt.text(
                -0.25,
                100 - 5 * (n + 1) - 10 * (n_strategies - i),
                labels[idx[i]],
                fontsize=12,
                verticalalignment="center",
                horizontalalignment="right",
            )

        # group cliques of statistically similar classifiers
        for i in range(np.size(clique, 0)):
            R = r[clique[i, :]]
            plt.plot(
                [
                    ((n_strategies - np.min(R)) / (n_strategies - 1)) + 0.015,
                    ((n_strategies - np.max(R)) / (n_strategies - 1)) - 0.015,
                ],
                [100 - 5 * (i + 1), 100 - 5 * (i + 1)],
                linewidth=6,
                color="black",
            )
        plt.show()
        return fig, ax

    def _get_column_name(self, metric_name, suffix="mean"):
        """Get column name in computed metrics dataframe."""
        return f"{metric_name}_{suffix}"

    def _check_is_evaluated(self):
        """Check if evaluator has evaluated any metrics."""
        if len(self._metric_names) == 0:
            raise NotEvaluatedError(
                "This evaluator has not evaluated any metric yet. Please call "
                "'evaluate' with the appropriate arguments before using this "
                "method."
            )

    def _validate_metric_name(self, metric_name):
        """Check if metric has already been evaluated."""
        if metric_name is None:
            metric_name = self._metric_names[
                -1
            ]  # if None, use the last evaluated metric

        if metric_name not in self._metric_names:
            raise ValueError(
                f"{metric_name} has not been evaluated yet. Please call "
                f"'evaluate' with the appropriate arguments first"
            )

        return metric_name

    def _get_metrics_per_estimator_dataset(self, metric_name):
        """Get old format back, to be deprecated."""
        # TODO deprecate in favor of new pandas data frame based data
        #  representation
        column = f"{metric_name}_mean"
        df = self.metrics_by_strategy_dataset.loc[
            :, ["strategy", "dataset", column]
        ].set_index("strategy")
        d = {}
        for strategy in df.index:
            val = df.loc[strategy, column].tolist()
            val = [val] if not isinstance(val, list) else val
            d[strategy] = val
        return d

    def _get_metrics_per_estimator(self, metric_name):
        """Get old format back, to be deprecated."""
        # TODO deprecate in favor of new pandas data frame based data
        #  representation
        columns = [
            "strategy",
            "dataset",
            f"{metric_name}_mean",
            f"{metric_name}_stderr",
        ]
        df = self.metrics_by_strategy_dataset.loc[:, columns]
        d = {}
        for dataset in df.dataset.unique():
            results = []
            for strategy in df.strategy.unique():
                row = df.loc[(df.strategy == strategy) & (df.dataset == dataset), :]
                m = row["accuracy_mean"].values[0]
                s = row["accuracy_stderr"].values[0]
                results.append([strategy, m, s])
            d[dataset] = results
        return d
