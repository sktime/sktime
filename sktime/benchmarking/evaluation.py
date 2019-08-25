__author__ = ["Viktor Kazakov", "Markus Löning"]
__all__ = ["Evaluator"]

import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import ranksums
from scipy.stats import ttest_ind

from sktime.benchmarking.base import BaseResults
from sktime.utils.exceptions import NotEvaluatedError

plt.style.use("seaborn-ticks")


class Evaluator:
    """
    Analyze results of machine learning experiments.
    """

    def __init__(self, results):
        if not isinstance(results, BaseResults):
            raise ValueError(f"`results` must inherit from BaseResults")
        self.results = results
        self._metric_dicts = []

        # preallocate dataframe for metrics
        self._metrics = pd.DataFrame(columns=["dataset", "strategy", "cv_fold"])
        self._metrics_by_strategy_dataset = pd.DataFrame(columns=["dataset", "strategy"])
        self._metrics_by_strategy = pd.DataFrame(columns=["strategy"])

        # keep track of metric names
        self._metric_names = []

    @property
    def metric_names(self):
        return self._metric_names

    @property
    def metrics(self):
        self._check_is_evaluated()
        return self._metrics

    @property
    def metrics_by_strategy(self):
        self._check_is_evaluated()
        return self._metrics_by_strategy

    @property
    def metrics_by_strategy_dataset(self):
        self._check_is_evaluated()
        return self._metrics_by_strategy_dataset

    def evaluate(self, metric, train_or_test="test", cv_fold="all"):
        """
        Calculates the average prediction error per estimator as well as the prediction error achieved by each
        estimator on individual datasets.
        """

        # check input
        if isinstance(cv_fold, int):
            cv_folds = [cv_fold]  # if single fold, make iterable
        elif cv_fold == "all":
            cv_folds = self.results.cv_folds
        else:
            raise ValueError(f"`cv_fold` must be either positive integer (>=0) or 'all', but found: {type(cv_fold)}")

        # load all predictions
        for cv_fold in cv_folds:
            for result in self.results.load_predictions(train_or_test=train_or_test, cv_fold=cv_fold):
                # unwrap result object
                strategy_name = result.strategy_name
                dataset_name = result.dataset_name
                index = result.index
                y_true = result.y_true
                y_pred = result.y_pred
                y_proba = result.y_proba

                # compute metric
                mean, stderr = metric.compute(y_true, y_pred)

                # store results
                metric_dict = {
                    "dataset": dataset_name,
                    "strategy": strategy_name,
                    "cv_fold": cv_fold,
                    self._get_column_name(metric.name, suffix="mean"): mean,
                    self._get_column_name(metric.name, suffix="stderr"): stderr
                }
                self._metric_dicts.append(metric_dict)

        # update metrics dataframe with computed metrics
        metrics = pd.DataFrame(self._metric_dicts)
        self._metrics = self._metrics.merge(metrics, how="outer")

        # aggregate results
        # aggregate over cv folds
        metrics_by_strategy_dataset = self._metrics.groupby(["dataset", "strategy"], as_index=False).agg(np.mean).drop(
            columns="cv_fold")
        self._metrics_by_strategy_dataset = self._metrics_by_strategy_dataset.merge(metrics_by_strategy_dataset,
                                                                                   how="outer")
        # aggregate over cv folds and datasets
        metrics_by_strategy = metrics_by_strategy_dataset.groupby(["strategy"], as_index=False).agg(np.mean)
        self._metrics_by_strategy = self._metrics_by_strategy.merge(metrics_by_strategy, how="outer")

        # append metric names
        self._metric_names.append(metric.name)

        # return aggregated results
        return self._metrics_by_strategy

    def plot_boxplots(self, metric_name=None, **kwargs):
        """Box plot of metric"""
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        column = self._get_column_name(metric_name, suffix="mean")

        fig, ax = plt.subplots(1)
        self.metrics_by_strategy_dataset.boxplot(by="strategy", column=column, grid=False, ax=ax, **kwargs)
        ax.set(title=f"{metric_name} by strategy", xlabel="strategies", ylabel=metric_name)
        fig.suptitle(None)
        plt.tight_layout()
        return fig, ax

    def rank(self, metric_name=None, ascending=False):
        """
        Calculates the average ranks based on the performance of each estimator on each dataset
        """
        self._check_is_evaluated()
        if not isinstance(ascending, bool):
            raise ValueError(f"`ascending` must be boolean, but found: {type(ascending)}")

        metric_name = self._validate_metric_name(metric_name)
        column = self._get_column_name(metric_name, suffix="mean")

        ranked = (self.metrics_by_strategy_dataset
                  .loc[:, ["dataset", "strategy", column]]
                  .set_index("strategy")
                  .groupby("dataset")
                  .rank(ascending=ascending)
                  .reset_index()
                  .groupby("strategy")
                  .mean()
                  .rename(columns={column: f"{metric_name}_mean_rank"})
                  )
        return ranked

    def t_test(self, metric_name=None):
        """
        Runs t-test on all possible combinations between the estimators.
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(metric_name)

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
                "p_val": p_val
            }

            t_df = t_df.append(t_test, ignore_index=True)
            values = np.append(values, t_stat)
            values = np.append(values, p_val)

        index = t_df["estimator_1"].unique()
        values_names = ["t_stat", "p_val"]
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return t_df, values_df_multiindex

    def sign_test(self, metric_name=None):
        """
        Non-parametric test for test for consistent differences between pairs of observations.
        See `<https://en.wikipedia.org/wiki/Sign_test>`_ for details about the test and
        `<https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.binom_test.html>`_
        for details about the scipy implementation.
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(metric_name)

        sign_df = pd.DataFrame()
        perms = itertools.product(metrics_per_estimator_dataset.keys(), repeat=2)

        for perm in perms:
            x = np.array(metrics_per_estimator_dataset[perm[0]])
            y = np.array(metrics_per_estimator_dataset[perm[1]])
            signs = np.sum([i[0] > i[1] for i in zip(x, y)])
            n = len(x)
            p_val = stats.binom_test(signs, n)
            sign_test = {
                "estimator_1": perm[0],
                "estimator_2": perm[1],
                "p_val": p_val
            }

            sign_df = sign_df.append(sign_test, ignore_index=True)
            sign_df_pivot = sign_df.pivot(index="estimator_1", columns="estimator_2", values="p_val")

        return sign_df, sign_df_pivot

    def ranksum_test(self, metric_name=None):
        """
        Non-parametric test for testing consistent differences between pairs of obeservations.
        The test counts the number of observations that are greater, smaller and equal to the mean
        `<http://en.wikipedia.org/wiki/Wilcoxon_rank-sum_test>`_.
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(metric_name)

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
                "p_val": p_val
            }
            ranksum_df = ranksum_df.append(ranksum, ignore_index=True)
            values = np.append(values, t_stat)
            values = np.append(values, p_val)

        index = ranksum_df["estimator_1"].unique()
        values_names = ["t_stat", "p_val"]
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return ranksum_df, values_df_multiindex

    def t_test_with_bonferroni_correction(self, metric_name=None, alpha=0.05):
        """
        correction used to counteract multiple comparissons
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

        bonfer_df = pd.DataFrame(bonfer_test_reshaped, index=idx_estim_1, columns=idx_estim_2)

        return bonfer_df

    def wilcoxon_test(self, metric_name=None):
        """http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
        `Wilcoxon signed-rank test <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`_.
        Tests whether two  related paired samples come from the same distribution. 
        In particular, it tests whether the distribution of the differences x-y is symmetric about zero
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(metric_name)

        wilcoxon_df = pd.DataFrame()
        values = np.array([])
        prod = itertools.product(metrics_per_estimator_dataset.keys(), repeat=2)
        for p in prod:
            estim_1 = p[0]
            estim_2 = p[1]
            w, p_val = stats.wilcoxon(metrics_per_estimator_dataset[p[0]],
                                      metrics_per_estimator_dataset[p[1]])

            w_test = {
                "estimator_1": estim_1,
                "estimator_2": estim_2,
                "statistic": w,
                "p_val": p_val
            }

            wilcoxon_df = wilcoxon_df.append(w_test, ignore_index=True)
            values = np.append(values, w)
            values = np.append(values, p_val)

        index = wilcoxon_df["estimator_1"].unique()
        values_names = ["statistic", "p_val"]
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return wilcoxon_df, values_df_multiindex

    def friedman_test(self, metric_name=None):
        """
        The Friedman test is a non-parametric statistical test used to detect differences 
        in treatments across multiple test attempts. The procedure involves ranking each row (or block) together, 
        then considering the values of ranks by columns.
        Implementation used:
        `scipy.stats <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.friedmanchisquare.html>`_.
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(metric_name)

        friedman_test = stats.friedmanchisquare(
            *[metrics_per_estimator_dataset[k] for k in metrics_per_estimator_dataset.keys()])
        values = [friedman_test[0], friedman_test[1]]
        values_df = pd.DataFrame([values], columns=["statistic", "p_value"])

        return friedman_test, values_df

    def nemenyi(self, metric_name=None):
        """
        Post-hoc test run if the `friedman_test` reveals statistical significance.
        For more information see `Nemenyi test <https://en.wikipedia.org/wiki/Nemenyi_test>`_.
        Implementation used `scikit-posthocs <https://github.com/maximtrp/scikit-posthocs>`_.
        """
        self._check_is_evaluated()
        metric_name = self._validate_metric_name(metric_name)
        metrics_per_estimator_dataset = self._get_metrics_per_estimator_dataset(metric_name)

        strategy_dict = pd.DataFrame(metrics_per_estimator_dataset)
        strategy_dict = strategy_dict.melt(var_name="groups", value_name="values")
        nemenyi = sp.posthoc_nemenyi(strategy_dict, val_col="values", group_col="groups")
        return nemenyi

    def _get_column_name(self, metric_name, suffix="mean"):
        return f"{metric_name}_{suffix}"

    def _check_is_evaluated(self):
        if len(self._metric_names) == 0:
            raise NotEvaluatedError("This evaluator has not evaluated any metric yet. Please call "
                                    "'evaluate' with the appropriate arguments before using this method.")

    def _validate_metric_name(self, metric_name):
        if metric_name is None:
            metric_name = self._metric_names[-1]  # if None, use the last evaluated metric

        if metric_name not in self._metric_names:
            raise ValueError(f"{metric_name} has not been evaluated yet. Please call "
                             f"'evaluate' with the appropriate arguments first")

        return metric_name

    def _get_metrics_per_estimator_dataset(self, metric_name):
        """Helper function to get old format back"""
        column = f"{metric_name}_mean"
        df = self.metrics_by_strategy_dataset.loc[:, ["strategy", "dataset", column]].set_index("strategy")
        losses_per_estimator = {}
        for strategy in df.index:
            val = df.loc[strategy, column].tolist()
            val = [val] if not isinstance(val, list) else val
            losses_per_estimator[strategy] = val
        return losses_per_estimator

    def _get_metrics_per_estimator(self, metric_name):
        df = self.metrics_by_strategy_dataset.loc[:, ["strategy", "dataset", f"{metric_name}_mean",
                                                      f"{metric_name}_stderr"]]
        d = {}
        for dataset in df.dataset.unique():
            l = []
            for strategy in df.strategy.unique():
                row = df.loc[(df.strategy == strategy) & (df.dataset == dataset), :]
                m = row["accuracy_mean"].values[0]
                s = row["accuracy_stderr"].values[0]
                l.append([strategy, m, s])
            d[dataset] = l
        return d
