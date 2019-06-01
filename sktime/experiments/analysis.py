import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy import stats
from scipy.stats import ranksums
from scipy.stats import ttest_ind

from .losses import Losses


class AnalyseResults(object):
    """
    Analyze results of machine learning experiments.

    Parameters
    ----------
    result : sktime result object
        class for storing the results
    """

    def __init__(self,
                 results):

        self._results_list = results.load()

    def prediction_errors(self, metric):
        """
        Calculates the average prediction error per estimator as well as the prediction error achieved by each estimator on individual datasets.

        Parameters
        -----------
        metric : `sktime.analyse_results.scores`
            Error function 
        Returns
        -------
        pickle of pandas DataFrame
            ``estimator_avg_error`` represents the average error and standard deviation achieved by each estimator. ``estimator_avg_error_per_dataset`` represents the average error and standard deviation achieved by each estimator on each dataset.
        """
        # load all predictions
        losses = Losses(metric)
        for res in self._results_list:
            y_pred = res.y_pred
            y_pred = list(map(float, y_pred))
            y_true = res.y_true
            y_true = list(map(float, y_true))

            losses.evaluate(predictions=y_pred,
                            true_labels=y_true,
                            dataset_name=res.dataset_name,
                            strategy_name=res.strategy_name)
        return losses.get_losses()

    def average_and_std_error(self, scores_dict):
        """
        Calculates simple average and standard error.

        Paramteters
        -----------
        scores_dict : dictionary
            Dictionary with estimators (keys) and corresponding prediction accuracies on different datasets.
        
        Returns
        -------
        pandas DataFrame
            result with average score and standard error
        """
        result = {}
        for k in scores_dict.keys():
            average = np.average(scores_dict[k])
            n = len(scores_dict[k])
            std_error = np.std(scores_dict[k]) / np.sqrt(n)
            result[k] = [average, std_error]

        res_df = pd.DataFrame.from_dict(result, orient='index')
        res_df.columns = ['avg_score', 'std_error']
        res_df = res_df.sort_values(['avg_score', 'std_error'], ascending=[1, 1])

        return res_df

    def plot_boxcharts(self, scores_dict):
        data = []
        labels = []
        avg_error = []
        for e in scores_dict.keys():
            data.append(scores_dict[e])
            avg_error.append(np.mean(scores_dict[e]))
            labels.append(e)
        # sort data and labels based on avg_error
        idx_sort = np.array(avg_error).argsort()
        data = [data[i] for i in idx_sort]
        labels = [labels[i] for i in idx_sort]
        # plot the results
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_xticklabels(labels, rotation=90)
        plt.tight_layout()

        return fig

    def ranks(self, strategy_dict, ascending=True):
        """
        Calculates the average ranks based on the performance of each estimator on each dataset

        Parameters
        ----------
        strategy_dict: dictionary
            dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        ascending: boolean
            Rank the values in ascending (True) or descending (False) order

        Returns
        -------
        DataFrame
            Returns the mean peformance rank for each estimator
        """
        if not isinstance(ascending, bool):
            raise ValueError('Variable ascending needs to be boolean')

        df = pd.DataFrame(strategy_dict)
        ranked = df.rank(axis=1, ascending=ascending)
        mean_r = pd.DataFrame(ranked.mean(axis=0))
        mean_r.columns = ['avg_rank']
        mean_r = mean_r.sort_values('avg_rank', ascending=ascending)
        return mean_r

    def t_test(self, strategy_dict):
        """
        Runs t-test on all possible combinations between the estimators.

        Parameters
        ----------
        strategy_dict: dictionary
            dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        Returns
        -------
        tuple 
            pandas DataFrame (Database style and MultiIndex)
        """
        t_df = pd.DataFrame()
        perms = itertools.product(strategy_dict.keys(), repeat=2)
        values = np.array([])
        for perm in perms:
            x = np.array(strategy_dict[perm[0]])
            y = np.array(strategy_dict[perm[1]])
            t_stat, p_val = ttest_ind(x, y)

            t_test = {
                'estimator_1': perm[0],
                'estimator_2': perm[1],
                't_stat': t_stat,
                'p_val': p_val
            }

            t_df = t_df.append(t_test, ignore_index=True)
            values = np.append(values, t_stat)
            values = np.append(values, p_val)

        index = t_df['estimator_1'].unique()
        values_names = ['t_stat', 'p_val']
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return t_df, values_df_multiindex

    def sign_test(self, strategy_dict):
        """
        Non-parametric test for test for consistent differences between pairs of observations. See `<https://en.wikipedia.org/wiki/Sign_test>`_ for details about the test and `<https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.binom_test.html>`_ for details about the scipy implementation.

        Parameters
        ----------
        strategy_dict: dictionary
            dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        Returns
        -------
        tuple of dataframes 
            pandas DataFrame (Database style), pivot table)
        """
        sign_df = pd.DataFrame()
        perms = itertools.product(strategy_dict.keys(), repeat=2)
        for perm in perms:
            x = np.array(strategy_dict[perm[0]])
            y = np.array(strategy_dict[perm[1]])
            signs = np.sum([i[0] > i[1] for i in zip(x, y)])
            n = len(x)
            p_val = stats.binom_test(signs, n)
            sign_test = {
                'estimator_1': perm[0],
                'estimator_2': perm[1],
                'p_val': p_val
            }

            sign_df = sign_df.append(sign_test, ignore_index=True)
            sign_df_pivot = sign_df.pivot(index='estimator_1', columns='estimator_2', values='p_val')

        return sign_df, sign_df_pivot

    def ranksum_test(self, strategy_dict):
        """
        Non-parametric test for testing consistent differences between pairs of obeservations.
        The test counts the number of observations that are greater, smaller and equal to the mean
        `<http://en.wikipedia.org/wiki/Wilcoxon_rank-sum_test>`_.

        Parameters
        ----------
        strategy_dict: dictionary
            dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        Returns
        -------
        tuple of pandas DataFrame 
            Database style and MultiIndex
        """
        ranksum_df = pd.DataFrame()
        perms = itertools.product(strategy_dict.keys(), repeat=2)
        values = np.array([])
        for perm in perms:
            comb = perm[0] + ' - ' + perm[1]
            x = strategy_dict[perm[0]]
            y = strategy_dict[perm[1]]
            t_stat, p_val = ranksums(x, y)
            ranksum = {
                'estimator_1': perm[0],
                'estimator_2': perm[1],
                't_stat': t_stat,
                'p_val': p_val
            }
            ranksum_df = ranksum_df.append(ranksum, ignore_index=True)
            values = np.append(values, t_stat)
            values = np.append(values, p_val)

        index = ranksum_df['estimator_1'].unique()
        values_names = ['t_stat', 'p_val']
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return ranksum_df, values_df_multiindex

    def t_test_with_bonferroni_correction(self, strategy_dict, alpha=0.05):
        """
        correction used to counteract multiple comparissons
        https://en.wikipedia.org/wiki/Bonferroni_correction

        
        Parameters
        ----------
        strategy_dict: dictionary
            dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        alpha: float
            confidence level.
        Returns
        -------
        DataFrame 
            MultiIndex DataFrame
        """
        df_t_test, _ = self.t_test(strategy_dict)
        idx_estim_1 = df_t_test['estimator_1'].unique()
        idx_estim_2 = df_t_test['estimator_2'].unique()
        estim_1 = len(idx_estim_1)
        estim_2 = len(idx_estim_2)
        critical_value = alpha / (estim_1 * estim_2)

        bonfer_test = df_t_test['p_val'] <= critical_value

        bonfer_test_reshaped = bonfer_test.values.reshape(estim_1, estim_2)

        bonfer_df = pd.DataFrame(bonfer_test_reshaped, index=idx_estim_1, columns=idx_estim_2)

        return bonfer_df

    def wilcoxon_test(self, strategy_dict):
        """http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
        `Wilcoxon signed-rank test <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`_.
        Tests whether two  related paired samples come from the same distribution. 
        In particular, it tests whether the distribution of the differences x-y is symmetric about zero

        Parameters
        ----------
        strategy_dict: dictionary
            Dictionary with errors on test sets achieved by estimators.
        Returns
        -------
        tuple 
            pandas DataFrame (Database style and MultiIndex)
        """
        wilcoxon_df = pd.DataFrame()
        values = np.array([])
        prod = itertools.product(strategy_dict.keys(), repeat=2)
        for p in prod:
            estim_1 = p[0]
            estim_2 = p[1]
            w, p_val = stats.wilcoxon(strategy_dict[p[0]],
                                      strategy_dict[p[1]])

            w_test = {
                'estimator_1': estim_1,
                'estimator_2': estim_2,
                'statistic': w,
                'p_val': p_val
            }

            wilcoxon_df = wilcoxon_df.append(w_test, ignore_index=True)
            values = np.append(values, w)
            values = np.append(values, p_val)

        index = wilcoxon_df['estimator_1'].unique()
        values_names = ['statistic', 'p_val']
        col_idx = pd.MultiIndex.from_product([index, values_names])
        values_reshaped = values.reshape(len(index), len(values_names) * len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return wilcoxon_df, values_df_multiindex

    def friedman_test(self, strategy_dict):
        """
        The Friedman test is a non-parametric statistical test used to detect differences 
        in treatments across multiple test attempts. The procedure involves ranking each row (or block) together, 
        then considering the values of ranks by columns.
        Implementation used: `scipy.stats <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.friedmanchisquare.html>`_. 
        
        Parameters
        ----------
        strategy_dict : dict
            Dictionary with errors on test sets achieved by estimators.
        Returns
        -------
        tuple 
            dictionary, pandas DataFrame.
        
        """

        """
        use the * operator to unpack a sequence
        https://stackoverflow.com/questions/2921847/what-does-the-star-operator-mean/2921893#2921893
        """
        friedman_test = stats.friedmanchisquare(*[strategy_dict[k] for k in strategy_dict.keys()])
        values = [friedman_test[0], friedman_test[1]]
        values_df = pd.DataFrame([values], columns=['statistic', 'p_value'])

        return friedman_test, values_df

    def nemenyi(self, strategy_dict):
        """
        Post-hoc test run if the `friedman_test` reveals statistical significance.
        For more information see `Nemenyi test <https://en.wikipedia.org/wiki/Nemenyi_test>`_.
        Implementation used `scikit-posthocs <https://github.com/maximtrp/scikit-posthocs>`_.
        
        Parameters
        ----------
        strategy_dict : dict
            Dictionary with errors on test sets achieved by estimators.
        Returns
        -------
        pandas DataFrame
            Results of te Nemenyi test
        """

        strategy_dict = pd.DataFrame(strategy_dict)
        strategy_dict = strategy_dict.melt(var_name='groups', value_name='values')
        nemenyi = sp.posthoc_nemenyi(strategy_dict, val_col='values', group_col='groups')
        return nemenyi
