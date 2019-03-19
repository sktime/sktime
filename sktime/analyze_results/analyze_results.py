from mlaut.shared.static_variables import (DATA_DIR, 
                                           HDF5_DATA_FILENAME, 
                                           EXPERIMENTS_PREDICTIONS_GROUP,
                                           SPLIT_DTS_GROUP,
                                           TRAIN_IDX,
                                           TEST_IDX, 
                                           RUNTIMES_GROUP, 
                                           RESULTS_DIR, 
                                           T_TEST_DATASET, 
                                           SIGN_TEST_DATASET, 
                                           BONFERRONI_CORRECTION_DATASET, 
                                           WILCOXON_DATASET, 
                                           FRIEDMAN_DATASET, 
                                           T_TEST_FILENAME,
                                           FRIEDMAN_TEST_FILENAME, 
                                           WILCOXON_TEST_FILENAME, 
                                           SIGN_TEST_FILENAME, 
                                           BONFERRONI_TEST_FILENAME)
import pandas as pd
import numpy as np
import itertools
from mlaut.shared.files_io import FilesIO
from mlaut.data.data import Data

from scipy import stats
from scipy.stats import ttest_ind
from scipy.stats import ranksums

from sklearn.metrics import accuracy_score, mean_squared_error
import scikit_posthocs as sp

from mlaut.analyze_results.losses import Losses

import matplotlib.pyplot as plt

class AnalyseResults(object):
    """
    Analyze results of machine learning experiments.

    Parameters
    ----------
    results: list
        list of sktime result objects
    """

    def __init__(self, 
                 results_list):

        self._results_list = results_list
    
    def prediction_errors(self, metric, strategies, exact_match=True):
        """
        Calculates the average prediction error per estimator as well as the prediction error achieved by each estimator on individual datasets.

        Parameters
        -----------
        metric: `sktime.analyse_results.scores`
            Error function 
        strategies: list
            List of sktime strategies
        exact_match: Boolean
            If `True` when predictions for all estimators in the estimators array is not available no evaluation is performed on the remaining estimators. 
        Returns
        --------
        pickle of pandas DataFrame
            ``estimator_avg_error`` represents the average error and standard deviation achieved by each estimator. ``estimator_avg_error_per_dataset`` represents the average error and standard deviation achieved by each estimator on each dataset.
        """
        #load all predictions
        dts_predictions_list, dts_predictions_list_full_path = self._data.list_datasets(self._output_h5_predictions_group, self._output_io)
        losses = Losses(metric, estimators, exact_match)
        for res in self._results_list:

            predictions = res.predictions
            y_test = res.true_labels

            losses.evaluate(predictions=predictions, 
                            true_labels=y_test,
                            dataset_name=res.dataset_name)
        return losses.get_losses()


    def average_and_std_error(self, scores_dict):
        """
        Calculates simple average and standard error.

        Args:
            scores_dict(dictionary): Dictionary with estimators (keys) and corresponding prediction accuracies on different datasets.
        
        Returns:
            pandas DataFrame
        """
        result = {}
        for k in scores_dict.keys():
            average = np.average(scores_dict[k])
            n = len(scores_dict[k])
            std_error = np.std(scores_dict[k])/np.sqrt(n)
            result[k]=[average,std_error]
        
        res_df = pd.DataFrame.from_dict(result, orient='index')
        res_df.columns=['avg_score','std_error']
        res_df = res_df.sort_values(['avg_score','std_error'], ascending=[1,1])

        return res_df.round(3)
    
    def plot_boxcharts(self, scores_dict):
        data = []
        labels = []
        avg_error = []
        for e in scores_dict.keys():
            data.append(scores_dict[e])
            avg_error.append(np.mean(scores_dict[e]))
            labels.append(e)
        #sort data and labels based on avg_error
        idx_sort=np.array(avg_error).argsort()
        data=[data[i] for i in idx_sort ]
        labels=[labels[i] for i in idx_sort ]
        #plot the results
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_xticklabels(labels, rotation=90)
        plt.tight_layout()
        plt.show()

        return fig
    def average_training_time(self, estimators):
        """
        Average training time for each estimator.

        Args:
            estimators(`mlaut_estimator` array): Estimator objects.
            exact_match(Boolean): If `True` when predictions for all estimators in the estimators array is not available no evaluation is performed on the remaining estimators. 

        Returns:
            tuple of pandas DataFrame (avg_training_time, trainig_time_per_dataset)
        """
        _, dts_run_times_full_path = self._data.list_datasets(self._run_times_group, self._output_io)
        estimator_dict = {estimator.properties['name']: [] for estimator in estimators}

        for dts in dts_run_times_full_path:
            run_times_per_estimator,_ = self._output_io.load_dataset_pd(dataset_path=dts, return_metadata=False)
            run_times_estimator_names = run_times_per_estimator['strategy_name'].tolist()
            #check whether we have data on all estimators that were passed as an argument

            for strat in estimator_dict.keys():
                index_estimator_run_time = run_times_per_estimator['strategy_name'] == strat
                index_count = np.count_nonzero(index_estimator_run_time)
                if index_count == 0:
                    in_sec = np.nan
                else:
                    strat_run_time = run_times_per_estimator.loc[index_estimator_run_time]
                    in_sec = np.float(strat_run_time['total_seconds'].mean(axis=0))
               
                estimator_dict[strat].append(in_sec)
            # for i in range(run_times_per_estimator.shape[0]):

            #     strategy_name = run_times_per_estimator.iloc[i]['strategy_name']
            #     total_seconds = run_times_per_estimator.iloc[i]['total_seconds']
            #     estimator_dict[strategy_name].append(total_seconds)
        #the long notation is necessary to handle situations when there are unequal number of obeservations per estimator
        # training_time_per_dataset = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in estimator_dict.items() ]))
        # training_time_per_dataset = training_time_per_dataset.round(3)
        # avg_training_time = pd.DataFrame(training_time_per_dataset.mean(axis=0))

        training_time_per_dataset = pd.DataFrame.from_dict(estimator_dict)
        avg_training_time = pd.DataFrame(training_time_per_dataset.mean(axis=0))
        avg_training_time.columns = ['avg training time (in sec)']
        avg_training_time = avg_training_time.sort_values('avg training time (in sec)',ascending=True).round(3)
        return avg_training_time, training_time_per_dataset


    def ranks(self, estimator_dict, ascending):
        """
        Calculates the average ranks based on the performance of each estimator on each dataset

        Args:
            estimator_dict (dictionary): dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
            ascending (boolean): Rank the values in ascending (True) or descending (False) order

        Returns:
            ranks(DataFrame): Returns the mean peformance rank for each estimator
        """
        if not isinstance(ascending, bool):
            raise ValueError('Variable ascending needs to be boolean')
        
        df = pd.DataFrame(estimator_dict)
        ranked = df.rank(axis=1, ascending=ascending)
        mean_r = pd.DataFrame(ranked.mean(axis=0))
        mean_r.columns=['avg_rank']
        mean_r = mean_r.sort_values('avg_rank', ascending=1)
        return mean_r.round(1)


    def cohens_d(self, estimator_dict):
        """
        Cohen's d is an effect size used to indicate the standardised difference between two means. The calculation is implemented natively (without the use of third-party libraries). More information can be found here: `Cohen\'s d <https://en.wikiversity.org/wiki/Cohen%27s_d>`_.

        Args:
            estimator_dict(dictionary): dictionay with keys `names of estimators` and values `errors achieved by estimators on test datasets`.
        Returns:
            pandas DataFrame.
        """
        
        comb = itertools.combinations(estimator_dict.keys(), r=2)
        cohens_d_df = pd.DataFrame(columns=['estimator_1', 'extimator_2','value'])
        for c in comb:
            pair=f'{c[0]}-{c[1]}'
            val1 = estimator_dict[c[0]]
            val2 = estimator_dict[c[1]]
            
            n1 = len(val1)
            n2 = len(val2)

            v1 = np.var(val1)
            v2 = np.var(val2)

            m1 = np.mean(val1)
            m2 = np.mean(val2)

            SDpooled = np.sqrt(((n1-1)*v1 + (n2-1)*v2)/(n1+n2-2))
            ef = (m2-m1)/SDpooled
            cohens_d = {
                'estimator_1': c[0],
                'estimator_2': c[1],
                'value': ef
            }
            cohens_d_df = cohens_d_df.append(cohens_d, ignore_index=True)
            cohens_d_df = cohens_d_df.round(3)


        table = pd.pivot_table(cohens_d_df, 
                               index='estimator_1', 
                               columns='estimator_2', 
                               values='value', 
                               fill_value='')
        return table

   

    def t_test(self, observations):
        """
        Runs t-test on all possible combinations between the estimators.

        Args:
            observations(dictionary): Dictionary with errors on test sets achieved by estimators.
        Returns:
            tuple of pandas DataFrame (Database style and MultiIndex)
        """
        t_df = pd.DataFrame()
        perms = itertools.product(observations.keys(), repeat=2)
        values = np.array([])
        for perm in perms:
            x = np.array(observations[perm[0]])
            y = np.array(observations[perm[1]])
            t_stat, p_val = ttest_ind(x,y)

            t_test = {
                'estimator_1': perm[0],
                'estimator_2': perm[1],
                't_stat': t_stat,
                'p_val': p_val
            }

            t_df = t_df.append(t_test, ignore_index=True)
            values = np.append(values,t_stat)
            values = np.append(values,p_val)
            
        index=t_df['estimator_1'].unique()
        values_names = ['t_stat','p_val']
        col_idx = pd.MultiIndex.from_product([index,values_names])
        values_reshaped = values.reshape(len(index), len(values_names)*len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return t_df.round(3), values_df_multiindex.round(3)

    def sign_test(self, observations):
        """
        Non-parametric test for test for consistent differences between pairs of observations. See `<https://en.wikipedia.org/wiki/Sign_test>`_ for details about the test and `<https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.binom_test.html>`_ for details about the scipy implementation.

        Args:
            observations(dictionary): Dictionary with errors on test sets achieved by estimators.
        Returns:
            tuple of dataframes (pandas DataFrame (Database style), pivot table)
        """
        sign_df = pd.DataFrame()
        perms = itertools.product(observations.keys(), repeat=2)
        for perm in perms:
            x = np.array(observations[perm[0]])
            y = np.array(observations[perm[1]])
            signs = np.sum([i[0] > i[1] for i in zip(x,y)])
            n = len(x)
            p_val = stats.binom_test(signs,n)
            sign_test = {
                'estimator_1': perm[0],
                'estimator_2': perm[1],
                'p_val': p_val
            }

            sign_df = sign_df.append(sign_test, ignore_index=True)
            sign_df_pivot = sign_df.pivot(index='estimator_1', columns='estimator_2', values='p_val')


        return sign_df.round(3), sign_df_pivot.round(3)

    def ranksum_test(self, observations):
        """
        Non-parametric test for testing consistent differences between pairs of obeservations.
        The test counts the number of observations that are greater, smaller and equal to the mean
        `<http://en.wikipedia.org/wiki/Wilcoxon_rank-sum_test>`_.

        Args:
            observations(dictionary): Dictionary with errors on test sets achieved by estimators.
        Returns:
            tuple of pandas DataFrame (Database style and MultiIndex)
        """
        ranksum_df = pd.DataFrame()
        perms = itertools.product(observations.keys(), repeat=2)
        values = np.array([])
        for perm in perms:
            comb  = perm[0] + ' - ' + perm[1]
            x = observations[perm[0]]
            y = observations[perm[1]]
            t_stat, p_val = ranksums(x,y)
            ranksum = {
                'estimator_1': perm[0],
                'estimator_2': perm[1],
                't_stat': t_stat,
                'p_val': p_val
            }
            ranksum_df = ranksum_df.append(ranksum, ignore_index=True)
            values = np.append(values,t_stat)
            values = np.append(values,p_val)

        index=ranksum_df['estimator_1'].unique()
        values_names = ['t_stat','p_val']
        col_idx = pd.MultiIndex.from_product([index,values_names])
        values_reshaped = values.reshape(len(index), len(values_names)*len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return ranksum_df.round(3), values_df_multiindex.round(3)
        
    def t_test_with_bonferroni_correction(self, observations, alpha=0.05):
        """
        correction used to counteract multiple comparissons
        https://en.wikipedia.org/wiki/Bonferroni_correction

        Args:
            observations(dictionary): Dictionary with errors on test sets achieved by estimators.
            alpha(float): confidence level.
        Returns:
            tuple of pandas DataFrame
        """
        df_t_test, _ = self.t_test(observations)
        idx_estim_1 = df_t_test['estimator_1'].unique()
        idx_estim_2 = df_t_test['estimator_2'].unique()
        estim_1 = len(idx_estim_1)
        estim_2 = len(idx_estim_2)
        critical_value = alpha/(estim_1*estim_2)

        bonfer_test = df_t_test['p_val'] <= critical_value
        
        bonfer_test_reshaped = bonfer_test.values.reshape(estim_1, estim_2)

        bonfer_df = pd.DataFrame(bonfer_test_reshaped, index=idx_estim_1, columns=idx_estim_2)

        return bonfer_df
        
    def wilcoxon_test(self, observations):
        """http://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
        `Wilcoxon signed-rank test <https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test>`_.
        Tests whether two  related paired samples come from the same distribution. 
        In particular, it tests whether the distribution of the differences x-y is symmetric about zero

        Args:
            observations(dictionary): Dictionary with errors on test sets achieved by estimators.
        Returns:
            tuple of pandas DataFrame (Database style and MultiIndex)
        """
        wilcoxon_df = pd.DataFrame()
        values = np.array([])
        prod = itertools.product(observations.keys(), repeat=2)
        for p in prod:
            estim_1 = p[0]
            estim_2 = p[1]
            w, p_val = stats.wilcoxon(observations[p[0]],
                         observations[p[1]])

            w_test = {
                'estimator_1': estim_1,
                'estimator_2': estim_2,
                'statistic': w,
                'p_val': p_val
            }

            wilcoxon_df = wilcoxon_df.append(w_test, ignore_index=True)
            values = np.append(values, w)
            values = np.append(values, p_val)

        index=wilcoxon_df['estimator_1'].unique()
        values_names = ['statistic','p_val']
        col_idx = pd.MultiIndex.from_product([index,values_names])
        values_reshaped = values.reshape(len(index), len(values_names)*len(index))

        values_df_multiindex = pd.DataFrame(values_reshaped, index=index, columns=col_idx)

        return wilcoxon_df.round(3), values_df_multiindex.round(3)
                        
    def friedman_test(self, observations):
        """
        The Friedman test is a non-parametric statistical test used to detect differences 
        in treatments across multiple test attempts. The procedure involves ranking each row (or block) together, 
        then considering the values of ranks by columns.
        Implementation used: `scipy.stats <https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.friedmanchisquare.html>`_. 
        
        Args:
            observations(dictionary): Dictionary with errors on test sets achieved by estimators.
        Returns:
            tuple of dictionary, pandas DataFrame.
        
        """

        """
        use the * operator to unpack a sequence
        https://stackoverflow.com/questions/2921847/what-does-the-star-operator-mean/2921893#2921893
        """
        friedman_test = stats.friedmanchisquare(*[observations[k] for k in observations.keys()])
        values = [friedman_test[0], friedman_test[1]]
        values_df = pd.DataFrame([values], columns=['statistic','p_value'])

        return friedman_test, values_df.round(3)
    
    def nemenyi(self, obeservations):
        """
        Post-hoc test run if the `friedman_test` reveals statistical significance.
        For more information see `Nemenyi test <https://en.wikipedia.org/wiki/Nemenyi_test>`_.
        Implementation used `scikit-posthocs <https://github.com/maximtrp/scikit-posthocs>`_.
        
        Args:
            observations(dictionary): Dictionary with errors on test sets achieved by estimators.
        Returns:
            pandas DataFrame.
        """

        obeservations = pd.DataFrame(obeservations)
        obeservations = obeservations.melt(var_name='groups', value_name='values')
        nemenyi =sp.posthoc_nemenyi(obeservations, val_col='values', group_col='groups')
        return nemenyi.round(3)