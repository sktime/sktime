import math
import Orange
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import networkx

from scipy.stats import friedmanchisquare, wilcoxon


def create_critical_difference_diagram(df, output_path, title):
    """Creates a critical difference diagram.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame should have three columns index 0 should be the estimators
        names, index 1 should be the dataset and index 3 should be the accuracy the
        estimator scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | accuracy |
        | cls1      | data1   | 1.2      |
        | cls2      | data2   | 3.4      |
        | cls1      | data2   | 1.4      |
        | cls2      | data1   | 1.3      |
        ----------------------------------
    """
    avg_ranks = compute_average_rank(df)
    p_values = compute_wilcoxon_signed_rank(df)

    cliques = form_cliques(p_values, avg_ranks.keys())

    joe = ''

    # cd = Orange.evaluation.compute_CD(avg_rank_values, num_datasets)
    # Orange.evaluation.graph_ranks(avg_rank_values, classifiers, cd=cd, width=6,
    #                               textspace=1.5)
    # plt.show()o


def form_cliques(p_values, estimators):
    """For clique for critical difference.

    This method is used to find grouping that are not critically different so that they
    can be connected in the final graph.

    Parameters
    ----------
    p_values: List[Tuple]
        List of tuples of length 4. Where index 0 is the name of the first estimator,
        index 1 is the name of the estimator it was compared to, index 2 is the p value
        and index 3 is a boolean that when true means two classifiers are not critically
        different and false means they are critically different.
    estimators: Pd.Index
        Index of keys that are the estimators.

    Returns
    -------
    List[List[int]]
        List where each list contains the index of the estimators that are not
        critically different.
    """
    m = len(estimators)
    graph = np.zeros((m, m), dtype=np.int64)
    for p in p_values:
        if not p[3]:
            i = np.where(estimators == p[0])[0]
            j = np.where(estimators == p[1])[0]
            min_i = min(i, j)
            max_j = max(i, j)
            graph[min_i, max_j] = 1

    return list(networkx.find_cliques(networkx.Graph(graph)))


def compute_wilcoxon_signed_rank(df: pd.DataFrame, alpha=0.05):
    """Compute the wilcoxon signed rank for a dataframe of result.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame should have three columns index 0 should be the estimators
        names, index 1 should be the dataset and index 3 should be the accuracy the
        estimator scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | accuracy |
        | cls1      | data1   | 1.2      |
        | cls2      | data2   | 3.4      |
        | cls1      | data2   | 1.4      |
        | cls2      | data1   | 1.3      |
        ----------------------------------
    alpha: float
        Alpha value to use to reject Holm hypotheiss.

    Returns
    -------
    List[Tuple]
        List of tuples of length 4. Where index 0 is the name of the first estimator,
        index 1 is the name of the estimator it was compared to, index 2 is the p value
        and index 3 is a boolean that when true means two classifiers are not critically
        different and false means they are critically different.
    """
    datasets = (df.iloc[:, 1]).unique()
    num_datasets = len(datasets)
    estimators = (df.iloc[:, 0]).unique()

    acc_arr = []
    for estimator in estimators:
        acc_arr.append((df.loc[df[df.columns[0]] == estimator][df.columns[-1]])
                       .to_numpy())

    friedman_p_value = friedmanchisquare(*acc_arr)[1]

    if friedman_p_value >= alpha:
        raise ValueError("The estimators results provided cannot reject the null"
                         "hypothesis.")
    p_values = []

    for i in range(len(estimators)):
        curr_estimator = estimators[i]
        curr_estimator_accuracy = \
            (df.loc[df[df.columns[0]] == curr_estimator][df.columns[-1]]).to_numpy()
        for j in range(i + 1, len(estimators)):
            curr_compare_estimator = estimators[j]
            curr_compare_estimator_accuracy = \
                (df.loc[df[df.columns[0]] == curr_compare_estimator][df.columns[-1]]) \
                    .to_numpy()

            p_value = wilcoxon(
                curr_estimator_accuracy,
                curr_compare_estimator_accuracy,
                zero_method='pratt'
            )[1]

            p_values.append((curr_estimator, curr_compare_estimator, p_value, False))

    p_values.sort(key=operator.itemgetter(2))

    for i in range(len(p_values)):
        new_alpha = float(alpha / (len(p_values) - i))
        if not p_values[i][2] <= new_alpha:
            break
        p_values[i] = (p_values[i][0], p_values[i][1], p_values[i][2], True)

    return p_values


def compute_average_rank(df: pd.DataFrame):
    """Compute the average ranking for each estimator.

    Parameters
    ----------
    df: pd.DataFrame
        The data frame should have three columns index 0 should be the estimators
        names, index 1 should be the dataset and index 3 should be the accuracy the
        estimator scored for the datasets. For examples:
        ----------------------------------
        | estimator | dataset | accuracy |
        | cls1      | data1   | 1.2      |
        | cls2      | data2   | 3.4      |
        | cls1      | data2   | 1.4      |
        | cls2      | data1   | 1.3      |
        ----------------------------------
    Returns
    -------
    Pd.Series
        Series where each element is a series where the classifier is index 0 and the
        rank is index 1.
    """
    datasets = (df.iloc[:, 1]).unique()
    num_datasets = len(datasets)
    estimators = (df.iloc[:, 0]).unique()

    sorted_df = df.loc[df[df.columns[0]].isin(estimators)]. \
        sort_values([df.columns[0], df.columns[1]])
    rank_data = np.array(sorted_df[df.columns[-1]]).reshape(
        len(estimators), num_datasets
    )

    df_ranks = pd.DataFrame(
        data=rank_data,
        index=np.sort(estimators),
        columns=np.unique(sorted_df[df.columns[1]])
    )

    ranking_values = \
        df_ranks.rank(ascending=False).mean(axis=1).sort_values(ascending=False)

    return ranking_values


if __name__ == '__main__':
    df = pd.read_csv('./test.csv', index_col=False)
    create_critical_difference_diagram(df, output_path='./', title='Accuracy')
