import Orange
import pandas as pd
import matplotlib.pyplot as plt


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
    num_datasets = len((df.iloc[:, 1]).unique())
    avg_ranks_dict = compute_average_rank(df)

    classifiers = list(avg_ranks_dict.keys())
    avg_rank_values = list(avg_ranks_dict.values())

    cd = Orange.evaluation.compute_CD(avg_rank_values, num_datasets)
    Orange.evaluation.graph_ranks(avg_rank_values, classifiers, cd=cd, width=6,
                                  textspace=1.5)
    plt.show()


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
    dict
        Dict containing the classifier as the key and the average rank as the value
    """
    datasets = (df.iloc[:, 1]).unique()
    num_datasets = len(datasets)
    estimators = (df.iloc[:, 0]).unique()

    ranking_values = {}
    for estimator in estimators:
        ranking_values[estimator] = 0

    for dataset in datasets:
        dataset_df = df.loc[df[df.columns[1]] == dataset]
        dataset_df_sorted = dataset_df.sort_values(by=[df.columns[-1]], ascending=False)
        i = 1
        for row in dataset_df_sorted.iloc[:, 0]:
            ranking_values[row] += i
            i += 1

    for estimator in ranking_values:
        curr_val = ranking_values[estimator]
        ranking_values[estimator] = curr_val / num_datasets

    return ranking_values


if __name__ == '__main__':
    df = pd.read_csv('./test.csv', index_col=False)
    create_critical_difference_diagram(df, output_path='./', title='Accuracy')
