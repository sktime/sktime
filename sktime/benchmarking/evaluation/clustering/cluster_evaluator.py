from typing import List
import pandas as pd

from sktime.benchmarking.evaluation.base import BaseEstimatorEvaluator
from sktime.datasets import read_clusterer_result_from_uea_format

def some_callable():
    pass

evaluation_metrics = {
    'RI': some_callable,  # Rand index
    'ARI': some_callable,  # Adjusted rand index
    'MI': some_callable,  # Mutual information
    'NMI': some_callable,  # Normalized mutual information
    'AMI': some_callable,  # Adjusted mutual information
    'FM': some_callable,  # Fowlkes-Mallows
    'SC': some_callable,  # Silhouette Coefficient
    'CHI': some_callable,  # Calinski-Harabasz Index
    'CM': some_callable,  # Contingency matrix
    'ACC': some_callable,  # Accuracy
    'PCM': some_callable,  # Pair confusion matrix
}

class ClusterEvaluator(BaseEstimatorEvaluator):

    def __init__(
            self,
            metrics: List[str],
            results_path: str,
            evaluation_out_path: str,
            experiment_name: str,
    ):
        for metric in metrics:
            if metric not in evaluation_metrics:
                raise ValueError(f'The metric: {metric} is invalid please check the '
                                 f'list of available metrics to use.')
        self.metrics = metrics
        super(ClusterEvaluator, self).__init__(
            results_path, evaluation_out_path, experiment_name
        )


    def evaluate_csv_data(self, csv_path: str):
        data = read_clusterer_result_from_uea_format(csv_path)
        first_line = data['first_line_comment']
        parameters = data['estimator_parameters']
        meta = pd.DataFrame(data['estimator_meta'])
        predictions = pd.DataFrame(data['predictions'])
        test = ''

if __name__ == '__main__':
    # from sktime.clustering.k_means import TimeSeriesKMeans
    # from sktime.benchmarking.experiments import run_clustering_experiment
    # from sktime.datasets import load_acsf1
    #
    # X_train, y_train = load_acsf1(split='train')
    # X_test, y_test = load_acsf1(split='test')
    # k_means_clusterer = TimeSeriesKMeans(
    #     n_clusters=5,  # Set to 10 as num classes is 10
    #     metric='euclidean',
    # )
    #
    # run_clustering_experiment(
    #     X_train,
    #     k_means_clusterer,
    #     results_path="./example-notebook-results",
    #     trainY=y_train,
    #     testX=X_test,
    #     testY=y_test,
    #     cls_name="kmeans",
    #     dataset_name="acsf1",
    #     resample_id=0,
    #     overwrite=False,
    # )

    joe = ClusterEvaluator(
        'C:\\Users\\chris\\Documents\\Projects\\sktime\\sktime\\benchmarking\\evaluation\\tests\\test_results',
        'C:\\Users\\chris\\Documents\\Projects\\sktime\\sktime\\benchmarking\\evaluation\\tests\\result_out',
        'example_experiment'
    )

    joe.run_evaluation(['kmeans', 'kmedoids'])
