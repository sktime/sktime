# -*- coding: utf-8 -*-
from typing import List, Dict, Tuple
import ast

import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    rand_score,
    mutual_info_score,
    adjusted_mutual_info_score,
    normalized_mutual_info_score,
    fowlkes_mallows_score,
    calinski_harabasz_score,
    accuracy_score,
    confusion_matrix
)


from sktime.benchmarking.evaluation.base import BaseEstimatorEvaluator
from sktime.datasets import read_clusterer_result_from_uea_format


def some_callable():
    pass


_evaluation_metrics_dict = {
    "RI": rand_score,  # Rand index
    "ARI": adjusted_rand_score,  # Adjusted rand index
    "MI": mutual_info_score,  # Mutual information
    "NMI": normalized_mutual_info_score,  # Normalized mutual information
    "AMI": adjusted_mutual_info_score,  # Adjusted mutual information
    # "FM": fowlkes_mallows_score,  # Fowlkes-Mallows
    # "SC": some_callable,  # Silhouette Coefficient
    # "CHI": calinski_harabasz_score,  # Calinski-Harabasz Index
    "ACC": accuracy_score,  # Accuracy
    # "PCM": confusion_matrix,  # Pair confusion matrix
}


class ClusterEvaluator(BaseEstimatorEvaluator):
    def __init__(
        self,
        results_path: str,
        evaluation_out_path: str,
        experiment_name: str,
        metrics: List[str] = None,
        naming_parameter_key: str = None
    ):
        if metrics is None:
            metrics = ['RI', 'ARI', 'MI', 'NMI', 'AMI', 'ACC']
        for metric in metrics:
            if metric not in _evaluation_metrics_dict:
                raise ValueError(
                    f"The metric: {metric} is invalid please check the "
                    f"list of available metrics to use."
                )
        self.metrics = metrics
        self.naming_parameter_key = naming_parameter_key
        super(ClusterEvaluator, self).__init__(
            results_path, evaluation_out_path, experiment_name
        )

    def evaluate_csv_data(self, csv_path: str) -> Tuple:
        data = read_clusterer_result_from_uea_format(csv_path)
        first_line = data["first_line_comment"]
        parameters = ast.literal_eval(','.join((data["estimator_parameters"])))
        meta = pd.DataFrame(data["estimator_meta"])
        temp = pd.DataFrame(data["predictions"])
        predictions_df = pd.DataFrame(temp[[0, 1]])
        probabilities_df = pd.DataFrame(temp[list(range(3, len(temp.columns)))])
        metrics_score = self._compute_metrics(predictions_df)
        estimator_name = self.get_estimator_name(first_line, parameters)
        return (first_line[0], first_line[1], estimator_name, metrics_score)

    def get_estimator_name(
            self,
            estimator_details: List[str],
            estimator_params: List[str]
    ) -> str:
        """Generate estimator name from parameters.

        Parameters
        ----------
        estimator_details: List[str]
            List of strings containing details about the estimator.
        estimator_params: List[str]
            List of strings containing details of the parameters the estimator used.

        Returns
        -------
        str
            Name of estimator to use.
        """
        estimator_name = estimator_details[1]
        if self.naming_parameter_key is not None:
            test = estimator_params
            estimator_name = \
                f"{estimator_name}-{estimator_params[self.naming_parameter_key]}"
        return estimator_name



    def _compute_metrics(self, predictions_df: pd.DataFrame) -> Dict:
        numpy_pred = predictions_df.to_numpy()
        true_class = [i[0] for i in numpy_pred[1:]]
        predicted_class = [i[1] for i in numpy_pred[1:]]
        metric_scores = {}
        for metric in self.metrics:
            metric_callable = _evaluation_metrics_dict[metric]
            metric_scores[metric] = metric_callable(true_class, predicted_class)

        return metric_scores




if __name__ == "__main__":
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
        "C:\\Users\\chris\\Documents\\Projects\\sktime\\sktime\\benchmarking\\evaluation\\tests\\test_results",
        "C:\\Users\\chris\\Documents\\Projects\\sktime\\sktime\\benchmarking\\evaluation\\tests\\result_out",
        experiment_name="example_experiment",
        naming_parameter_key='metric'
    )

    joe.run_evaluation(["kmeans", "kmedoids"])
