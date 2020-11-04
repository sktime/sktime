from sktime.benchmarking.results import RAMResults
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import PairwiseMetric
from sklearn.metrics import accuracy_score
from sktime.series_as_features.model_selection import PresplitFilesCV
import numpy as np
import pandas as pd

def dummy_results():
    results = RAMResults()
    results.cv = PresplitFilesCV()
    results.save_predictions(strategy_name='alg1',
                            dataset_name='dataset1',
                            index=np.array([1,2,3,4]),
                            y_true=np.array([1,1,1,1]),
                            y_pred=np.array([1,1,1,1]),
                            y_proba=None,
                            cv_fold=0,
                            train_or_test="test")
    results.save_predictions(strategy_name='alg1',
                            dataset_name='dataset2',
                            index=np.array([1,2,3,4]),
                            y_true=np.array([0,0,0,0]),
                            y_pred=np.array([0,0,0,0]),
                            y_proba=None,
                            cv_fold=0,
                            train_or_test="test")

    results.save_predictions(strategy_name='alg2',
                            dataset_name='dataset1',
                            index=np.array([1,2,3,4]),
                            y_true=np.array([1,1,1,1]),
                            y_pred=np.array([0,0,0,0]),
                            y_proba=None,
                            cv_fold=0,
                            train_or_test="test")
    results.save_predictions(strategy_name='alg2',
                            dataset_name='dataset2',
                            index=np.array([1,2,3,4]),
                            y_true=np.array([0,0,0,0]),
                            y_pred=np.array([1,1,1,1]),
                            y_proba=None,
                            cv_fold=0,
                            train_or_test="test")

    results.save_predictions(strategy_name='alg3',
                            dataset_name='dataset1',
                            index=np.array([1,2,3,4]),
                            y_true=np.array([1,1,1,1]),
                            y_pred=np.array([1,1,0,1]),
                            y_proba=None,
                            cv_fold=0,
                            train_or_test="test")
    results.save_predictions(strategy_name='alg3',
                            dataset_name='dataset2',
                            index=np.array([1,2,3,4]),
                            y_true=np.array([0,0,0,0]),
                            y_pred=np.array([0,0,1,0]),
                            y_proba=None,
                            cv_fold=0,
                            train_or_test="test")
    return results

def test_rank():
    evaluator = Evaluator(dummy_results())
    metric = PairwiseMetric(func=accuracy_score, name="accuracy")
    metrics_by_strategy = evaluator.evaluate(metric=metric)

    expected_ranks = pd.DataFrame({'strategy':['alg1','alg2','alg3'],'accuracy_mean_rank':[1.0,3.0,2.0]})
    generated_ranks = evaluator.rank()
    assert expected_ranks.equals(generated_ranks)

def test_accuracy_score():
    evaluator = Evaluator(dummy_results())
    metric = PairwiseMetric(func=accuracy_score, name="accuracy")
    metrics_by_strategy = evaluator.evaluate(metric=metric)

    expected_accuracy = pd.DataFrame({'strategy':['alg1','alg2','alg3'],
                                    'accuracy_mean':[1.00,0.00,0.75],
                                    'accuracy_stderr':[0.00,0.00,0.25]
                                    })
    
    assert metrics_by_strategy.equals(expected_accuracy)
print(test_accuracy_score())