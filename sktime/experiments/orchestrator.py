import logging
import os
import json
import re
import numpy as np
import pandas as pd


class Orchestrator:
    """
    Orchestrates the sequencing of running the machine learning experiments.
    """
    def __init__(self, tasks, datasets, strategies, cv, result):
        """
        Parameters
        ----------
        tasks: sktime.highlevel.Task
            task object
        datasets: pandas dataframe
            datasets in pandas skitme format
        strategies: list of sktime strategy
            strategy as per sktime.highlevel
        cv: sklearn.model_selection cross validation
            sklearn cross validation method. Must implement split()
        result: sktime result class
            Object for saving the results
        """
        self._tasks = tasks
        self._datasets = datasets
        self._strategies = strategies
        self._cv = cv
        self._result = result


    def run(self, predict_on_runtime=True, save_strategies=True):
        """
        Method for running the orchestrator
        
        Parameters
        ----------
        predict_on_runtime:Boolean
            If True makes predictions after the estimator is trained
        save_strategies: Boolean
            If True saves the trained strategies on the disk
        """
        
        for task, data in zip(self._tasks, self._datasets):
            dts_loaded = data.load()
            for strategy in self._strategies:
                for i, (train, test) in enumerate(self._cv.split(dts_loaded)):
                    
                    strategy.fit(task, train)

                    if predict_on_runtime:
                        y_pred = strategy.predict(test)
                        self._result.save(dataset_name=data.dataset_name, 
                                          strategy_name=strategy.name, 
                                          y_true=test[task.target], 
                                          y_pred=y_pred)
                    if save_strategies:
                        self._result.save_trained_strategy(strategy=strategy, dataset_name=data.dataset_name)
    
    def predict(self):
        """
        TODO load saved strategies and make predictions
        """
        raise NotImplementedError()

