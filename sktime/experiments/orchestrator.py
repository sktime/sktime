import logging
import os
import json
import re


class Orchestrator:
    """
    Orchestrates the sequencing of running the machine learning experiments.
    """
    def __init__(self,
                data_dir='data',
                experiments_trained_estimators_dir='trained_estimators',
                experiments_results_dir='results'):
        """
        Parameters
        ----------
        data_dir: string
            data directory for saving output of orchestrator
        experiments_trained_estimators_dir: string
            path on the disk for saving the trained estimators
        experiments_results_dir: string
            path for saving json objects with results 
        """
        self._data_dir = data_dir
        self._experiments_trained_estimators_dir = experiments_trained_estimators_dir
        self._experiments_results_dir = experiments_results_dir
  

    def set_strategies(self, strategies):
        """
        Sets the learing strategies

        Parameters
        ----------
        strategies: array of sktime estimators
            Array of sktime estimators used in the benchmarking exercise
        """
        self._strategies=strategies
    
    def set_tasks(self, tasks):
        """
        Sets the tasks used in the learning strategies

        Parameters
        -----------
        tasks: array of sktime tasks
            Array of sktime tasks
        """
        self._tasks = tasks

    def set_resampling(self, resampling):
        """
        Parameters
        ----------
        resampling: sktime.resampling object
            resampling strategy for the data.
        """
        self._resampling = resampling

    def _save_results(self, results, save_path, overwrite_predictions):
        """
        Saves the results of the experiments to disk
        
        Parameters
        ----------
        results: json
            json with the results of the experiment
        save_path: string
            directory where the results will be saved
        overwrite_predictions: Boolean
            If True overwrites previous saved results
        """
        save_file_exists = os.path.isfile(save_path)

        #strip the filename fron the save path
        re_srch = re.search('.*\/',save_path)
        dir_only = save_path[re_srch.span()[0]:re_srch.span()[1]]
        if save_file_exists is False or (save_file_exists is True and overwrite_predictions is True):
            if not os.path.exists(dir_only):
                os.makedirs(dir_only)
            with open(save_path, 'w') as outfile:
                json.dump(results, outfile)
            logging.warning(f'Saved predictions to: {save_path}')
        else:
            logging.warning(f'Path {save_path} exists. Set overwrite_predictions=True if you wish to overwrite it.')
    def run(self, 
            overwrite_saved_estimators=False,
            verbose=True,
            predict_on_runtime=True,
            overwrite_predictions=False,
            save_resampling_splits=True):
        """
        Main method for running the experiments. Iterates though all strategies and through all tasks.

        Parameters
        ----------
        overwrite_saved_estimators:Boolean
            If True overwrites the esimators that were saved on the disk
        verbose:Boolean
            If True outputs messages during training
        predict_on_runtime:Boolean
            If True makes predictions after the estimator is trained
        overwrite_preictions:Boolean
            If True overwrites the predictions made at previous runs
        save_resampling_splits: Boolean
            If `True` saves resampling splits in database
        """
        for tsk in self._tasks:
            dts_trained=0
            logging.warning(f'Training estimators on {tsk.dataset_name}')
            data = tsk.data
            y = data[tsk.target]
            X = data.drop([tsk.target],axis=1)
            train_idx, test_idx = self._resampling.resample(X,y)
            if save_resampling_splits is True:
                tsk_idx = self._tasks.index(tsk)
                tsk.set_resampling_splits(train_idx=train_idx, test_idx=test_idx)
                self._tasks[tsk_idx]=tsk

            for strat in self._strategies:

                #check whether the model was already trained
                path_to_check = f'{self._experiments_trained_estimators_dir}{os.sep}{tsk.dataset_name}{os.sep}{strat.name}'
                strategy_estists = os.path.isfile(path_to_check)

                if strategy_estists is True and overwrite_saved_estimators is False:
                    if verbose is True:
                        logging.warning(f'Estimator {strat.name} already trained on {tsk.dataset_name}. Skipping it.')
                
                else:
                    strat.fit(tsk, data.iloc[train_idx])

                    #TODO: save trained strategy to disk

                    if predict_on_runtime is True:
                        logging.warning(f'Making predictions for strategy: {strat.name} on dataset: {tsk.dataset_name}')
                        predictions = strat.predict(data.iloc[test_idx])
                        #save predictions as json object
                        results = {}
                        results['true_labels']= y.iloc[test_idx].tolist()
                        results['predictions']=predictions.tolist()
                        save_path = f'{self._data_dir}{os.sep}{self._experiments_results_dir}{os.sep}{tsk.dataset_name}{os.sep}{strat.name}.json'
                        self._save_results(results, save_path,overwrite_predictions)
                        
        
