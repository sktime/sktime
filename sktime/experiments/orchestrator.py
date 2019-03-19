import logging
import os
import json
import re
from sktime.highlevel import Result

class Orchestrator:
    """
    Orchestrates the sequencing of running the machine learning experiments.
    """
    def __init__(self,
                data_holders,
                strategies,
                resampling,
                data_dir='data',
                experiments_trained_estimators_dir='trained_estimators',
                experiments_results_dir='results'):
        """
        Parameters
        ----------
        data_holders: list 
            list of sktime DataHolder objects
        strategies: list
            list of sktime strategies
        resampling: sktime.resampling object
            resampling strategy for the data.
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
        self._strategies = strategies
        self._data_holders = data_holders
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
        
        Returns
        -------
        list 
            sktime.highlevel.results objects 
        """
        results_list = []
        for dh in self._data_holders:
            dts_trained=0
            logging.warning(f'Training estimators on {dh.dataset_name}')
            data = dh.data
            y = data[dh.task.target]
            X = data.drop([dh.task.target],axis=1)
            train_idx, test_idx = self._resampling.resample(X,y)
            if save_resampling_splits is True:
                dh_idx = self._data_holders.index(dh)
                dh.set_resampling_splits(train_idx=train_idx, test_idx=test_idx)
                self._data_holders[dh_idx]=dh

            for strat in self._strategies:

                #check whether the model was already trained
                path_to_check = os.path.join(self._data_dir, 
                                             self._experiments_results_dir,
                                             dh.dataset_name, 
                                             strat.name)

                strategy_estists = os.path.isfile(path_to_check)

                if strategy_estists is True and overwrite_saved_estimators is False:
                    if verbose is True:
                        logging.warning(f'Estimator {strat.name} already trained on {dh.dataset_name}. Skipping it.')
                
                else:
                    strat.fit(dh.task, data.iloc[train_idx])

                    #TODO: save trained strategy to disk

                    if predict_on_runtime is True:
                        logging.warning(f'Making predictions for strategy: {strat.name} on dataset: {dh.dataset_name}')
                        predictions = strat.predict(data.iloc[test_idx])
                        #create a results object
                        result = Result(dataset_name=dh.dataset_name,
                                        strategy_name=strat.name,
                                        true_labels=y.iloc[test_idx].tolist(),
                                        predictions=predictions.tolist()) 
                        
                        results_list.append(result)
                        
                        # TODO: save results to disk/hdd
                        # save_path = os.path.join(self._data_dir,
                        #                          self._experiments_results_dir,
                        #                          dh.dataset_name,
                        #                          strat.name) + '.json'
                        
                        # self._save_results(results, save_path,overwrite_predictions)
        return results_list
        

