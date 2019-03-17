import logging
import os
import json
import re


class Orchestrator:
    """
    Orchestrates the sequencing of running the machine learning experiments.
    """
    def __init__(self,
                tasks,
                data,
                dataset_names,
                data_dir='data',
                experiments_trained_estimators_dir='trained_estimators',
                experiments_results_dir='results'):
        """
        Parameters
        ----------
        tasks: array of sktime tasks
            array of sktime tasks
        data: array of pandas DataFrame
            datasets in pandas format
        dataset_names: array of strings
            names of the datasets
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
        
        self._data_holder = []

        for x in zip(tasks, data, dataset_names):
            self._data_holder.append( _DataHolder(tasks=x[0],data=x[1], dataset_name=x[2]) )

    def set_strategies(self, strategies):
        """
        Sets the learing strategies

        Parameters
        ----------
        strategies: array of sktime estimators
            Array of sktime estimators used in the benchmarking exercise
        """
        self._strategies=strategies
    

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
        for dh in self._data_holder:
            dts_trained=0
            logging.warning(f'Training estimators on {dh.dataset_name}')
            data = dh.data
            y = data[dh.task.target]
            X = data.drop([dh.task.target],axis=1)
            train_idx, test_idx = self._resampling.resample(X,y)
            if save_resampling_splits is True:
                dh_idx = self._data_holder.index(dh)
                dh.set_resampling_splits(train_idx=train_idx, test_idx=test_idx)
                self._data_holder[dh_idx]=dh

            for strat in self._strategies:

                #check whether the model was already trained
                path_to_check = os.path.join(self._data_dir, 
                                             self._experiments_trained_estimators_dir, 
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
                        #save predictions as json object
                        results = {}
                        results['true_labels']= y.iloc[test_idx].tolist()
                        results['predictions']=predictions.tolist()
                        save_path = os.path.join(self._data_dir,
                                                 self._experiments_results_dir,
                                                 dh.dataset_name,
                                                 strat.name) + '.json'
                        
                        self._save_results(results, save_path,overwrite_predictions)
                        
        

class _DataHolder:
    """
    Class for holdig the data, schema, resampling splits and metadata
    """
    def __init__(self, data, tasks, dataset_name):
        """
        Parameters
        ----------
        data: pandas DataFrame
                dataset in pandas format
        tasks: sktime tasks
            sktime tasks
        dataset_name: string
            Name of the dataset
        """

        self._data=data
        self._tasks=tasks
        self._dataset_name=dataset_name
    
    @property
    def data(self):
        return self._data
    
    @property
    def task(self):
        return self._tasks
    
    @property
    def dataset_name(self):
        return self._dataset_name

    def set_resampling_splits(self, train_idx, test_idx):
        """
        Saves the train test indices after the data is resampled

        Parameters
        -----------
        train_idx: numpy array
            array with indices of the train set
        test_idx: numpy array
            array with indices of the test set
        """
        self._train_idx = train_idx
        self._test_idx = test_idx