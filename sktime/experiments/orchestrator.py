import logging
import os
class Orchestrator:
    """
    Orchestrates the sequencing of running the machine learning experiments.
    """
    def __init__(self,
                 experiments_trained_estimators_dir='data/trained_estimators'):
        """
        Parameters
        ----------
        experiments_trained_estimators_dir: string
            path on the disk for saving the trained estimators
        """
        self._experiments_trained_estimators_dir=experiments_trained_estimators_dir

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
    
    def run(self, 
            overwrite_saved_estimators=False,
            verbose=True,
            predict_on_runtime=True,
            overwrite_predictoins=False,
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
            logging.log(1, f'Training estimators on  {tsk.dataset_name}')
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
                        logging.info(f'Estimator {strat.name} already trained on {tsk.dataset_name}. Skipping it.')
                
                else:
                    strat.fit(tsk, data.iloc[train_idx])

                    #TODO: save trained strategy on disk

                    if predict_on_runtime is True:
                        predictions = strat.predict(data.iloc[test_idx])
                        #TODO: save predictions
        
