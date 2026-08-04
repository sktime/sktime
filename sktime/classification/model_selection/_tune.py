"""Tuning for time series classifiers."""

__author__ = ["fkiraly", "achieveordie"]

import numpy as np
from sklearn.metrics import accuracy_score

from sktime.classification._delegate import _DelegatedClassifier
from sktime.classification.model_evaluation import evaluate
from sktime.utils.model_selection import (
    _make_cv_results,
    _make_evaluate_cv,
    _refit_estimator,
    _resolve_scorers,
    _run_grid_search,
    _select_best_index,
)


class TSCGridSearchCV(_DelegatedClassifier):
    """Exhaustive search over specified parameter values for an estimator.

    Performs native grid search for sktime time series classifiers.

    Optimizes hyper-parameters of ``estimators`` by exhaustive grid search.

    Parameters
    ----------
    estimator : sktime classifier, BaseClassifier instance or interface compatible
        The classifier to tune, must implement the sktime classifier interface.

    param_grid : dict or list of dictionaries
        Dictionary with parameters names (``str``) as keys and lists of
        parameter settings to try as values, or a list of such
        dictionaries, in which case the grids spanned by each dictionary
        in the list are explored. This enables searching over any sequence
        of parameter settings.

    scoring : str, callable, list, tuple or dict, default=None
        Strategy to evaluate the performance of the cross-validated model on
        the test set.

        A string is resolved as a scikit-learn scorer. A callable, list/tuple of
        callables or strings, or dictionary of named callables/strings can be used
        for single- or multi-metric evaluation. If ``None``, defaults to
        ``accuracy_score``.

    n_jobs : int, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    refit : bool, str, or callable, default=True
        Refit an estimator using the best found parameters on the whole
        dataset. If ``False``, the ``predict`` and ``predict_proba`` will not work.

        For multiple metric evaluation, this needs to be a ``str`` denoting the
        scorer that would be used to find the best parameters for refitting
        the estimator at the end.

        Where there are considerations other than maximum score in
        choosing a best estimator, ``refit`` can be set to a function which
        returns the selected ``best_index_`` given ``cv_results_``. In that
        case, the ``best_estimator_`` and ``best_params_`` will be set
        according to the returned ``best_index_`` while the ``best_score_``
        attribute will not be available.

        The refitted estimator is made available at the ``best_estimator_``
        attribute and permits using ``predict`` directly on this
        search instance.

        Also for multiple metric evaluation, the attributes ``best_index_``,
        ``best_score_`` and ``best_params_`` will only be available if
        ``refit`` is set and all of them will be determined w.r.t this specific
        scorer.

        See ``scoring`` parameter to know more about multiple metric
        evaluation.

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross validation,
        - integer, to specify the number of folds in a ``(Stratified)KFold``,
        - :term:`CV splitter`,
        - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if the estimator is a classifier and ``y`` is
        either binary or multiclass, :class:`StratifiedKFold` is used. In all
        other cases, :class:`KFold` is used. These splitters are instantiated
        with ``shuffle=False`` so the splits will be the same across calls.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    verbose : int
        Controls the verbosity: the higher, the more messages.

        - >1 : the computation time for each fold and parameter candidate is
          displayed;
        - >2 : the score is also displayed;
        - >3 : the fold and candidate parameter indexes are also displayed
          together with the starting time of the computation.

    pre_dispatch : int, or str, default='2*n_jobs'
        Controls the number of jobs that get dispatched during parallel
        execution. Reducing this number can be useful to avoid an
        explosion of memory consumption when more jobs get dispatched
        than CPUs can process. This parameter can be:

            - None, in which case all the jobs are immediately
              created and spawned. Use this for lightweight and
              fast-running jobs, to avoid delays due to on-demand
              spawning of the jobs

            - An int, giving the exact number of total jobs that are
              spawned

            - A str, giving an expression as a function of n_jobs,
              as in '2*n_jobs'

    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
        If set to 'raise', the error is raised. If a numeric value is given,
        FitFailedWarning is raised. This parameter does not affect the refit
        step, which will always raise the error.

    return_train_score : bool, default=False
        If ``False``, the ``cv_results_`` attribute will not include training
        scores.
        Computing training scores is used to get insights on how different
        parameter settings impact the overfitting/underfitting trade-off.
        However computing the scores on the training set can be computationally
        expensive and is not strictly required to select the parameters that
        yield the best generalization performance.

    tune_by_variable : bool, optional (default=False)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the estimator will be fit to each variable separately,
        and are available in fields of the classifiers_ attribute.
        Has the same effect as applying ColumnEnsembleClassifier wrapper to self.
        If False, the same best parameter is selected for all variables.

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        A dict with keys as column headers and values as columns, that can be
        imported into a pandas ``DataFrame``.

        For multi-metric evaluation, the scores for all the scorers are
        available in the ``cv_results_`` dict at the keys ending with that
        scorer's name (``'_<scorer_name>'``) instead of ``'_score'`` shown
        above. ('split0_test_precision', 'mean_train_precision' etc.)

    best_estimator_ : estimator
        Estimator that was chosen by the search, i.e. estimator
        which gave highest score (or smallest loss if specified)
        on the left out data. Not available if ``refit=False``.

        See ``refit`` parameter for more information on allowed values.

    best_score_ : float
        Mean cross-validated score of the best_estimator

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

        This attribute is not available if ``refit`` is a function.

    best_params_ : dict
        Parameter setting that gave the best results on the hold out data.

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    best_index_ : int
        The index (of the ``cv_results_`` arrays) which corresponds to the best
        candidate parameter setting.

        The dict at ``search.cv_results_['params'][search.best_index_]`` gives
        the parameter setting for the best model, that gives the highest
        mean score (``search.best_score_``).

        For multi-metric evaluation, this is present only if ``refit`` is
        specified.

    scorer_ : function or a dict
        Scorer function used on the held out data to choose the best
        parameters for the model.

        For multi-metric evaluation, this attribute holds the validated
        ``scoring`` dict which maps the scorer key to the scorer callable.

    n_splits_ : int
        The number of cross-validation splits (folds/iterations).

    refit_time_ : float
        Seconds used for refitting the best model on the whole dataset.

        This is present only if ``refit`` is not False.

    multimetric_ : bool
        Whether or not the scorers compute several metrics.

    classes_ : ndarray of shape (n_classes,)
        The classes labels. This is present only if ``refit`` is specified and
        the underlying estimator is a classifier.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if
        ``best_estimator_`` is defined (see the documentation for the ``refit``
        parameter for more details) and that ``best_estimator_`` exposes
        ``n_features_in_`` when fit.

    feature_names_in_ : ndarray of shape (``n_features_in_``,)
        Names of features seen during :term:`fit`. Only defined if
        ``best_estimator_`` is defined (see the documentation for the ``refit``
        parameter for more details) and that ``best_estimator_`` exposes
        ``feature_names_in_`` when fit.

    See Also
    --------
    ParameterGrid : Generates all the combinations of a hyperparameter grid.
    train_test_split : Utility function to split the data into a development
        set usable for fitting a grid-search instance and an evaluation set
        for its final evaluation.
    sklearn.metrics.make_scorer : Make a scorer from a performance metric or
        loss function.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly", "achieveordie"],
        # estimator type
        # --------------
        "X_inner_mtype": ["nested_univ", "numpy3D"],
        "y_inner_mtype": ["numpy2D"],
        "capability:multivariate": True,
        "capability:multioutput": True,
        "capability:unequal_length": True,
        "capability:missing_values": True,
        "capability:multithreading": True,
        "capability:predict_proba": True,
        "capability:categorical_in_X": True,
        # CI and test flags
        # -----------------
        "tests:core": True,  # should tests be triggered by framework changes?
    }

    def __init__(
        self,
        estimator,
        param_grid,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
        tune_by_variable=False,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.refit = refit
        self.cv = cv
        self.verbose = verbose
        self.pre_dispatch = pre_dispatch
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.tune_by_variable = tune_by_variable

        super().__init__()
        self.estimator_ = estimator.clone()

        if self.tune_by_variable:
            self.set_tags(**{"capability:multioutput": False})

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
            3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "pd-multiindex:":
            pd.DataFrame with columns = variables,
            index = pd.MultiIndex with first level = instance indices,
            second level = time indices
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            1D iterable, of shape [n_instances]
            or 2D iterable, of shape [n_instances, n_dimensions]
            class labels for fitting
            if self.get_tag("capaility:multioutput") = False, guaranteed to be 1D
            if self.get_tag("capaility:multioutput") = True, guaranteed to be 2D

        Returns
        -------
        self : Reference to self.
        """
        if y.shape[1] == 1:
            y = y.flatten()

        scorers = _resolve_scorers(self.scoring, accuracy_score)
        cv = _make_evaluate_cv(self.cv, y)
        raw_results = _run_grid_search(
            estimator=self.estimator,
            param_grid=self.param_grid,
            cv=cv,
            scorers=scorers,
            evaluate=evaluate,
            X=X,
            y=y,
            error_score=self.error_score,
            n_jobs=self.n_jobs,
            pre_dispatch=self.pre_dispatch,
            verbose=self.verbose,
            return_train_score=self.return_train_score,
        )

        self.cv_results_ = _make_cv_results(
            raw_results, scorers, self.return_train_score
        )
        self.n_splits_ = cv.get_n_splits()
        self.scorer_ = next(iter(scorers.values())) if len(scorers) == 1 else scorers
        self.multimetric_ = len(scorers) > 1

        scoring_names = list(scorers)
        self.best_index_, score_key = _select_best_index(
            self.cv_results_, scoring_names, self.refit
        )
        self.best_params_ = self.cv_results_["params"][self.best_index_]
        if score_key is not None:
            self.best_score_ = self.cv_results_[score_key][self.best_index_]

        if self.refit:
            self.best_estimator_, self.refit_time_ = _refit_estimator(
                self.estimator, self.best_params_, X, y
            )
            self.estimator_ = self.best_estimator_
            if hasattr(self.best_estimator_, "classes_"):
                self.classes_ = self.best_estimator_.classes_
        else:
            self.estimator_ = self.estimator.clone().set_params(**self.best_params_)

        return self

    def _predict(self, X):
        """Predict labels for sequences in X.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 1D np.array of int, of shape [n_instances] - predicted class labels
            indices correspond to instance indices in X
        """
        if not self.refit:
            raise RuntimeError(
                f"In {self.__class__.__name__}, refit must be True "
                "to make predictions, "
                f"but found refit=False."
            )

        estimator = self._get_delegate()
        y_pred = estimator.predict(X=X)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
        return y_pred

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        if not self.refit:
            return {}
        return self._get_delegate().get_fitted_params()

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        from sklearn.gaussian_process.kernels import RBF, DotProduct
        from sklearn.metrics import accuracy_score

        from sktime.classification.kernel_based import TimeSeriesSVC
        from sktime.dists_kernels import AggrDist

        mean_eucl_tskernel = AggrDist(DotProduct())
        mean_rbf_tskernel = AggrDist(RBF())

        param1 = {
            "estimator": TimeSeriesSVC(kernel=mean_rbf_tskernel, probability=True),
            "param_grid": {"C": [0.1, 1]},
        }

        param2 = {
            "estimator": TimeSeriesSVC(kernel=mean_eucl_tskernel, probability=True),
            "param_grid": {"kernel__transformer": [DotProduct(), RBF()]},
            "scoring": accuracy_score,
        }

        return [param1, param2]
