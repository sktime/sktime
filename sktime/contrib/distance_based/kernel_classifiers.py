import scipy.stats
import scipy.stats.stats
import sklearn.model_selection
import sklearn.neighbors
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.svm

import sktime.classifiers.base
from sktime.classifiers.distance_based.proximity_forest import CachedTransformer
from sktime.contrib.transformers.kernels import (
    DtwKernel, PandasToNumpy, ParametersFromDatasetWrapper,
    find_params_using,
    dtw_parameter_space_getter,
    InvertKernel,
    RbfKernel,
    full_dtw_parameter_space_getter,
    WdtwKernel,
    wdtw_parameter_space_getter,
    LcssKernel,
    lcss_parameter_space_getter,
    ErpKernel,
    erp_parameter_space_getter,
    msm_parameter_space_getter,
    MsmKernel,
    TwedKernel,
    twe_parameter_space_getter,
    ed_parameter_space_getter,
    TriKernel,
    PolyKernel,
    Kl2Kernel,
    HellKernel,
    NegToZero,
    EigKernel,
    NegToAbs,
    NegToMin,
)
from sklearn.svm import SVC
from sktime.transformers.summarise import DerivativeSlopeTransformer
from sklearn.neighbors import KNeighborsClassifier


def build_ed_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', InvertKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_ed_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter, iid=False,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_tri_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TriKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_poly_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', PolyKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_kl2_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', Kl2Kernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_hell_rbf_svm(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', HellKernel()),
        ('i', RbfKernel()),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_ed_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', InvertKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_ed_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter, iid=False,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_tri_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TriKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_poly_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', PolyKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_kl2_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', Kl2Kernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_hell_rbf_1nn(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', HellKernel()),
        ('i', RbfKernel()),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_ed_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_ed_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter, iid=False,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_tri_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TriKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_poly_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', PolyKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_kl2_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', Kl2Kernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_hell_rbf_svm_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', HellKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_ed_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_ed_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter, iid=False,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_tri_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TriKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_poly_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', PolyKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_kl2_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', Kl2Kernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_hell_rbf_1nn_eig_min(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', HellKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToMin())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_ed_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_ed_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter, iid=False,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_tri_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TriKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_poly_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', PolyKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_kl2_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', Kl2Kernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_hell_rbf_svm_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', HellKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_ed_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_ed_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier


def build_dtw_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_dtw_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter, iid=False,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_ddtw_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier


def build_full_ddtw_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier


def build_wdtw_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_wddtw_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier


def build_lcss_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier


def build_erp_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier


def build_msm_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier


def build_twed_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier


def build_tri_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TriKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_poly_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', PolyKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_kl2_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', Kl2Kernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model


def build_hell_rbf_1nn_eig_abs(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', HellKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToAbs())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model

def build_ed_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier

def build_dtw_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier

def build_full_dtw_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier

def build_ddtw_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier

def build_full_ddtw_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier

def build_wdtw_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier

def build_wddtw_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier

def build_lcss_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier

def build_erp_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier

def build_msm_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier

def build_twed_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier

def build_ed_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier

def build_dtw_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier

def build_full_dtw_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter, iid=False,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier

def build_ddtw_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier

def build_full_ddtw_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier

def build_wdtw_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier

def build_wddtw_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier

def build_lcss_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier

def build_erp_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier

def build_msm_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier

def build_twed_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier

def build_tri_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TriKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model

def build_poly_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', PolyKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model

def build_kl2_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', Kl2Kernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model

def build_hell_rbf_svm_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', HellKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('svm', SVC(probability=True, kernel='precomputed')),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low = 1, high = 10 + 1),
        'svm__C': scipy.stats.expon(scale=100)
    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model

def build_ed_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier

def build_dtw_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier

def build_full_dtw_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier

def build_ddtw_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier

def build_full_ddtw_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier

def build_wdtw_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier

def build_wddtw_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier

def build_lcss_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier

def build_erp_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier

def build_msm_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier

def build_twed_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', InvertKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier

def build_ed_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(ed_parameter_space_getter))
    return classifier

def build_dtw_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier

def build_full_dtw_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter, iid=False,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier

def build_ddtw_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(dtw_parameter_space_getter))
    return classifier

def build_full_ddtw_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', DtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(full_dtw_parameter_space_getter))
    return classifier

def build_wdtw_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier

def build_wddtw_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('der', CachedTransformer(DerivativeSlopeTransformer())),
        ('d', WdtwKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(wdtw_parameter_space_getter))
    return classifier

def build_lcss_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', LcssKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(lcss_parameter_space_getter))
    return classifier

def build_erp_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', ErpKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(erp_parameter_space_getter))
    return classifier

def build_msm_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', MsmKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(msm_parameter_space_getter))
    return classifier

def build_twed_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TwedKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    classifier = ParametersFromDatasetWrapper(model, find_params_using(twe_parameter_space_getter))
    return classifier

def build_tri_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', TriKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model

def build_poly_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', PolyKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model

def build_kl2_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', Kl2Kernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model

def build_hell_rbf_1nn_eig_zero(
        cv=10,
        n_jobs=1,
        n_iter=100,
        verbose=0,
        random_state=0):
    pipe = sklearn.pipeline.Pipeline([
        ('pd_to_np', PandasToNumpy()),
        ('d', HellKernel()),
        ('i', RbfKernel()),
        ('e', EigKernel(transformer=NegToZero())),
        ('1nn', KNeighborsClassifier(n_neighbors=1)),
    ])
    cv_params = {
        'd__degree': scipy.stats.randint(low=1, high=10 + 1),

    }
    model = sklearn.model_selection.RandomizedSearchCV(pipe,
                                                       cv_params,
                                                       cv=cv,
                                                       n_jobs=n_jobs,
                                                       n_iter=n_iter,
                                                       verbose=verbose,
                                                       random_state=random_state
                                                       )
    return model
