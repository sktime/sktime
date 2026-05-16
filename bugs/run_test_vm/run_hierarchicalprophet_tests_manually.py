from sktime.tests.test_class_register import get_test_classes_for_obj
from sktime.forecasting.prophetverse import HierarchicalProphet

estimator = HierarchicalProphet
test_clss_for_est = get_test_classes_for_obj(estimator)


results = {}
for test_cls in test_clss_for_est:
    test_cls_results = test_cls().run_tests(
        estimator=estimator,
        raise_exceptions=True,
        tests_to_run=None,
        fixtures_to_run=None,
        tests_to_exclude=None,
        fixtures_to_exclude=None,
        verbose=True,
    )
    results.update(test_cls_results)

# (Pdb++) p unreserved_param_names
# {'feature_transformer', 'correlation_matrix_concentration', 'likelihood', 'inference_engine', 'default_effect', 'noise_scale', 'shared_features', 'exogenous_effects', 'trend', 'rng_key'}
# (Pdb++) p reserved_set
# set()
# (Pdb++) p param_names
# ['correlation_matrix_concentration', 'default_effect', 'exogenous_effects', 'feature_transformer', 'inference_engine', 'likelihood', 'noise_scale', 'rng_key', 'shared_features', 'trend']
# 
