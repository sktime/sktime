"""Debug script for issue #7622."""

from skbase.base import BaseObject

from sktime.registry import all_estimators

list_of_all_estimators: list[tuple[str, BaseObject]] = all_estimators(
    as_dataframe=False
)

sorted_estimators = sorted(list_of_all_estimators, key=lambda x: x[0])

print("Total Estimators", len(sorted_estimators))
for name, _ in sorted_estimators:
    print(name)

estimators_with_less_than_two_test_parameters = []
for name, estimator in sorted_estimators:
    estimator_test_parameters = estimator.get_test_params()

    if (
        isinstance(estimator_test_parameters, list)
        and len(estimator_test_parameters) < 2
    ):
        estimators_with_less_than_two_test_parameters.append(name)
    elif isinstance(estimator_test_parameters, dict):
        estimators_with_less_than_two_test_parameters.append(name)
    elif isinstance(estimator_test_parameters, list):
        continue
    else:
        print(
            f"Unexpected type of test parameters for {name}",
            type(estimator_test_parameters),
        )

print(
    "Total estimators with less than two test parameters",
    len(estimators_with_less_than_two_test_parameters),
)
for name in estimators_with_less_than_two_test_parameters:
    print(name)
