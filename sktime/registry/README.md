
This folder contains registries of key `sktime` concepts and objects:

* `_base_classes` : estimator scitypes and their base classes
* `_tags` : estimator tags used within `sktime`

New objects should be added to this registry for automated recognition by the test suite.

The `_lookup` sub-module contains runtime inspection functionality for the registry:
* `all_estimators` for returning lists of estimators by specified selection criteria
