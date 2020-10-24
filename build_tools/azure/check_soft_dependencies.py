#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

# sktime supports a number of soft dependencies which are necessary for using
# a certain module but otherwise not necessary. Here we check if soft
# dependencies have been properly isolated and are not required to run other
# modules.
import pkgutil
import re
from importlib import import_module

SOFT_DEPENDENCIES = {
    "sktime.benchmarking.evaluation": ["matplotlib"],
    "sktime.transformers.series_as_features.tsfresh": ["tsfresh"],
    "sktime.forecasting.arima": ["pmdarima"],
    "sktime.forecasting.all": ["pmdarima"],
    "sktime.classification.all": ["tsfresh"],
    "sktime.regression.all": ["tsfresh"],
}
MODULES_TO_IGNORE = ("tests", "contrib")


def _extract_dependency_from_error_msg(msg):
    match = re.search(r"\'(.+?)\'", msg)
    if match:
        return match.group(1)
    else:
        raise ValueError("No dependency found in error msg.")


for _, modname, _ in pkgutil.walk_packages(path=["./sktime/"], prefix="sktime."):
    mod_parts = modname.split(".")
    if any(part in MODULES_TO_IGNORE for part in mod_parts):
        continue

    try:
        import_module(modname)
    except ModuleNotFoundError as e:
        error_msg = str(e)

        soft_dependencies = SOFT_DEPENDENCIES.get(modname, [])
        if any(soft_dependency in error_msg for soft_dependency in soft_dependencies):
            continue

        dependency = _extract_dependency_from_error_msg(error_msg)
        raise ModuleNotFoundError(
            f"{modname} should not require any soft dependencies, "
            f"but tried importing: {dependency}"
        ) from e
