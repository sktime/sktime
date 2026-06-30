# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Test that type-specific modules do not cross-import each other at module level.

Rule: modules in TYPE_MODULES must not import from a sibling TYPE_MODULE at the
top level of a file. Imports inside functions, classes, or TYPE_CHECKING blocks are
allowed, since they are deferred and do not cause cascading import chains.

Existing violations are listed in KNOWN_EXCEPTIONS and should be cleaned up
in follow-up PRs. New violations will fail this test.
"""

__author__ = ["Nischal1425"]

import ast
from pathlib import Path

import pytest

# Modules that should not cross-import each other at module level.
TYPE_MODULES = frozenset(
    {
        "alignment",
        "classification",
        "clustering",
        "detection",
        "forecasting",
        "networks",
        "param_est",
        "regression",
        "transformations",
    }
)

# Existing violations: (path relative to sktime root, importing module, imported module)
# These are pre-existing and should be fixed in follow-up PRs, not in this one.
KNOWN_EXCEPTIONS = frozenset(
    {
        ("classification/compose/_pipeline.py", "classification", "transformations"),
        (
            "classification/deep_learning/cnn/_cnn_tf.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/cnn/_cnn_torch.py",
            "classification",
            "networks",
        ),
        ("classification/deep_learning/cntc.py", "classification", "networks"),
        (
            "classification/deep_learning/convtran/_convtran_torch.py",
            "classification",
            "networks",
        ),
        ("classification/deep_learning/fcn.py", "classification", "networks"),
        (
            "classification/deep_learning/inceptiontime/_inceptiontime_tf.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/inceptiontime/_inceptiontime_torch.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/lstmfcn/_lstmfcn_tf.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/lstmfcn/_lstmfcn_torch.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/macnn/_macnn_tf.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/macnn/_macnn_torch.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/mcdcnn/_mcdcnn_tf.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/mcdcnn/_mcdcnn_torch.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/mlp/_mlp_tf.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/mlp/_mlp_torch.py",
            "classification",
            "networks",
        ),
        ("classification/deep_learning/resnet.py", "classification", "networks"),
        (
            "classification/deep_learning/rnn/_rnn_tf.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/rnn/_rnn_torch.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/tapnet/_tapnet_tf.py",
            "classification",
            "networks",
        ),
        (
            "classification/deep_learning/tapnet/_tapnet_torch.py",
            "classification",
            "networks",
        ),
        (
            "classification/dictionary_based/_boss.py",
            "classification",
            "transformations",
        ),
        (
            "classification/dictionary_based/_muse.py",
            "classification",
            "transformations",
        ),
        (
            "classification/dictionary_based/_tde.py",
            "classification",
            "transformations",
        ),
        (
            "classification/dictionary_based/_weasel.py",
            "classification",
            "transformations",
        ),
        (
            "classification/distance_based/_elastic_ensemble.py",
            "classification",
            "transformations",
        ),
        (
            "classification/distance_based/_proximity_forest.py",
            "classification",
            "transformations",
        ),
        (
            "classification/distance_based/_shape_dtw.py",
            "classification",
            "transformations",
        ),
        ("classification/ensemble/_ctsf.py", "classification", "transformations"),
        (
            "classification/feature_based/_catch22_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_fresh_prince.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_matrix_profile_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_random_interval_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_signature_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_summary_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/feature_based/_tsfresh_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/interval_based/_cif.py",
            "classification",
            "transformations",
        ),
        (
            "classification/interval_based/_drcif.py",
            "classification",
            "transformations",
        ),
        (
            "classification/kernel_based/_arsenal.py",
            "classification",
            "transformations",
        ),
        (
            "classification/kernel_based/_rocket_classifier.py",
            "classification",
            "transformations",
        ),
        (
            "classification/plotting/temporal_importance_diagram.py",
            "classification",
            "transformations",
        ),
        (
            "classification/shapelet_based/_stc.py",
            "classification",
            "transformations",
        ),
        ("clustering/compose/_as_transform.py", "clustering", "transformations"),
        ("clustering/compose/_pipeline.py", "clustering", "transformations"),
        ("detection/clasp.py", "detection", "transformations"),
        ("detection/compose/_as_transform.py", "detection", "transformations"),
        ("detection/eagglo.py", "detection", "transformations"),
        ("detection/stray.py", "detection", "transformations"),
        ("detection/wclust.py", "detection", "clustering"),
        ("forecasting/boxcox_biasadj.py", "forecasting", "transformations"),
        ("forecasting/cinn.py", "forecasting", "networks"),
        ("forecasting/cinn.py", "forecasting", "transformations"),
        ("forecasting/compose/_bagging.py", "forecasting", "transformations"),
        ("forecasting/compose/_grouped.py", "forecasting", "transformations"),
        (
            "forecasting/compose/_hierarchy_ensemble.py",
            "forecasting",
            "transformations",
        ),
        ("forecasting/compose/_reduce.py", "forecasting", "transformations"),
        ("forecasting/enbpi.py", "forecasting", "transformations"),
        ("forecasting/es_rnn.py", "forecasting", "networks"),
        ("forecasting/rbf.py", "forecasting", "networks"),
        ("forecasting/reconcile.py", "forecasting", "transformations"),
        ("forecasting/theta.py", "forecasting", "transformations"),
        ("param_est/compose/_pipeline.py", "param_est", "transformations"),
        ("param_est/plugin/_forecaster.py", "param_est", "forecasting"),
        ("param_est/plugin/_transformer.py", "param_est", "transformations"),
        ("regression/compose/_ensemble.py", "regression", "transformations"),
        ("regression/compose/_pipeline.py", "regression", "transformations"),
        ("regression/deep_learning/cnn/_cnn_tf.py", "regression", "networks"),
        ("regression/deep_learning/cnn/_cnn_torch.py", "regression", "networks"),
        ("regression/deep_learning/cntc.py", "regression", "networks"),
        (
            "regression/deep_learning/convtran/_convtran_torch.py",
            "regression",
            "networks",
        ),
        ("regression/deep_learning/fcn.py", "regression", "networks"),
        (
            "regression/deep_learning/inceptiontime/_inceptiontime_tf.py",
            "regression",
            "networks",
        ),
        (
            "regression/deep_learning/inceptiontime/_inceptiontime_torch.py",
            "regression",
            "networks",
        ),
        (
            "regression/deep_learning/lstmfcn/_lstmfcn_tf.py",
            "regression",
            "networks",
        ),
        (
            "regression/deep_learning/lstmfcn/_lstmfcn_torch.py",
            "regression",
            "networks",
        ),
        ("regression/deep_learning/macnn/_macnn_tf.py", "regression", "networks"),
        (
            "regression/deep_learning/macnn/_macnn_torch.py",
            "regression",
            "networks",
        ),
        (
            "regression/deep_learning/mcdcnn/_mcdcnn_tf.py",
            "regression",
            "networks",
        ),
        (
            "regression/deep_learning/mcdcnn/_mcdcnn_torch.py",
            "regression",
            "networks",
        ),
        ("regression/deep_learning/mlp/_mlp_tf.py", "regression", "networks"),
        ("regression/deep_learning/mlp/_mlp_torch.py", "regression", "networks"),
        ("regression/deep_learning/resnet.py", "regression", "networks"),
        ("regression/deep_learning/rnn/_rnn_tf.py", "regression", "networks"),
        ("regression/deep_learning/rnn/_rnn_torch.py", "regression", "networks"),
        (
            "regression/deep_learning/tapnet/_tapnet_tf.py",
            "regression",
            "networks",
        ),
        (
            "regression/deep_learning/tapnet/_tapnet_torch.py",
            "regression",
            "networks",
        ),
        (
            "regression/kernel_based/_rocket_regressor.py",
            "regression",
            "transformations",
        ),
        ("transformations/basisfunction.py", "transformations", "networks"),
        ("transformations/detrend/_detrend.py", "transformations", "forecasting"),
        ("transformations/impute.py", "transformations", "forecasting"),
        ("transformations/theta.py", "transformations", "forecasting"),
    }
)

_SKTIME_ROOT = Path(__file__).parent.parent


def _is_test_path(path):
    return any(p in ("tests", "test") for p in path.parts)


def _is_type_checking_block(node):
    if not isinstance(node, ast.If):
        return False
    test = node.test
    if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
        return True
    if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
        return True
    return False


def _get_import_target(node):
    """Return the top-level sktime submodule imported, or None."""
    if isinstance(node, ast.ImportFrom):
        if node.module and node.module.startswith("sktime."):
            parts = node.module.split(".")
            if len(parts) >= 2:
                return parts[1]
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name.startswith("sktime."):
                parts = alias.name.split(".")
                if len(parts) >= 2:
                    return parts[1]
    return None


def _collect_violations():
    """Return set of (rel_path, source_module, imported_module) for cross-imports."""
    violations = set()
    for py_file in _SKTIME_ROOT.rglob("*.py"):
        if _is_test_path(py_file):
            continue
        rel = py_file.relative_to(_SKTIME_ROOT)
        parts = rel.parts
        if not parts or parts[0] not in TYPE_MODULES:
            continue
        source = parts[0]
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for node in ast.iter_child_nodes(tree):
            if _is_type_checking_block(node):
                continue
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                target = _get_import_target(node)
                if target and target != source and target in TYPE_MODULES:
                    violations.add((str(rel).replace("\\", "/"), source, target))
    return violations


def test_no_new_cross_module_imports():
    """No new module-level cross-imports between type-specific sktime modules.

    Type-specific modules (forecasting, transformations, classification, etc.)
    must not import each other at module level to avoid cascading import chains
    that can lead to circular imports.

    Existing violations are grandfathered in KNOWN_EXCEPTIONS. Only new ones fail.
    To fix an exception: move the import inside the function/class that needs it,
    then remove its entry from KNOWN_EXCEPTIONS.
    """
    violations = _collect_violations()
    new_violations = violations - KNOWN_EXCEPTIONS
    stale_exceptions = KNOWN_EXCEPTIONS - violations

    messages = []
    if new_violations:
        lines = "\n".join(
            f"  {path}  ({src} -> {tgt})"
            for path, src, tgt in sorted(new_violations)
        )
        messages.append(
            f"New cross-module imports found (move import inside the function "
            f"or class that uses it):\n{lines}"
        )
    if stale_exceptions:
        lines = "\n".join(
            f"  {path}  ({src} -> {tgt})"
            for path, src, tgt in sorted(stale_exceptions)
        )
        messages.append(
            f"KNOWN_EXCEPTIONS entries no longer violated (remove them):\n{lines}"
        )

    if messages:
        pytest.fail("\n\n".join(messages))
