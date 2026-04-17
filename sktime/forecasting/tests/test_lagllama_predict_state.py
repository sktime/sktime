"""Regression tests for LagLlamaForecaster._predict non-mutation contract."""

__author__ = ["shaun0927"]

import ast
from pathlib import Path


def test_lagllama_predict_does_not_write_self_is_range_index():
    """``_predict`` must not assign to ``self._is_range_index``.

    ``_is_range_index`` is pre-computed in ``_fit`` precisely because
    ``_predict`` is required to leave ``__dict__`` deeply unchanged (sktime
    non-state-changing contract, see the author's comment in ``_fit``).

    Re-assigning ``self._is_range_index`` inside ``_predict`` reintroduced
    that contract violation. We parse ``lagllama.py`` directly so the test
    stays cheap and runs without the foundation-model stack.
    """
    source_path = (
        Path(__file__).resolve().parents[1] / "lagllama.py"
    )
    tree = ast.parse(source_path.read_text())

    forecaster_cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "LagLlamaForecaster"
    )
    predict_method = next(
        node
        for node in forecaster_cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "_predict"
    )

    offenders = []
    for node in ast.walk(predict_method):
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AugAssign):
            targets = [node.target]
        else:
            continue
        for target in targets:
            if (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "self"
                and target.attr == "_is_range_index"
            ):
                offenders.append(node.lineno)

    assert not offenders, (
        "LagLlamaForecaster._predict must not assign to self._is_range_index "
        f"(violating lines: {offenders})."
    )
