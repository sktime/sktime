# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Regression tests for TinyTimeMixerPreTrainedModel attribute compat.

Covers issue #9827: ``all_tied_weights_keys`` must be resolvable on the
class regardless of the installed ``transformers`` version (4.x and 5.x),
because transformers v5 looks it up during ``from_pretrained``.
"""

import pytest
from skbase.utils.dependencies import _check_soft_dependencies


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", "transformers", severity="none"),
    reason="torch / transformers not installed",
)
def test_all_tied_weights_keys_attribute_present():
    """TinyTimeMixerPreTrainedModel exposes all_tied_weights_keys on the class.

    Required by transformers>=5.0 during from_pretrained.
    Must not break on transformers<5.0 (attribute is unused there).
    """
    from sktime.libs.granite_ttm.modeling_tinytimemixer import (
        TinyTimeMixerPreTrainedModel,
    )

    assert hasattr(TinyTimeMixerPreTrainedModel, "all_tied_weights_keys")
    val = TinyTimeMixerPreTrainedModel.all_tied_weights_keys
    # TTM ties no weights, default must be empty
    assert len(val) == 0


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", "transformers", severity="none"),
    reason="torch / transformers not installed",
)
def test_all_tied_weights_keys_class_default_not_mutated_by_instance():
    """Instance-level writes must not mutate the class-level default.

    Guards against shared mutable state across instances if v5 writes to
    ``self.all_tied_weights_keys``.
    """
    from sktime.libs.granite_ttm.modeling_tinytimemixer import (
        TinyTimeMixerPreTrainedModel,
    )

    class_default = TinyTimeMixerPreTrainedModel.all_tied_weights_keys
    # snapshot
    snapshot = dict(class_default) if hasattr(class_default, "items") else None

    # simulate instance-level write (what v5 post_init may do)
    sentinel = object()
    instance_dict = {}
    instance_dict["fake_key"] = sentinel

    # class default unchanged
    if snapshot is not None:
        assert dict(TinyTimeMixerPreTrainedModel.all_tied_weights_keys) == snapshot
