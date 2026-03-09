import pytest

from sktime.libs._torch_self_attention.tests.seq_self_attention.util import (
    TestMaskShapeTorch,
)
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

pytestmark = pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._torch_self_attention")
    or not _check_soft_dependencies("torch", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)


class TestMask(TestMaskShapeTorch):
    def test_return_attention(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        attention = SeqSelfAttentionTorch(return_attention=True, attention_width=3)
        self.check_mask_shape(attention)

    def test_padded_pairs_zeroed(self):
        from sktime.libs._torch_self_attention import SeqSelfAttentionTorch

        attention = SeqSelfAttentionTorch(return_attention=True)
        sentences, input_data, mask = self.get_input_data()
        torch_tensor = _safe_import("torch.tensor")
        x = torch_tensor(input_data)
        m = torch_tensor(mask)
        _, a = attention(x, mask=m)
        max_len = input_data.shape[1]
        for i, sentence in enumerate(sentences):
            for j in range(len(sentence), max_len):
                for k in range(len(sentence), max_len):
                    assert a[i, j, k].item() < 1e-6
