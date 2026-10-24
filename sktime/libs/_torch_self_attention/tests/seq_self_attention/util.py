import numpy as np

from sktime.utils.dependencies import _safe_import


class TestMaskShapeTorch:
    @staticmethod
    def get_input_data(n_features=8):
        sentences = [
            ["All", "work", "and", "no", "play"],
            ["makes", "Jack", "a", "dull", "boy", "."],
            ["From", "that", "day", "forth", "my", "arm", "changed"],
        ]
        sentence_len = max(map(len, sentences))
        rng = np.random.default_rng(12345)
        input_data = np.zeros(
            (len(sentences), sentence_len, n_features), dtype=np.float32
        )
        mask = np.zeros((len(sentences), sentence_len), dtype=np.float32)
        for i, sentence in enumerate(sentences):
            cur_len = len(sentence)
            mask[i, :cur_len] = 1.0
            input_data[i, :cur_len] = rng.normal(size=(cur_len, n_features))
        return sentences, input_data, mask

    def check_mask_shape(self, attention):
        sentences, input_data, mask = self.get_input_data()
        torch_tensor = _safe_import("torch.tensor")
        x = torch_tensor(input_data)
        m = torch_tensor(mask)
        if attention.return_attention:
            v, a = attention(x, mask=m)
        else:
            v = attention(x, mask=m)
            a = attention.attention
        assert v.shape == x.shape
        assert a.shape == (x.shape[0], x.shape[1], x.shape[1])

        if attention.attention_width is None:
            attention_width = 1e9
        else:
            attention_width = attention.attention_width
        history_only = attention.history_only
        for i, sentence in enumerate(sentences):
            for j in range(len(sentence)):
                row = a[i, j]
                assert abs(row.sum().item() - 1.0) < 1e-5
                for k in range(len(sentence)):
                    if history_only:
                        allowed = 0 <= j - k < attention_width
                    else:
                        allowed = abs(j - k) <= attention_width // 2
                    if allowed:
                        assert row[k].item() > 0.0
                    else:
                        assert row[k].item() < 1e-6
