import numpy as np
import pytest

from sktime.tests.test_switch import run_test_module_changed
from sktime.utils.dependencies import _check_soft_dependencies, _safe_import

keras = _safe_import("tensorflow.keras")


@pytest.mark.skipif(
    not run_test_module_changed("sktime.libs._keras_self_attention")
    or not _check_soft_dependencies("tensorflow", severity="none"),
    reason="Execute tests for iff anything in the module has changed",
)
class TestMaskShape:
    @staticmethod
    def get_input_data():
        sentences = [
            ["All", "work", "and", "no", "play"],
            ["makes", "Jack", "a", "dull", "boy", "."],
            ["From", "that", "day", "forth", "my", "arm", "changed"],
        ]
        token_dict = {
            "": 0,
            "<UNK>": 1,
        }
        sentence_len = max(map(len, sentences))
        input_data = [[0] * sentence_len for _ in range(len(sentences))]
        for i, sentence in enumerate(sentences):
            for j, token in enumerate(sentence):
                if token in token_dict:
                    input_data[i][j] = token_dict[token]
                elif np.random.randint(0, 5) == 0:
                    input_data[i][j] = token_dict["<UNK>"]
                else:
                    input_data[i][j] = len(token_dict)
                    token_dict[token] = len(token_dict)
        return sentences, np.asarray(input_data), token_dict

    @staticmethod
    def get_model(attention, token_dict):
        inputs = keras.layers.Input(shape=(None,), name="Input")
        embd = keras.layers.Embedding(
            input_dim=len(token_dict), output_dim=16, mask_zero=True, name="Embedding"
        )(inputs)
        lstm = keras.layers.Bidirectional(
            keras.layers.LSTM(units=16, return_sequences=True), name="Bi-LSTM"
        )(embd)
        if attention.return_attention:
            att, weights = attention(lstm)
        else:
            att = attention(lstm)
        dense = keras.layers.Dense(units=5, name="Dense")(att)
        loss = {"Dense": "sparse_categorical_crossentropy"}
        if attention.return_attention:
            model = keras.models.Model(inputs=inputs, outputs=[dense, weights])
            loss[attention.name] = "mse"
        else:
            model = keras.models.Model(inputs=inputs, outputs=dense)
        model.compile(optimizer="adam", loss=loss)
        model.summary(line_length=100)
        return model

    def check_mask_shape(self, attention):
        sentences, input_data, token_dict = self.get_input_data()
        model = self.get_model(attention, token_dict)
        outputs = model.predict(input_data)
        if attention.attention_width is None:
            attention_width = 1e9
        else:
            attention_width = attention.attention_width
        history_only = attention.history_only
        attention_output = outputs[1]
        for i, sentence in enumerate(sentences):
            for j in range(len(sentence)):
                for k in range(len(sentence)):
                    if history_only and 0 <= j - k < attention_width:
                        assert attention_output[i][j][k] > 0.0
                    elif not history_only and abs(j - k) <= attention_width // 2:
                        assert attention_output[i][j][k] > 0.0
                    else:
                        assert attention_output[i][j][k] == 0.0
                assert abs(np.sum(attention_output[i][j]) - 1.0) < 1e-6
