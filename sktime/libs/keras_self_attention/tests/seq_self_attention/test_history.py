from keras_self_attention import SeqSelfAttention
from .util import TestMaskShape


class TestHistory(TestMaskShape):

    def test_history(self):
        attention = SeqSelfAttention(return_attention=True,
                                     attention_width=3,
                                     history_only=True,
                                     name='Attention')
        self.check_mask_shape(attention)

    def test_infinite_history(self):
        attention = SeqSelfAttention(return_attention=True,
                                     history_only=True,
                                     name='Attention')
        self.check_mask_shape(attention)
