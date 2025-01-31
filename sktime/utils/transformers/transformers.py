"""Dispatch transformers functions to isolate transformers."""

from sktime.utils.dependencies import _check_soft_dependencies

# Exports transformers if available, otherwise provides a dummy implementation
if _check_soft_dependencies("transformers", severity="none"):
    import transformers
else:

    class transformers:
        """Dummy transformers class if transformers library is unavailable."""

        def __init__(self):
            raise ImportError(
                "Please install transformers to use this functionality. "
                "You can install it with `pip install transformers`."
            )

        class BertConfig:
            """Dummy BertConfig class if unavailable."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Please install transformers to use this functionality. "
                    "You can install it with `pip install transformers`."
                )

        class BertModel:
            """Dummy BertModel class if unavailable."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Please install transformers to use this functionality. "
                    "You can install it with `pip install transformers`."
                )

        class BertTokenizer:
            """Dummy BertTokenizer class if unavailable."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Please install transformers to use this functionality. "
                    "You can install it with `pip install transformers`."
                )

        class GPT2Config:
            """Dummy GPT2Config class if unavailable."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Please install transformers to use this functionality. "
                    "You can install it with `pip install transformers`."
                )

        class GPT2Model:
            """Dummy GPT2Model class if unavailable."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Please install transformers to use this functionality. "
                    "You can install it with `pip install transformers`."
                )

        class GPT2Tokenizer:
            """Dummy GPT2Tokenizer class if unavailable."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Please install transformers to use this functionality. "
                    "You can install it with `pip install transformers`."
                )

        class LlamaConfig:
            """Dummy LlamaConfig class if unavailable."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Please install transformers to use this functionality. "
                    "You can install it with `pip install transformers`."
                )

        class LlamaModel:
            """Dummy LlamaModel class if unavailable."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Please install transformers to use this functionality. "
                    "You can install it with `pip install transformers`."
                )

        class LlamaTokenizer:
            """Dummy LlamaTokenizer class if unavailable."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Please install transformers to use this functionality. "
                    "You can install it with `pip install transformers`."
                )

        class logging:
            """Dummy logging class if unavailable."""

            def __init__(self, *args, **kwargs):
                raise ImportError(
                    "Please install transformers to use this functionality. "
                    "You can install it with `pip install transformers`."
                )

            class set_verbosity_error:
                """Dummy set_verbosity_error class if unavailable."""

                def __init__(self, *args, **kwargs):
                    raise ImportError(
                        "Please install transformers to use this functionality. "
                        "You can install it with `pip install transformers`."
                    )
