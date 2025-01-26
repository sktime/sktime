"""Dispatch torch functions used to isolate torch."""

from sktime.utils.dependencies import _check_soft_dependencies

# exports torch if torch is present, otherwise an identity torch
if _check_soft_dependencies("torch", severity="none"):
    import torch

else:

    class torch:
        """Dummy class if torch is unavailable."""

        def __init__(self):
            raise ImportError(
                "Please install torch to use this functionality. "
                "You can install it with `pip install torch`."
            )

        class nn:
            """Dummy nn class if torch.nn is unavailable."""

            def __init__(self):
                raise ImportError(
                    "Please install torch to use this functionality. "
                    "You can install it with `pip install torch`."
                )

            class Module:
                """Dummy nn.Module class if unavailable."""

                def __init__(self, *args, **kwargs):
                    raise ImportError(
                        "Please install torch to use this functionality. "
                        "Please install torch with `pip install torch`."
                    )
