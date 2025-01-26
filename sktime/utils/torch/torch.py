"""Dispatch torch functions used to isolate torch."""

from sktime.utils.dependencies import _check_soft_dependencies

# exports torch if torch is present, otherwise an identity torch
if _check_soft_dependencies("torch", severity="none"):
    import torch

else:

    class torch:
        """Dummy class if torch is unavailable."""

        class nn:
            """Dummy class if nn is unavailable."""

            class Module:
                """Dummy class if nn.Module is unavailable."""

                pass

            pass

        pass
