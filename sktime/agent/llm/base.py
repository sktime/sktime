# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class for LLM interfaces."""

from sktime.base import BaseObject

class BaseLLM(BaseObject):
    """Base class for LLM interfaces."""

    def __init__(self):
        super().__init__()

    def generate_code(self, prompt: str) -> str:
        """Generate Python code from a prompt.

        Parameters
        ----------
        prompt : str
            The prompt to send to the LLM.

        Returns
        -------
        str
            The generated Python code.
        """
        raise NotImplementedError("abstract method")
