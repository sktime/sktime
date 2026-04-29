# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Base class for agents."""

from sktime.base import BaseObject

class BaseAgent(BaseObject):
    """Base class for sktime agents.

    Agents are objects that can take natural language queries and
    return sktime objects or perform actions.
    """

    def __init__(self):
        super().__init__()

    def ask(self, query: str):
        """Process a natural language query.

        Parameters
        ----------
        query : str
            The natural language query to process.

        Returns
        -------
        object
            An sktime object or a result of the query.
        """
        raise NotImplementedError("abstract method")
