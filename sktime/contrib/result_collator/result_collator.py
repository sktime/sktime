# -*- coding: utf-8 -*-
__all__ = ["ResultCollator"]

from typing import List, Any
from httpx import AsyncClient, Response
import requests
import asyncio


class ResultCollator:
    """
    ResultCollator is a class used to collate results from external sources
    i.e. timeseriesclassification.com

    Parameters
    ----------
    urls: List[str]
        Array of urls to get results from
    """

    def __init__(self, urls: List[str]):
        self.urls: List[str] = urls

    def _format_result(self, response: str) -> Any:
        """
        Method that is used to take the result and format it for your
        use case.

        Parameters
        ----------
        response: str
            str response

        Returns
        -------
        formatted_response: Any
            response in the format for use case can be of any type
        """
        raise NotImplementedError("abstract method")

    def get_results(self) -> List[Any]:
        """
        Method used to get results by requesting and formatting response
        from urls

        Returns
        -------
        formatted_response: List[Any]
            List of formatted responses
        """
        responses: Response = asyncio.run(self.async_request(self.urls))

        formatted_responses: List[Any] = []
        for response in responses:
            formatted_responses.append(self._format_result(response.text))
        return formatted_responses

    @staticmethod
    async def async_request(urls) -> List[Response]:
        """
        Method used to asynchronously request data from urls

        Returns
        -------
        response: List[Responses]
            List contains the response objects
        """
        async with AsyncClient() as client:
            tasks = (client.get(url) for url in urls)
            responses = await asyncio.gather(*tasks)

        return responses
