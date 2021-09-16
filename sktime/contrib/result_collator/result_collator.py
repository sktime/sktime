# -*- coding: utf-8 -*-
"""File contains classes needed to collect results from https resource."""
__all__ = ["ResultCollator"]

from typing import List, Any
import requests


class ResultCollator:
    """ResultCollator is a class used to collate results from external sources.

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
        """Method used to get results by requesting and formatting response from urls.

        Returns
        -------
        formatted_response: List[Any]
            List of formatted responses
        """
        responses: List[str] = []
        for url in self.urls:
            response: str = (ResultCollator.request_resource(url)).text
            responses.append(self._format_result(response))
        return responses

    @staticmethod
    def request_resource(url: str, num_attempts: int = 3) -> requests.Response:
        """Method used to request a resource.

        Parameters
        ----------
        url: str
            Str that is the url to request resource from

        num_attempts: int, defaults = 3
            Int that is the number of times to retry requesting
            the resource

        Returns
        -------
        response: Any
            Response from the resource
        """
        try:
            response: requests.Response = requests.get(url)
        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            if num_attempts >= 1:
                ResultCollator.request_resource(url, num_attempts=num_attempts - 1)
            else:
                raise requests.exceptions.RequestException(
                    "Request continued to fail after "
                    "multiple retries. Please try a new "
                    "url or try again later"
                )
        except requests.exceptions.TooManyRedirects:
            raise requests.exceptions.TooManyRedirects(
                "Either the url is bad or "
                "the server is down. Change the "
                "url or try again later"
            )
        return response
