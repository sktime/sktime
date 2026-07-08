"""Fetch commit counts and sort contributors."""

from __future__ import annotations

import os
import time

import requests

DEFAULT_OWNER = "sktime"
DEFAULT_REPO = "sktime"
STATS_URL = "https://api.github.com/repos/{owner}/{repo}/stats/contributors"
MAX_RETRIES = 8
RETRY_DELAY_SECONDS = 5


def _github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def fetch_commit_counts(
    owner: str = DEFAULT_OWNER,
    repo: str = DEFAULT_REPO,
) -> dict[str, int]:
    """Fetch all-time commit counts per GitHub login from the GitHub API.

    The stats endpoint may return HTTP 202 while data is being generated;
    this function retries until a 200 response is received.

    Parameters
    ----------
    owner : str
        Repository owner.
    repo : str
        Repository name.

    Returns
    -------
    dict[str, int]
        Mapping of lowercase GitHub login to total commit count.
    """
    url = STATS_URL.format(owner=owner, repo=repo)
    headers = _github_headers()

    for _attempt in range(MAX_RETRIES):
        response = requests.get(url, headers=headers, timeout=30)
        if response.status_code == 202:
            time.sleep(RETRY_DELAY_SECONDS)
            continue
        response.raise_for_status()
        counts: dict[str, int] = {}
        for entry in response.json():
            author = entry.get("author")
            if not author:
                continue
            login = author.get("login")
            if not login:
                continue
            counts[login.lower()] = int(entry.get("total", 0))
        return counts

    raise RuntimeError(
        f"GitHub stats/contributors did not become ready after {MAX_RETRIES} attempts."
    )


def sort_logins(logins: list[str], commit_counts: dict[str, int]) -> list[str]:
    """Sort logins by commit count descending, then username ascending.

    Parameters
    ----------
    logins : list[str]
        GitHub usernames from CONTRIBUTORS.md.
    commit_counts : dict[str, int]
        Mapping of lowercase login to commit count.

    Returns
    -------
    list[str]
        Sorted usernames.
    """
    return sorted(
        logins,
        key=lambda login: (-commit_counts.get(login.lower(), 0), login.lower()),
    )
