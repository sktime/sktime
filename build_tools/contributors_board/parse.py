"""Parse GitHub usernames from CONTRIBUTORS.md."""

from __future__ import annotations

import re
from html import unescape

LIST_START = "ALL-CONTRIBUTORS-LIST:START"
LIST_END = "ALL-CONTRIBUTORS-LIST:END"

CELL_RE = re.compile(
    r'<td align="center"[^>]*>(.*?)</td>',
    re.DOTALL | re.IGNORECASE,
)
PROFILE_RE = re.compile(
    r'href="https://github\.com/([^"/?#]+)/?"',
    re.IGNORECASE,
)
AUTHOR_RE = re.compile(
    r"github\.com/sktime/sktime/"
    r"(?:commits\?author=|issues\?q=author(?:%3A|=))([^&\"]+)",
    re.IGNORECASE,
)

RESERVED_LOGINS = frozenset({"sktime", "sktime-org"})


def _extract_login(cell: str) -> str | None:
    """Extract a GitHub login from a single contributor table cell."""
    for match in PROFILE_RE.finditer(cell):
        candidate = unescape(match.group(1))
        if candidate.lower() in RESERVED_LOGINS:
            continue
        if "/" in candidate:
            continue
        return candidate

    author_match = AUTHOR_RE.search(cell)
    if author_match:
        return unescape(author_match.group(1))

    return None


def parse_contributors(markdown_text: str) -> list[str]:
    """Parse unique GitHub usernames from CONTRIBUTORS.md in document order.

    Parameters
    ----------
    markdown_text : str
        Full contents of CONTRIBUTORS.md.

    Returns
    -------
    list[str]
        Deduplicated GitHub usernames in first-seen order.
    """
    start = markdown_text.index(LIST_START)
    end = markdown_text.index(LIST_END)
    section = markdown_text[start:end]

    seen: set[str] = set()
    logins: list[str] = []

    for cell in CELL_RE.findall(section):
        login = _extract_login(cell)
        if login is None:
            continue
        key = login.lower()
        if key in seen:
            continue
        seen.add(key)
        logins.append(login)

    return logins
