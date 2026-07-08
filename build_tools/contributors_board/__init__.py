"""Generate a static contributor avatar board from CONTRIBUTORS.md."""

from build_tools.contributors_board.avatars import fetch_avatar
from build_tools.contributors_board.commits import fetch_commit_counts, sort_logins
from build_tools.contributors_board.parse import parse_contributors
from build_tools.contributors_board.render import render_board

__all__ = [
    "parse_contributors",
    "fetch_commit_counts",
    "sort_logins",
    "fetch_avatar",
    "render_board",
]
