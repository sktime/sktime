"""Tests for the contributors board generator."""

from PIL import Image

from build_tools.contributors_board.commits import sort_logins
from build_tools.contributors_board.layout import choose_columns, compute_layout
from build_tools.contributors_board.parse import parse_contributors
from build_tools.contributors_board.render import make_circular_avatar, render_board

SAMPLE_MARKDOWN = """
<!-- ALL-CONTRIBUTORS-LIST:START -->
<table>
  <tbody>
    <tr>
      <td align="center">
        <a href="https://github.com/alice"><img alt="Alice"/></a>
      </td>
      <td align="center">
        <a href="https://www.linkedin.com/in/bob/"><img alt="Bob"/></a>
        <a href="https://github.com/sktime/sktime/commits?author=bob-dev">💻</a>
      </td>
      <td align="center">
        <a href="https://github.com/alice"><img alt="Alice duplicate"/></a>
      </td>
    </tr>
  </tbody>
</table>
<!-- ALL-CONTRIBUTORS-LIST:END -->
"""


def test_parse_contributors_extracts_github_and_fallback():
    logins = parse_contributors(SAMPLE_MARKDOWN)
    assert logins == ["alice", "bob-dev"]


def test_sort_logins_by_commits_then_name():
    logins = ["zebra", "alice", "bob"]
    counts = {"alice": 10, "bob": 10, "zebra": 1}
    assert sort_logins(logins, counts) == ["alice", "bob", "zebra"]


def test_choose_columns_prefers_square_layout():
    assert choose_columns(16, avatar_size=64, padding=8, margin=16) == 4


def test_compute_layout_dimensions():
    layout = compute_layout(10, avatar_size=64, padding=8, margin=16)
    assert layout.columns == 3
    assert layout.rows == 4
    assert layout.width == 3 * 64 + 2 * 8 + 2 * 16
    assert layout.height == 4 * 64 + 3 * 8 + 2 * 16


def test_render_board_produces_rgba_image():
    avatars = [Image.new("RGBA", (64, 64), (255, 0, 0, 255)) for _ in range(4)]
    board = render_board(avatars, avatar_size=32, padding=4, margin=8)
    assert board.mode == "RGBA"
    assert board.width > 0 and board.height > 0


def test_make_circular_avatar_is_square():
    avatar = Image.new("RGBA", (100, 50), (0, 255, 0, 255))
    circular = make_circular_avatar(avatar, 32)
    assert circular.size == (32, 32)
