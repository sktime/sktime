#!/usr/bin/env python3
"""Generate the static contributor avatar board for the README."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from build_tools.contributors_board.avatars import fetch_avatar  # noqa: E402
from build_tools.contributors_board.commits import (  # noqa: E402
    fetch_commit_counts,
    sort_logins,
)
from build_tools.contributors_board.parse import parse_contributors  # noqa: E402
from build_tools.contributors_board.render import render_board  # noqa: E402

DEFAULT_CONTRIBUTORS_MD = REPO_ROOT / "CONTRIBUTORS.md"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "source" / "images" / "contributors-board.png"
DEFAULT_CACHE = REPO_ROOT / ".cache" / "contributors_board"


def generate_board(
    contributors_md: Path = DEFAULT_CONTRIBUTORS_MD,
    output_path: Path = DEFAULT_OUTPUT,
    cache_dir: Path = DEFAULT_CACHE,
    avatar_size: int = 40,
    padding: int = 8,
    margin: int = 16,
) -> Path:
    """Generate the contributor board PNG and return the output path."""
    markdown_text = contributors_md.read_text(encoding="utf-8")
    logins = parse_contributors(markdown_text)
    if not logins:
        raise ValueError(f"No contributors found in {contributors_md}")

    commit_counts = fetch_commit_counts()
    ordered = sort_logins(logins, commit_counts)

    avatars = [fetch_avatar(login, cache_dir) for login in ordered]
    board = render_board(
        avatars,
        avatar_size=avatar_size,
        padding=padding,
        margin=margin,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    board.save(output_path, "PNG", optimize=True, compress_level=9)
    return output_path


def main() -> None:
    """Run the contributor board generator from the command line."""
    parser = argparse.ArgumentParser(
        description="Generate a static contributor avatar board from CONTRIBUTORS.md.",
    )
    parser.add_argument(
        "--contributors-md",
        type=Path,
        default=DEFAULT_CONTRIBUTORS_MD,
        help="Path to CONTRIBUTORS.md",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PNG path",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Avatar cache directory",
    )
    parser.add_argument("--avatar-size", type=int, default=40)
    parser.add_argument("--padding", type=int, default=8)
    parser.add_argument("--margin", type=int, default=16)
    args = parser.parse_args()

    output = generate_board(
        contributors_md=args.contributors_md,
        output_path=args.output,
        cache_dir=args.cache_dir,
        avatar_size=args.avatar_size,
        padding=args.padding,
        margin=args.margin,
    )
    print(f"Wrote contributor board to {output}")


if __name__ == "__main__":
    main()
