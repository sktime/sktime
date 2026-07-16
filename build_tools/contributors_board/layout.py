"""Grid layout helpers for the contributor board."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BoardLayout:
    """Computed geometry for the contributor avatar grid."""

    columns: int
    rows: int
    width: int
    height: int
    avatar_size: int
    padding: int
    margin: int


def choose_columns(
    count: int,
    avatar_size: int,
    padding: int,
    margin: int,
) -> int:
    """Choose column count so the board is approximately square."""
    if count <= 0:
        return 1

    best_cols = 1
    best_score = float("inf")

    for cols in range(1, count + 1):
        rows = math.ceil(count / cols)
        width = 2 * margin + cols * avatar_size + (cols - 1) * padding
        height = 2 * margin + rows * avatar_size + (rows - 1) * padding
        score = abs(width - height)
        if score < best_score:
            best_score = score
            best_cols = cols

    return best_cols


def compute_layout(
    count: int,
    avatar_size: int = 64,
    padding: int = 8,
    margin: int = 16,
) -> BoardLayout:
    """Compute full board layout for a given contributor count."""
    columns = choose_columns(count, avatar_size, padding, margin)
    rows = max(1, math.ceil(count / columns)) if count else 1
    width = 2 * margin + columns * avatar_size + (columns - 1) * padding
    height = 2 * margin + rows * avatar_size + (rows - 1) * padding

    return BoardLayout(
        columns=columns,
        rows=rows,
        width=width,
        height=height,
        avatar_size=avatar_size,
        padding=padding,
        margin=margin,
    )


def avatar_position(
    index: int,
    layout: BoardLayout,
    total: int,
) -> tuple[int, int]:
    """Return top-left (x, y) for the avatar at ``index``."""
    row = index // layout.columns
    col = index % layout.columns

    avatars_in_row = min(layout.columns, total - row * layout.columns)
    row_offset = 0
    if avatars_in_row < layout.columns:
        empty_slots = layout.columns - avatars_in_row
        row_offset = (empty_slots * (layout.avatar_size + layout.padding)) // 2

    x = layout.margin + row_offset + col * (layout.avatar_size + layout.padding)
    y = layout.margin + row * (layout.avatar_size + layout.padding)
    return x, y
