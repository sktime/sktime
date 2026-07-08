"""Render the contributor avatar board PNG."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageOps

from build_tools.contributors_board.layout import avatar_position, compute_layout


def make_circular_avatar(img: Image.Image, size: int) -> Image.Image:
    """Crop an image to a circle of the given diameter."""
    fitted = ImageOps.fit(img.convert("RGBA"), (size, size), Image.Resampling.LANCZOS)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((0, 0, size, size), fill=255)
    output = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    output.paste(fitted, (0, 0), mask=mask)
    return output


def render_board(
    avatars: list[Image.Image],
    avatar_size: int = 64,
    padding: int = 8,
    margin: int = 16,
) -> Image.Image:
    """Composite avatars into a single transparent PNG board."""
    layout = compute_layout(len(avatars), avatar_size, padding, margin)
    board = Image.new("RGBA", (layout.width, layout.height), (0, 0, 0, 0))

    for index, avatar in enumerate(avatars):
        circular = make_circular_avatar(avatar, layout.avatar_size)
        x, y = avatar_position(index, layout, len(avatars))
        board.paste(circular, (x, y), circular)

    return board
