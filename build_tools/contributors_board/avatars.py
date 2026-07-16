"""Download and cache contributor avatar images."""

from __future__ import annotations

import os
import time
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw, ImageFont

AVATAR_URL = "https://avatars.githubusercontent.com/{username}?s=256"
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 2
PLACEHOLDER_COLOR = (200, 200, 200, 255)
PLACEHOLDER_TEXT_COLOR = (80, 80, 80, 255)


def _github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github.v3+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"
    return headers


def _cache_path(cache_dir: Path, username: str) -> Path:
    safe_name = username.replace("/", "_")
    return cache_dir / f"{safe_name}.png"


def _make_placeholder(username: str, size: int) -> Image.Image:
    """Create a circular placeholder avatar with initials."""
    img = Image.new("RGBA", (size, size), PLACEHOLDER_COLOR)
    draw = ImageDraw.Draw(img)

    initials = username[:2].upper()
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size // 3,
        )
    except OSError:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), initials, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.text(
        ((size - text_w) / 2, (size - text_h) / 2 - bbox[1]),
        initials,
        fill=PLACEHOLDER_TEXT_COLOR,
        font=font,
    )
    return img


def fetch_avatar(
    username: str,
    cache_dir: Path,
    size: int = 256,
) -> Image.Image:
    """Fetch a contributor avatar, using the local cache when available.

    Parameters
    ----------
    username : str
        GitHub username.
    cache_dir : Path
        Directory for cached avatar PNG files.
    size : int
        Target avatar size in pixels (used for placeholders).

    Returns
    -------
    PIL.Image.Image
        RGBA avatar image.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = _cache_path(cache_dir, username)

    if cached.exists():
        return Image.open(cached).convert("RGBA")

    url = AVATAR_URL.format(username=username)
    headers = _github_headers()

    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url, headers=headers, timeout=30)
            if response.status_code == 404:
                break
            if response.status_code == 429:
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
                continue
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGBA")
            img.save(cached, "PNG")
            return img
        except requests.RequestException:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            continue

    placeholder = _make_placeholder(username, size)
    placeholder.save(cached, "PNG")
    return placeholder
