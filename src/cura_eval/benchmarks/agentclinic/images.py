"""Fetch, cache, downscale, and base64-encode NEJM case images for vision turns."""

from __future__ import annotations

import base64
import hashlib
import io
import os
import time
import urllib.request
from pathlib import Path

from cura_eval.loaders import cache_root


class ImageFetchError(RuntimeError):
    pass


def _download(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "cura-eval/0.1"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read()


def fetch_image(url: str, *, retries: int = 3) -> Path:
    cache_dir = cache_root() / "agentclinic_images"
    cache_dir.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(url.split("?")[0])[1] or ".img"
    dest = cache_dir / (hashlib.sha256(url.encode()).hexdigest()[:16] + ext)
    if dest.exists():
        return dest
    last: Exception | None = None
    for attempt in range(retries):
        try:
            dest.write_bytes(_download(url))
            return dest
        except Exception as exc:  # retry any download/IO failure, then raise below
            last = exc
            time.sleep(min(2**attempt, 10))
    raise ImageFetchError(f"failed to fetch {url} after {retries} tries: {last}")


def _media_type(raw: bytes) -> str:
    """Sniff the image media type from magic bytes (the cache ext is often ``.img``)."""
    if raw[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if raw[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if raw[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def image_data_uri(path: Path, max_size: int = 1024) -> str:
    """``data:`` URI for ``path``, downscaled so the long edge <= ``max_size``.

    Downscaling bounds per-turn image-token cost; falls back to the original bytes when
    Pillow is missing or the decode fails.
    """
    raw = Path(path).read_bytes()
    media = _media_type(raw)
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(raw))
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size))
            fmt = "PNG" if media == "image/png" else "JPEG"
            if fmt == "JPEG" and img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = io.BytesIO()
            img.save(buf, format=fmt)
            raw = buf.getvalue()
            media = "image/png" if fmt == "PNG" else "image/jpeg"
    except Exception:
        pass  # send the original bytes; the provider may still accept/resize them
    return f"data:{media};base64,{base64.standard_b64encode(raw).decode()}"
