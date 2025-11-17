# Purpose: Manage conversation-local asset storage for multi-modal inputs.
# Why: Keeps binary payloads out of history while enabling reproducible context.
"""Asset management utilities for multi-modal chat inputs."""

from __future__ import annotations

import base64
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List
from uuid import uuid4

import fitz  # PyMuPDF

from .config import DATA_DIR

PDF_RENDER_DPI = 200  # Higher DPI for clearer PDF page renders.
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}


@dataclass(frozen=True)
class StoredAsset:
    """Representation of an asset stored on disk."""

    path: Path
    mime_type: str


class ConversationAssets:
    """Manage assets belonging to a single conversation."""

    def __init__(self, base_dir: Path, conversation_id: str, pdf_render_dpi: int = PDF_RENDER_DPI) -> None:
        self._dir = base_dir / conversation_id
        self._pdf_render_dpi = pdf_render_dpi

    def store(self, source: Path) -> List[StoredAsset]:
        """Store an image or PDF file and return asset descriptors."""
        if not source.exists():
            raise FileNotFoundError(f"Asset not found: {source}")
        self._dir.mkdir(parents=True, exist_ok=True)
        suffix = source.suffix.lower()
        if suffix == ".pdf":
            return self._store_pdf(source)
        if suffix in IMAGE_MIME_TYPES:
            return [self._store_image(source)]
        raise ValueError(f"Unsupported asset type: {source}")

    def _store_image(self, source: Path) -> StoredAsset:
        dest = self._dir / f"{uuid4()}{source.suffix.lower()}"
        shutil.copyfile(source, dest)
        return StoredAsset(path=dest, mime_type=IMAGE_MIME_TYPES[source.suffix.lower()])

    def _store_pdf(self, source: Path) -> List[StoredAsset]:
        assets: List[StoredAsset] = []
        with fitz.open(source) as doc:
            scale = self._pdf_render_dpi / 72.0
            matrix = fitz.Matrix(scale, scale)
            for index in range(doc.page_count):
                page = doc.load_page(index)
                pix = page.get_pixmap(matrix=matrix)  # type: ignore[attr-defined]
                dest = self._dir / f"{uuid4()}_page{index + 1}.png"
                pix.save(str(dest))
                assets.append(StoredAsset(path=dest, mime_type="image/png"))
        if not assets:
            raise ValueError(f"PDF has no renderable pages: {source}")
        return assets


class AssetStore:
    """Factory for conversation-specific asset managers."""

    def __init__(self, base_dir: Path | None = None, pdf_render_dpi: int = PDF_RENDER_DPI) -> None:
        self._base_dir = base_dir or DATA_DIR / "assets"
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._pdf_render_dpi = pdf_render_dpi

    def for_conversation(self, conversation_id: str) -> ConversationAssets:
        """Return an asset manager scoped to the given conversation."""
        return ConversationAssets(self._base_dir, conversation_id, self._pdf_render_dpi)


def encode_asset(path: Path, mime_type: str) -> str:
    """Return a data URL for the asset path."""
    data = path.read_bytes()
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"
