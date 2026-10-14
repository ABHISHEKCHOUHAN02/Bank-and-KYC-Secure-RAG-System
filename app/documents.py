"""Local document management with full-index rebuilds after corpus changes.

Uploaded files live under data/raw/uploads and can be removed through the API.
The original RBI and fictional-bank source folders are listed but are never
deletable through these management functions.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from uuid import uuid4

from app.config import settings
from retrieval.ingest import SUPPORTED_EXTS, build_index, discover_files

logger = logging.getLogger(__name__)

# API query and corpus-write operations share this lock so requests cannot query
# while Chroma/BM25/chunks are being replaced.
INDEX_LOCK = threading.RLock()
MAX_UPLOAD_BYTES = 25 * 1024 * 1024

UPLOAD_DIRS = {
    "rbi": settings.uploads_rbi_dir,
    "fictional_bank": settings.uploads_fictional_dir,
}
class DocumentOperationError(ValueError):
    """A document upload/removal request is invalid or conflicts with the corpus."""


class IndexRebuildError(RuntimeError):
    """A corpus change failed to rebuild the indexes."""


def _invalidate_loaded_indexes() -> None:
    """Drop process-local readers and cache namespace after an index rebuild."""
    from app.pipeline import get_graph
    from cache.semantic import get_semantic_cache
    from retrieval.hybrid import get_retriever

    get_graph.cache_clear()
    get_retriever.cache_clear()
    get_semantic_cache.cache_clear()


def _rebuild() -> None:
    build_index()
    _invalidate_loaded_indexes()


def list_documents() -> list[dict]:
    with INDEX_LOCK:
        entries = []
        managed_dirs = {path.resolve() for path in UPLOAD_DIRS.values()}
        for path, source_type in discover_files():
            entries.append({
                "filename": path.name,
                "source_type": source_type.value,
                "managed_upload": path.parent.resolve() in managed_dirs,
                "size_bytes": path.stat().st_size,
            })
        return entries


def _validate_filename(filename: str) -> str:
    if (not filename or len(filename) > 120 or filename in {".", ".."}
            or Path(filename).name != filename
            or any(char in filename for char in '<>:/\\|?*"')
            or any(ord(char) < 32 for char in filename)):
        raise DocumentOperationError("Use a simple filename without folders or special path characters.")
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTS:
        raise DocumentOperationError("Supported file types are PDF, TXT, and Markdown.")
    if filename.rstrip(" .") != filename:
        raise DocumentOperationError("Filename cannot end with a dot or space.")
    if Path(filename).stem.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
                                      *(f"LPT{i}" for i in range(1, 10))}:
        raise DocumentOperationError("That filename is reserved by Windows.")
    return filename


def _validate_file_bytes(filename: str, content: bytes) -> None:
    if not content:
        raise DocumentOperationError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise DocumentOperationError("Maximum upload size is 25 MiB.")
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf" and not content.startswith(b"%PDF-"):
        raise DocumentOperationError("The file does not have a valid PDF header.")
    if suffix in {".txt", ".md"}:
        try:
            content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise DocumentOperationError("Text and Markdown files must use UTF-8 encoding.") from exc


def upload_document(filename: str, source_type: str, content: bytes) -> dict:
    filename = _validate_filename(filename)
    if source_type not in UPLOAD_DIRS:
        raise DocumentOperationError("source_type must be 'rbi' or 'fictional_bank'.")
    _validate_file_bytes(filename, content)

    with INDEX_LOCK:
        known_names = {path.name.casefold() for path, _ in discover_files()}
        if filename.casefold() in known_names:
            raise DocumentOperationError("A document with that filename already exists.")

        target_dir = UPLOAD_DIRS[source_type]
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / filename
        temporary = target_dir / f".{uuid4().hex}.uploading"
        temporary.write_bytes(content)
        temporary.replace(target)
        try:
            _rebuild()
        except Exception as exc:
            target.unlink(missing_ok=True)
            try:
                _rebuild()
            except Exception:
                logger.exception("Failed to restore the previous index after upload failure.")
                raise IndexRebuildError(
                    "Index rebuild and recovery both failed. Stop the API and run `uv run python -m retrieval.ingest`."
                ) from exc
            raise IndexRebuildError(
                "Index rebuild failed; the uploaded file was removed and the previous corpus was rebuilt."
            ) from exc

        return {
            "filename": filename,
            "source_type": source_type,
            "indexed": True,
            "message": "Document uploaded and the full search index was rebuilt.",
        }


def delete_document(filename: str, source_type: str) -> dict:
    filename = _validate_filename(filename)
    if source_type not in UPLOAD_DIRS:
        raise DocumentOperationError("source_type must be 'rbi' or 'fictional_bank'.")

    with INDEX_LOCK:
        upload_dir = UPLOAD_DIRS[source_type].resolve()
        target = (upload_dir / filename).resolve()
        if target.parent != upload_dir or not target.is_file():
            raise DocumentOperationError("Only documents in the uploads area can be removed here.")

        staged = upload_dir / f".{uuid4().hex}.deleting"
        target.replace(staged)
        try:
            _rebuild()
        except Exception as exc:
            staged.replace(target)
            try:
                _rebuild()
            except Exception:
                logger.exception("Failed to restore the previous index after removal failure.")
                raise IndexRebuildError(
                    "Index rebuild and recovery both failed. Stop the API and run `uv run python -m retrieval.ingest`."
                ) from exc
            raise IndexRebuildError(
                "Index rebuild failed; the document was restored and the previous corpus was rebuilt."
            ) from exc

        staged.unlink(missing_ok=True)
        return {
            "filename": filename,
            "source_type": source_type,
            "indexed": True,
            "message": "Uploaded document removed and the full search index was rebuilt.",
        }
