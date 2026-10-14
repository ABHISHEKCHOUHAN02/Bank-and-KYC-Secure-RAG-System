"""
Ingestion pipeline for bank-rag.

    raw PDFs/MD/TXT  ->  clean  ->  split into sections  ->  token-based chunks
                     ->  data/indexes/chunks.jsonl   (source of truth for chunks)
                     ->  ChromaDB collection         (dense retrieval)
                     ->  data/indexes/bm25.pkl       (sparse retrieval)

Run from the repo root:
    python -m retrieval.ingest

Metadata priority per document:
    1. data/manifest.json entry (keyed by file name)  <- most reliable
    2. regex extraction from the first pages          <- best effort only

Manifest format (flat, or nested under a "documents" key):
    {
      "documents": {
        "kyc_master_direction.pdf": {
          "title": "Master Direction - Know Your Customer (KYC) Direction",
          "circular_no": "RBI/DBR/2015-16/18",
          "date": "2016-02-25",
          "url": "https://...",
          "download_date": "2026-10-11"
        }
      }
    }

Security note: bm25.pkl is a pickle. Only load files you generated yourself.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import logging
import pickle
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.schemas import Chunk, ChunkMetadata, SourceType

logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pdf", ".txt", ".md"}
MIN_SECTION_CHARS = 200  # sections shorter than this are merged into the next one

# ======================================================================
# Cleanup helpers (markdown tables, headings, junk chunks, context header, amounts)
# ======================================================================
# ----------------------------------------------------------------------
# 1. Markdown + ASCII-table cleanup (apply to .md/.txt files only)
#    Turns  "* **\+-----+-----+**" / "* **| a | b |**"  into a real markdown table,
#    strips ** and \ escapes, converts "* " bullets to "- ".
# ----------------------------------------------------------------------
_TABLE_BULLET_RE = re.compile(r"^\s*[*\-]\s+(?=\*{0,2}\s*(?:\||\\?\+[-=]))")
_BULLET_RE = re.compile(r"^(\s*)\*\s+")
_UNESCAPE_RE = re.compile(r"\\([\\`*_{}\[\]()#+\-.!|>~])")
_RULE_RE = re.compile(r"^\s*\+[-=+]{3,}\+\s*$")  # +-----+-----+
_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")  # | a | b |


def _clean_line(line: str) -> str:
    line = _TABLE_BULLET_RE.sub("", line)  # bullet in front of a table line
    line = _BULLET_RE.sub(r"\1- ", line)  # other "* " bullets -> "- "
    line = line.replace("*", "")  # bold / italics markers
    line = _UNESCAPE_RE.sub(r"\1", line)  # "4\." -> "4.", "\+" -> "+"
    return line.rstrip()


def _cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _render_block(block: list[str]) -> list[str]:
    if not any(_RULE_RE.match(l) for l in block):
        return [l.strip() for l in block]  # already a proper markdown table

    # ASCII table: lines between rule lines form ONE logical row (multi-line cells).
    logical: list[list[list[str]]] = []
    cur: list[list[str]] = []
    for l in block:
        if _RULE_RE.match(l):
            if cur:
                logical.append(cur)
                cur = []
        else:
            cur.append(_cells(l))
    if cur:
        logical.append(cur)

    rows: list[list[str]] = []
    for phys in logical:
        ncol = max(len(c) for c in phys)
        rows.append(
            [" ".join(p for p in (c[i] if i < len(c) else "" for c in phys) if p) for i in range(ncol)]
        )
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + " --- |" * width]
    out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return out


def normalize_markdown(text: str) -> str:
    out: list[str] = []
    block: list[str] = []

    def flush() -> None:
        if block:
            out.extend(_render_block(block))
            block.clear()

    for line in (_clean_line(l) for l in text.splitlines()):
        if _RULE_RE.match(line) or _ROW_RE.match(line):
            block.append(line)
        else:
            flush()
            out.append(line)
    flush()
    return "\n".join(out)


# ----------------------------------------------------------------------
# 2. Safer heading detection (replaces HEADING_RES[0], CHAPTER_ALONE_RE and the
#    "next line is the title" check). Case-sensitive, so body text like
#    "Section 13 of the PML Act..." or "Annex I of these Directions..." no longer matches.
# ----------------------------------------------------------------------
HEADING_KEYWORD_RE = re.compile(
    r"^(?:Chapter|CHAPTER|Part|PART|Annex(?:ure)?|ANNEX(?:URE)?)\s*[-–]?\s*"
    r"(?:[IVXLCDM]+|\d+|[A-Z])[A-Za-z]?\.?(?:\s*[-–:]\s*\S.*|\s*\(.*\))?$"
)

CHAPTER_ALONE_RE = re.compile(
    r"^(?:Chapter|CHAPTER|Part|PART|Annex(?:ure)?|ANNEX(?:URE)?)\s*[-–]?\s*"
    r"(?:[IVXLCDM]+|\d+|[A-Z])[A-Za-z]?\.?\s*$"
)
_ENUM_START_RE = re.compile(r"^\(?(?:[ivxlcdm]+|[a-z]|\d+)[.)]", re.I)


def looks_like_title(s: str) -> bool:
    """Is this next line a plausible chapter title (and not 'iv) any OVD or proof of...')?"""
    s = s.strip()
    return (
        bool(s)
        and len(s) <= 80
        and s[0].isupper()
        and not s.endswith((",", ";", ":"))
        and not _ENUM_START_RE.match(s)
    )


# ----------------------------------------------------------------------
# 3. Junk-chunk filter ("19. Deleted  20. Deleted ...", near-empty chunks)
# ----------------------------------------------------------------------
_DELETED_RE = re.compile(
    r"^\s*\d+[A-Za-z]?\.?\s*(?:\(?\w{1,4}\)\s*)?\[?(?:deleted|omitted|repealed)\]?\.?\s*$", re.I
)


def is_junk_chunk(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return True
    if sum(bool(_DELETED_RE.match(l)) for l in lines) / len(lines) >= 0.5:
        return True
    few_words = len(re.findall(r"[A-Za-z]{3,}", text)) < 8
    few_figures = len(re.findall(r"\d+", text)) < 2  # keep short tables / fee lists
    return few_words and few_figures


# ----------------------------------------------------------------------
# 4. Context header for embeddings and BM25 (store raw c.text in Chroma documents)
# ----------------------------------------------------------------------
def contextualize(c) -> str:
    m = c.metadata
    head = " | ".join(x for x in (m.doc, m.circular_no, m.section) if x)
    return f"{head}\n{c.text}"


# ----------------------------------------------------------------------
# 5. Amount normalisation inside tokenize(): "₹10,000" -> "10000", "1,00,000" -> "100000"
# ----------------------------------------------------------------------
_THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{2,3}(?:\D|$))")


def normalize_numbers(text: str) -> str:
    return _THOUSANDS_RE.sub("", text)

# ----------------------------------------------------------------------
# Regexes (heuristics: tune them after inspecting your actual PDFs)
# ----------------------------------------------------------------------
TOC_LINE_RE = re.compile(r"\.{4,}\s*\d+\s*$|^\s*(?:table of )?contents\s*$", re.I)  # "Introduction ..... 4", "CONTENTS"
PAGE_NUMBER_RE = re.compile(r"^\s*(?:page\s+)?\d+(?:\s*(?:of|/)\s*\d+)?\s*$", re.I)

HEADING_RES = [
    # "Chapter II", "PART A", "Annex 1", "Annexure - II"
    HEADING_KEYWORD_RE,  # case-sensitive: body text like "Section 13 of the Act..." no longer matches
    # NOTE: a generic "3. Title" numbered-line rule was removed. RBI paragraphs are numbered,
    # so it mislabelled wrapped sentences and "22. Deleted" as section headings.
    # ALL CAPS line
    re.compile(r"^[A-Z][A-Z ,&()\-/]{5,80}$"),  # no digits: avoids matching "RBI/2024-2025/97"
    # Markdown headings
    re.compile(r"^#{1,4}\s+.+$"),
]

# "CHAPTER III" / "Part V" on a line by itself: the title is on the next line.
# (CHAPTER_ALONE_RE is defined in the cleanup helpers above)
# Footnote numbers glued to the next word, e.g. "116Use", "73Notwithstanding".
# Heuristic: 1-3 digits preceded by whitespace/start and followed by Capital+lowercase.
FOOTNOTE_GLUE_RE = re.compile(r"(?<![^\s])\d{1,3}(?=[A-Z][a-z])")

RBI_NO_RE = re.compile(r"RBI/(?:[A-Za-z&.]+/)?\d{4}-\d{2,4}/\d+")
_MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*"
DATE_MDY_RE = re.compile(rf"\b({_MONTH})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b")
DATE_DMY_RE = re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTH})\.?,?\s+(\d{{4}})\b")

TOKEN_RE = re.compile(r"[a-z0-9]+(?:[./_\-][a-z0-9]+)*")
SEP_RE = re.compile(r"[./_\-]")


# ----------------------------------------------------------------------
# Tokenizer for BM25 (hybrid.py must import and use this same function)
# ----------------------------------------------------------------------
STOPWORDS = frozenset(
    "a an the and or of to in on at for by with from as is are was were be been "
    "this that these those it its what which who how when where why per".split()
)


def tokenize(text: str) -> list[str]:
    """
    Lowercase word tokenizer that keeps compound codes (e.g. 'dbr.aml.bc.no.81')
    as one token AND also emits their parts, so exact-code queries match and
    partial queries still work.
    """
    tokens: list[str] = []
    for m in TOKEN_RE.findall(normalize_numbers(text.lower())):
        tokens.append(m)
        if SEP_RE.search(m):
            tokens.extend(SEP_RE.split(m))
    return [t for t in tokens if t not in STOPWORDS]


# ----------------------------------------------------------------------
# Data holders
# ----------------------------------------------------------------------
Page = tuple[int | None, str]  # (page number or None for non-PDF, text)


@dataclass
class Section:
    heading: str | None
    lines: list[str] = field(default_factory=list)
    page_offsets: list[tuple[int, int | None]] = field(default_factory=list)
    line_pages: list[int | None] = field(default_factory=list)
    length: int = 0

    def add_line(self, line: str, page: int | None) -> None:
        if not self.page_offsets or self.page_offsets[-1][1] != page:
            self.page_offsets.append((self.length, page))
        self.lines.append(line)
        self.line_pages.append(page)
        self.length += len(line) + 1

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    def page_at(self, offset: int) -> int | None:
        if not self.page_offsets:
            return None
        starts = [o for o, _ in self.page_offsets]
        i = bisect.bisect_right(starts, offset) - 1
        return self.page_offsets[max(i, 0)][1]


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_manifest() -> dict[str, dict]:
    path = settings.manifest_path
    if not path.exists():
        logger.warning("No manifest at %s; relying on regex metadata only.", path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("documents", data)


def discover_files() -> list[tuple[Path, SourceType]]:
    found: list[tuple[Path, SourceType]] = []
    for directory, stype in (
        (settings.raw_rbi_dir, SourceType.RBI),
        (settings.raw_fictional_dir, SourceType.FICTIONAL_BANK),
        (settings.uploads_rbi_dir, SourceType.RBI),
        (settings.uploads_fictional_dir, SourceType.FICTIONAL_BANK),
    ):
        if not directory.exists():
            logger.warning("Missing directory: %s", directory)
            continue
        for p in sorted(directory.iterdir()):
            if p.suffix.lower() in SUPPORTED_EXTS:
                found.append((p, stype))
    return found


def extract_pages(path: Path) -> list[Page]:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        pages: list[Page] = []
        for i, page in enumerate(reader.pages, start=1):
            try:
                pages.append((i, page.extract_text() or ""))
            except Exception as exc:  # corrupt page: skip, don't abort the whole run
                logger.warning("%s page %d: extraction failed (%s)", path.name, i, exc)
                pages.append((i, ""))
        if sum(len(t.strip()) for _, t in pages) < 200:
            logger.warning(
                "%s has almost no extractable text (scanned PDF?). OCR is not handled here.",
                path.name,
            )
        return pages
    text = path.read_text(encoding="utf-8", errors="ignore")
    return [(None, normalize_markdown(text))]


# ----------------------------------------------------------------------
# Cleaning
# ----------------------------------------------------------------------
def _norm_line(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip().lower()))


def clean_pages(pages: list[Page]) -> list[Page]:
    """Drop page numbers and repeated headers/footers; fix hyphenated line breaks."""
    n_pages = len(pages)
    repeated: set[str] = set()
    if n_pages >= 4:
        counts: Counter[str] = Counter()
        for _, text in pages:
            seen = {_norm_line(l) for l in text.splitlines() if 0 < len(l.strip()) <= 120}
            counts.update(seen)
        repeated = {k for k, c in counts.items() if c / n_pages >= 0.5}

    cleaned: list[Page] = []
    for page_no, text in pages:
        text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate
        text = FOOTNOTE_GLUE_RE.sub("", text)  # "117Aadhaar" -> "Aadhaar" (helps BM25 matching)
        kept = []
        for line in text.splitlines():
            s = line.rstrip()
            if PAGE_NUMBER_RE.match(s) or TOC_LINE_RE.search(s):
                continue
            if s.strip() and _norm_line(s) in repeated:
                continue
            kept.append(re.sub(r"[ \t]+", " ", s))
        cleaned.append((page_no, "\n".join(kept)))
    return cleaned


# ----------------------------------------------------------------------
# Sectioning
# ----------------------------------------------------------------------
IGNORE_HEADINGS = {"RESERVE BANK OF INDIA", "UNIFIED", "PAYMENTS", "INTERFACE"}


def is_heading(line: str) -> bool:
    s = line.strip()
    return (
        bool(s)
        and len(s) <= 100
        and s not in IGNORE_HEADINGS
        and any(r.match(s) for r in HEADING_RES)
    )


def split_sections(pages: list[Page]) -> list[Section]:
    flat: list[tuple[int | None, str]] = [
        (page_no, line) for page_no, text in pages for line in text.splitlines()
    ]
    sections: list[Section] = [Section(heading=None)]
    for i, (page_no, line) in enumerate(flat):
        if not line.strip():
            sections[-1].add_line("", page_no)
            continue
        if is_heading(line):
            heading = line.strip().lstrip("# ").strip()
            heading = re.sub(r"[*\\]", "", heading).strip()  # "**2\\. Title**" -> "2. Title"
            if CHAPTER_ALONE_RE.match(heading):
                # "CHAPTER III" alone on a line: take the title from the next non-empty line.
                nxt = next((l.strip() for _, l in flat[i + 1 : i + 4] if l.strip()), "")
                if looks_like_title(nxt):
                    heading = f"{heading} - {nxt}"
            sections.append(Section(heading=heading[:100]))
        sections[-1].add_line(line.strip(), page_no)

    # Merge tiny sections into the following one (keeps heading-only stubs out of the index).
    merged: list[Section] = []
    carry: Section | None = None
    for sec in sections:
        if carry is not None:
            for ln, pg in zip(sec.lines, sec.line_pages):
                carry.add_line(ln, pg)  # keep each line's own page number
            carry.heading = carry.heading or sec.heading  # merged text starts with carry's heading
            sec = carry
            carry = None
        if len(sec.text.strip()) < MIN_SECTION_CHARS and sec is not sections[-1]:
            carry = sec
        else:
            merged.append(sec)
    if carry is not None:
        merged.append(carry)
    return [s for s in merged if s.text.strip()]


# ----------------------------------------------------------------------
# Metadata extraction
# ----------------------------------------------------------------------
def _to_iso(day: str, month: str, year: str) -> str | None:
    try:
        m = datetime.strptime(month[:3].title(), "%b").month
        d = datetime(int(year), m, int(day))
    except ValueError:
        return None
    return d.date().isoformat() if 1990 <= d.year <= 2100 else None


def extract_date(text: str) -> str | None:
    """First plausible date in the text, as ISO. May be an issue date OR an amendment date."""
    candidates: list[tuple[int, str]] = []
    for m in DATE_MDY_RE.finditer(text):
        iso = _to_iso(m.group(2), m.group(1), m.group(3))
        if iso:
            candidates.append((m.start(), iso))
    for m in DATE_DMY_RE.finditer(text):
        iso = _to_iso(m.group(1), m.group(2), m.group(3))
        if iso:
            candidates.append((m.start(), iso))
    return min(candidates)[1] if candidates else None


def doc_metadata(
    path: Path, stype: SourceType, pages: list[Page], manifest: dict[str, dict]
) -> dict:
    entry = manifest.get(path.name, {})
    head = "\n".join(t for _, t in pages[:3])
    circular = entry.get("circular_no")
    if not circular and stype == SourceType.RBI:
        m = RBI_NO_RE.search(head)
        circular = m.group(0) if m else None
    date = entry.get("date") or extract_date(head)
    if stype == SourceType.RBI and not circular:
        logger.warning("%s: no circular_no found. Add it to the manifest.", path.name)
    return {
        "doc": entry.get("title") or path.stem.replace("_", " "),
        "source_type": stype,
        "circular_no": circular,
        "date": date,
        "url": entry.get("url"),
    }


# ----------------------------------------------------------------------
# Chunking
# ----------------------------------------------------------------------
def _load_tokenizer():
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(settings.embedding_model)
    tok.model_max_length = 10**6  # silence "longer than max length" warnings; we check sizes ourselves
    return tok


def _make_splitter(tokenizer):
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    return RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer,
        chunk_size=settings.chunk_size_tokens,
        chunk_overlap=settings.chunk_overlap_tokens,
    )


def _slug(path: Path) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", path.stem.lower()).strip("-")[:40]
    return f"{base}-{hashlib.md5(path.name.encode()).hexdigest()[:6]}"


def build_chunks() -> list[Chunk]:
    manifest = load_manifest()
    files = discover_files()
    if not files:
        raise SystemExit(f"No documents found under {settings.raw_dir}")

    tokenizer = _load_tokenizer()
    splitter = _make_splitter(tokenizer)
    chunks: list[Chunk] = []

    for path, stype in files:
        pages = clean_pages(extract_pages(path))
        meta = doc_metadata(path, stype, pages, manifest)
        slug, n = _slug(path), 0
        seen_texts: set[str] = set()
        for sec in split_sections(pages):
            sec_text = sec.text
            for piece in splitter.split_text(sec_text):
                piece = piece.strip()
                if not piece or is_junk_chunk(piece):
                    continue
                key = re.sub(r"\s+", " ", piece)
                if key in seen_texts:  # identical chunk already added for this document
                    continue
                seen_texts.add(key)
                idx = sec_text.find(piece[:60])  # approximate page lookup
                page = sec.page_at(idx) if idx >= 0 else sec.page_offsets[0][1]
                chunks.append(
                    Chunk(
                        chunk_id=f"{slug}-{n:04d}",
                        text=piece,
                        metadata=ChunkMetadata(section=sec.heading, page=page, **meta),
                    )
                )
                n += 1
        logger.info("%s -> %d chunks", path.name, n)

    # Sanity check: no chunk should exceed the embedding model's limit.
    lengths = [len(tokenizer.encode(contextualize(c))) for c in chunks]  # includes the context header
    too_long = sum(1 for l in lengths if l > settings.embedding_max_tokens)
    logger.info(
        "Total %d chunks | tokens min/avg/max = %d/%d/%d | over model limit: %d",
        len(chunks), min(lengths), sum(lengths) // len(lengths), max(lengths), too_long,
    )
    if too_long:
        logger.warning("%d chunks exceed %d tokens and will be truncated by the embedder.",
                       too_long, settings.embedding_max_tokens)
    return chunks


# ----------------------------------------------------------------------
# Persistence and indexes
# ----------------------------------------------------------------------
def save_chunks(chunks: list[Chunk]) -> None:
    with settings.chunks_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(c.model_dump_json() + "\n")


def load_chunks() -> list[Chunk]:
    """Used by retrieval/hybrid.py."""
    with settings.chunks_path.open(encoding="utf-8") as f:
        return [Chunk.model_validate_json(line) for line in f if line.strip()]


def build_bm25(chunks: list[Chunk]) -> None:
    from rank_bm25 import BM25Okapi

    bm25 = BM25Okapi([tokenize(contextualize(c)) for c in chunks])
    with settings.bm25_path.open("wb") as f:
        pickle.dump({"bm25": bm25, "chunk_ids": [c.chunk_id for c in chunks]}, f)
    logger.info("BM25 index saved to %s", settings.bm25_path)


def build_chroma(chunks: list[Chunk]) -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)
    # Passages are embedded WITHOUT the query instruction prefix.
    embeddings = model.encode(
        [contextualize(c) for c in chunks],  # header + text; Chroma still stores raw c.text
        batch_size=32,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).tolist()

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    try:
        client.delete_collection(settings.chroma_collection)  # full rebuild
    except Exception:
        pass
    collection = client.create_collection(
        settings.chroma_collection, metadata={"hnsw:space": "cosine"}
    )

    def _clean_meta(c: Chunk) -> dict:
        raw = c.metadata.model_dump(mode="json")
        return {k: v for k, v in raw.items() if v is not None}  # Chroma rejects None values

    batch = 500
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        collection.add(
            ids=[c.chunk_id for c in part],
            documents=[c.text for c in part],
            embeddings=embeddings[i : i + batch],
            metadatas=[_clean_meta(c) for c in part],
        )
    logger.info("Chroma collection '%s' built with %d chunks", settings.chroma_collection, len(chunks))


def build_index() -> None:
    settings.ensure_dirs()
    chunks = build_chunks()
    save_chunks(chunks)
    build_bm25(chunks)
    build_chroma(chunks)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    build_index()
