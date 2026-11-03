# Author: Pradeep Sahoo
"""Splits filing text into overlapping, citation-friendly chunks with section metadata."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

SECTION_RE = re.compile(
    r"(?:^|\n)((?:ITEM|Item)\s+\d+[A-Z]?\s*[.\-–—]\s*[^\n]+)",
    re.MULTILINE,
)


@dataclass
class Chunk:
    id: str
    text: str
    ticker: str
    company: str
    filing_date: str
    accession_number: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    words = text.split()
    n = len(words)
    if not n:
        return []
    step = chunk_size - overlap
    chunks: list[str] = []
    start = 0
    while start < n:
        chunks.append(" ".join(words[start : start + chunk_size]))
        if start + chunk_size >= n:
            break
        start += step
    return chunks


def _extract_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_label, section_text) pairs.

    If no Item headers are found, returns [("", text)] so chunking still works.
    """
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    text_len = len(text)
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else text_len
        section_text = text[start:end].strip()
        if section_text:
            sections.append((m.group(1).strip(), section_text))
    return sections or [("", text)]


def chunk_filing(ticker: str, filing: dict, text: str) -> list[Chunk]:
    company = filing["company"]
    filing_date = filing["filing_date"]
    accession_number = filing["accession_number"]
    chunks: list[Chunk] = []
    for section_label, section_text in _extract_sections(text):
        for raw in _split_text(section_text):
            idx = len(chunks)
            chunks.append(
                Chunk(
                    id=f"{ticker}-{accession_number}-{idx}",
                    text=raw,
                    ticker=ticker,
                    company=company,
                    filing_date=filing_date,
                    accession_number=accession_number,
                    chunk_index=idx,
                    metadata={"section": section_label},
                )
            )
    return chunks
