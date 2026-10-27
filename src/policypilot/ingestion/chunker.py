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
    if not words:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(words):
        window = words[start : start + chunk_size]
        chunks.append(" ".join(window))
        if start + chunk_size >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def _extract_sections(text: str) -> list[tuple[str, str]]:
    """Return list of (section_label, section_text) pairs.

    If no Item headers are found, returns [("", text)] so chunking still works.
    """
    matches = list(SECTION_RE.finditer(text))
    if not matches:
        return [("", text)]
    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        label = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end].strip()
        if section_text:
            sections.append((label, section_text))
    return sections if sections else [("", text)]


def chunk_filing(ticker: str, filing: dict, text: str) -> list[Chunk]:
    company = filing["company"]
    filing_date = filing["filing_date"]
    accession_number = filing["accession_number"]
    sections = _extract_sections(text)
    chunks: list[Chunk] = []
    global_index = 0
    for section_label, section_text in sections:
        for raw in _split_text(section_text):
            chunks.append(
                Chunk(
                    id=f"{ticker}-{accession_number}-{global_index}",
                    text=raw,
                    ticker=ticker,
                    company=company,
                    filing_date=filing_date,
                    accession_number=accession_number,
                    chunk_index=global_index,
                    metadata={"section": section_label},
                )
            )
            global_index += 1
    return chunks
