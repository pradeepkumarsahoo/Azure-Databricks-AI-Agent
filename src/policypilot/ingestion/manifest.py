# Author: Pradeep Sahoo
"""Tracks which filings have been ingested. Backs the agent's structured
"filing metadata" tool (agent/tools.py) — the retriever handles unstructured text,
this handles the "when was X filed" kind of question a UC function would answer in prod.
"""

from __future__ import annotations

import json
import os
from typing import Mapping

from policypilot.config import DATA_DIR

MANIFEST_PATH = DATA_DIR / "manifest.json"


def load_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def _entry(filing: dict) -> dict:
    return {
        "company": filing["company"],
        "filing_date": filing["filing_date"],
        "accession_number": filing["accession_number"],
        "form": "10-K",
    }


def _write_manifest(manifest: dict) -> None:
    """Atomically persist the manifest so a crash can't leave a half-written file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = MANIFEST_PATH.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(manifest, indent=2))
    os.replace(tmp_path, MANIFEST_PATH)


def record_filing(ticker: str, filing: dict) -> None:
    manifest = load_manifest()
    manifest[ticker] = _entry(filing)
    _write_manifest(manifest)


def record_filings(filings_by_ticker: Mapping[str, dict]) -> None:
    """Record many filings in one read/write cycle — avoids the N-reads/N-writes
    pattern when ingesting a batch of tickers."""
    if not filings_by_ticker:
        return
    manifest = load_manifest()
    for ticker, filing in filings_by_ticker.items():
        manifest[ticker] = _entry(filing)
    _write_manifest(manifest)
