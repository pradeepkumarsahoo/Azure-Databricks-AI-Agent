# Author: Pradeep Sahoo
"""Thin client for SEC EDGAR's public JSON/HTML APIs. No auth required, but SEC's
fair-access policy requires a descriptive User-Agent with a real contact on every request.
"""
from __future__ import annotations

import time

import requests
from bs4 import BeautifulSoup
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"

_RETRYABLE = (requests.HTTPError, requests.ConnectionError, requests.Timeout)
_STRIPPED_TAGS = ("script", "style")


class EdgarClient:
    def __init__(self, user_agent: str):
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": user_agent})
        self._ticker_cik_map: dict[str, int] | None = None

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _get(self, url: str) -> requests.Response:
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        time.sleep(0.15)  # stay well under SEC's request-rate guidance
        return resp

    def _load_ticker_cik_map(self) -> dict[str, int]:
        """Fetch SEC's full ticker→CIK table once and reuse it for subsequent lookups.

        The upstream JSON is ~1 MB and the same for every ticker, so hitting it once
        per client (not once per ticker) is a strict win on both latency and SEC quota.
        """
        if self._ticker_cik_map is None:
            data = self._get(TICKERS_URL).json()
            self._ticker_cik_map = {
                entry["ticker"].upper(): int(entry["cik_str"]) for entry in data.values()
            }
        return self._ticker_cik_map

    def get_cik_for_ticker(self, ticker: str) -> int:
        mapping = self._load_ticker_cik_map()
        try:
            return mapping[ticker.upper()]
        except KeyError:
            raise ValueError(f"No CIK found for ticker {ticker!r}") from None

    def get_recent_10k(self, cik: int) -> dict:
        """Return metadata for the most recent 10-K filing for a company."""
        data = self._get(SUBMISSIONS_URL.format(cik=cik)).json()
        recent = data["filings"]["recent"]
        forms = recent["form"]
        for i, form in enumerate(forms):
            if form == "10-K":
                return {
                    "cik": cik,
                    "accession_number": recent["accessionNumber"][i],
                    "primary_document": recent["primaryDocument"][i],
                    "filing_date": recent["filingDate"][i],
                    "company": data.get("name", str(cik)),
                }
        raise ValueError(f"No 10-K found in recent filings for CIK {cik}")

    def download_filing_text(self, filing: dict) -> str:
        """Download a filing's primary document and return its extracted plain text."""
        accession_nodash = filing["accession_number"].replace("-", "")
        url = ARCHIVES_URL.format(
            cik=filing["cik"],
            accession_nodash=accession_nodash,
            document=filing["primary_document"],
        )
        html = self._get(url).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(_STRIPPED_TAGS):
            tag.decompose()
        text = soup.get_text(separator="\n")
        return "\n".join(filter(None, (line.strip() for line in text.splitlines())))
