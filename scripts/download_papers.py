"""Fetch every document in papers.csv and verify its SHA-256.

The PDFs are not committed because licenses differ between sources. Journal
articles come from the LitXBench repository pinned to one commit, because MDPI
blocks scripted downloads and a pinned copy keeps page numbers stable. NASA
reports come from the NTRS API and the patent from Google's patent image store.

Usage: python scripts/download_papers.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tabletrace.paths import RAW_PDFS, load_papers  # noqa: E402

HEADERS = {"User-Agent": "tabletrace/0.1 (research; table extraction benchmark)"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, retries: int = 3) -> None:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=120)
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError(f"not a PDF (content-type {r.headers.get('content-type')})")
            dest.write_bytes(r.content)
            return
        except Exception as exc:  # network errors, HTTP errors, bad payloads
            if attempt == retries:
                raise
            print(f"  retry {attempt}: {exc}")
            time.sleep(2 * attempt)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if a verified copy exists")
    args = ap.parse_args()

    RAW_PDFS.mkdir(parents=True, exist_ok=True)
    failures = 0
    for paper in load_papers():
        dest = paper.raw_pdf
        if dest.exists() and not args.force and sha256(dest) == paper.sha256:
            print(f"ok      {paper.paper_id}")
            continue
        print(f"fetch   {paper.paper_id} <- {paper.url}")
        try:
            fetch(paper.url, dest)
        except Exception as exc:
            print(f"FAILED  {paper.paper_id}: {exc}")
            failures += 1
            continue
        digest = sha256(dest)
        if digest != paper.sha256:
            print(f"MISMATCH {paper.paper_id}: got {digest}, expected {paper.sha256}")
            failures += 1
        else:
            print(f"ok      {paper.paper_id} (verified)")
    print(f"{len(load_papers()) - failures} documents ready in {RAW_PDFS}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
