"""Make sure the fine-tuned Tesseract model is the real file, and verify its SHA-256.

The model (15 MB) is stored with Git LFS. A clone made without git-lfs only has a
small pointer file in its place, so this downloads the real one. `make ocr-model`
rebuilds it from scratch instead.

Usage: python scripts/fetch_model.py [--force]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models" / "tessdata" / "eng_tt.traineddata"
URL = "https://media.githubusercontent.com/media/tanaymihani/tabletrace/main/models/tessdata/eng_tt.traineddata"
SHA256 = "9f46490a8ead4944e41e20184303bbacad3db7dd96d7bd7143d95c0647418d82"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-download even if a verified copy exists")
    args = ap.parse_args()

    if DEST.exists() and not args.force and sha256(DEST) == SHA256:
        print(f"ok      {DEST.relative_to(ROOT)}")
        return 0
    print(f"fetch   {URL}")
    r = requests.get(URL, timeout=300)
    r.raise_for_status()
    DEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = DEST.with_suffix(".part")
    tmp.write_bytes(r.content)
    digest = sha256(tmp)
    if digest != SHA256:
        tmp.unlink()
        print(f"MISMATCH: got {digest}, expected {SHA256}")
        return 1
    tmp.replace(DEST)
    print(f"ok      {DEST.relative_to(ROOT)} (verified)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
