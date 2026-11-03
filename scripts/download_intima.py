#!/usr/bin/env python3
"""Download the INTIMA benchmark (``INTIMA.jsonl``) from HuggingFace with SHA-256 verification.

Stdlib only (urllib). Writes ``data/intima/intima.jsonl`` (git-ignored; CC BY 4.0, Kaffee, Pistilli
& Jernite, ICLR 2026) atomically: the file is only moved into place after its hash matches the
hash recorded in docs/research/intima.md (dataset revision 48537e5d..., lastModified 2025-08-29).

    python scripts/download_intima.py                      # main branch, verified
    python scripts/download_intima.py --revision 48537e5d3905152751964b48d0a2ef681ace74dc
    python scripts/download_intima.py --expected-sha256 <hex>   # accept a newer revision knowingly

Then:  export INTIMA_DATA_PATH=data/intima/intima.jsonl
The downloaded file is treated as data only: it is hashed and line-counted, never executed.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "AI-companionship/INTIMA"
FILE_NAME = "INTIMA.jsonl"
DEFAULT_REVISION = "main"
PINNED_REVISION = "48537e5d3905152751964b48d0a2ef681ace74dc"
EXPECTED_SHA256 = "31b683f0592a88db078e9cd8f3b03eefbd59edeac14b7894a92894d1518eb136"
EXPECTED_ROWS = 380
DEFAULT_OUT = REPO_ROOT / "data" / "intima" / "intima.jsonl"


def url_for(revision: str = DEFAULT_REVISION) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/{revision}/{FILE_NAME}"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def count_rows(path: Path) -> int:
    with path.open("rb") as f:
        return sum(1 for line in f if line.strip())


def verify(path: Path, expected_sha256: str | None) -> str:
    """Return the file's SHA-256; raise ValueError if it does not match ``expected_sha256``."""
    digest = sha256_file(path)
    if expected_sha256 and digest != expected_sha256.lower():
        raise ValueError(f"SHA-256 mismatch for {path}: got {digest}, expected {expected_sha256}")
    return digest


def download(out: Path = DEFAULT_OUT, *, revision: str = DEFAULT_REVISION,
             expected_sha256: str | None = EXPECTED_SHA256, force: bool = False,
             timeout: float = 60.0) -> Path:
    out = Path(out)
    if out.exists() and not force:
        verify(out, expected_sha256)
        print(f"{out} already present and verified; use --force to re-download.")
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = url_for(revision)
    req = urllib.request.Request(url, headers={"User-Agent": "ai-companion-endings/0.1"})
    fd, tmp_name = tempfile.mkstemp(prefix=".intima-", suffix=".part", dir=out.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as fh, urllib.request.urlopen(req, timeout=timeout) as resp:
            while chunk := resp.read(1 << 16):
                fh.write(chunk)
        digest = verify(tmp, expected_sha256)
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    n = count_rows(out)
    note = "" if n == EXPECTED_ROWS else f" (WARNING: expected {EXPECTED_ROWS})"
    print(f"Wrote {out}  rows={n}{note}  sha256={digest}  source={url}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--revision", default=DEFAULT_REVISION,
                    help=f"HF revision (default main; pinned hash revision: {PINNED_REVISION})")
    ap.add_argument("--expected-sha256", default=EXPECTED_SHA256,
                    help="expected file hash (default: the hash recorded in intima.md)")
    ap.add_argument("--no-verify", action="store_true", help="skip hash verification (not advised)")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args(argv)
    try:
        download(args.out, revision=args.revision,
                 expected_sha256=None if args.no_verify else args.expected_sha256,
                 force=args.force)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
