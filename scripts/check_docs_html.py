"""Validate that every docs/*.html file is well-formed enough to parse.

Used by .github/workflows/frontend-ci.yml as the "frontend build" check for
the static GitHub Pages site (docs/) -- there is no bundler/build step, so a
successful HTML parse plus a non-empty JS syntax check stands in for one.
"""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path

DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"


def main() -> int:
    failed = False
    html_files = sorted(DOCS_DIR.glob("*.html"))
    if not html_files:
        print(f"::error::No HTML files found under {DOCS_DIR}")
        return 1

    for html_file in html_files:
        content = html_file.read_text(encoding="utf-8")
        if not content.strip():
            print(f"::error file={html_file}::File is empty")
            failed = True
            continue
        try:
            HTMLParser().feed(content)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"::error file={html_file}::{exc}")
            failed = True
        else:
            print(f"OK: {html_file}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
