"""
Download NCERT textbook PDFs for Classes 1-12.

The script saves PDFs under:
    Blindterminal/data/textbooks/ncert/

It discovers book codes from NCERT's official textbook page/assets at runtime
instead of relying on one frozen list. If NCERT changes the page structure, pass
an explicit manifest with --manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = BASE_DIR / "data" / "textbooks" / "ncert"

NCERT_BASE_URL = "https://ncert.nic.in"
NCERT_TEXTBOOK_PAGE = f"{NCERT_BASE_URL}/textbook.php?ln=en"
NCERT_PDF_DIR = f"{NCERT_BASE_URL}/textbook/pdf"

USER_AGENT = (
    "BlindAssist NCERT Downloader/1.0 "
    "(educational offline-access script; contact: local user)"
)

# NCERT textbook codes use first letter as class marker:
# a=1, b=2, ..., l=12. Examples: jesc1 = Class 10 Science,
# leph1 = Class 12 Physics Part 1.
CLASS_CODE_TO_STANDARD = {chr(ord("a") + i): i + 1 for i in range(12)}
BOOK_CODE_RE = re.compile(r"\b([a-l][a-z]{3}\d)\b", re.IGNORECASE)
PDF_RE = re.compile(
    r"(?:https?://ncert\.nic\.in)?/?textbook/pdf/([a-l][a-z]{3}\d)(?:ps|\d{2})?\.pdf",
    re.IGNORECASE,
)
ASSET_RE = re.compile(
    r"""(?:src|href)\s*=\s*["']([^"']+\.(?:js|json|php)(?:\?[^"']*)?)["']""",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Book:
    code: str
    standard: int
    title: str = ""
    subject: str = ""
    language: str = ""

    @property
    def url(self) -> str:
        return f"{NCERT_PDF_DIR}/{self.code}ps.pdf"

    @property
    def filename(self) -> str:
        suffix = slugify(self.title) if self.title else self.code
        return f"{self.code}_{suffix}.pdf"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "book"


def request_bytes(url: str, timeout: int = 30) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def request_text(url: str, timeout: int = 30) -> str:
    return request_bytes(url, timeout=timeout).decode("utf-8", errors="replace")


def absolute_url(url: str, base: str = NCERT_TEXTBOOK_PAGE) -> str:
    return urllib.parse.urljoin(base, url)


def standard_from_code(code: str) -> int | None:
    return CLASS_CODE_TO_STANDARD.get(code[0].lower())


def build_book(code: str, title: str = "", subject: str = "", language: str = "") -> Book | None:
    code = code.lower()
    standard = standard_from_code(code)
    if standard is None or not 1 <= standard <= 12:
        return None
    return Book(
        code=code,
        standard=standard,
        title=title.strip(),
        subject=subject.strip(),
        language=language.strip(),
    )


def books_from_text(text: str) -> dict[str, Book]:
    books: dict[str, Book] = {}

    for match in PDF_RE.finditer(text):
        book = build_book(match.group(1))
        if book:
            books.setdefault(book.code, book)

    for match in BOOK_CODE_RE.finditer(text):
        book = build_book(match.group(1))
        if book:
            books.setdefault(book.code, book)

    return books


def discover_asset_urls(page_html: str) -> list[str]:
    urls = []
    seen = set()
    for match in ASSET_RE.finditer(page_html):
        url = absolute_url(match.group(1))
        if urllib.parse.urlparse(url).netloc.endswith("ncert.nic.in") and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def discover_from_ncert() -> dict[str, Book]:
    print(f"Fetching NCERT textbook page: {NCERT_TEXTBOOK_PAGE}")
    page_html = request_text(NCERT_TEXTBOOK_PAGE)
    books = books_from_text(page_html)

    for asset_url in discover_asset_urls(page_html):
        try:
            print(f"Scanning asset: {asset_url}")
            asset_text = request_text(asset_url, timeout=20)
        except Exception as exc:
            print(f"  warning: could not read asset: {exc}")
            continue

        for code, book in books_from_text(asset_text).items():
            books.setdefault(code, book)

    return books


def load_manifest(path: Path) -> dict[str, Book]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Manifest must be a JSON list of book objects.")

    books: dict[str, Book] = {}
    for item in raw:
        if not isinstance(item, dict) or "code" not in item:
            raise ValueError("Each manifest item must be an object with a 'code'.")
        book = build_book(
            str(item["code"]),
            title=str(item.get("title", "")),
            subject=str(item.get("subject", "")),
            language=str(item.get("language", "")),
        )
        if book:
            books[book.code] = book

    return books


def filter_books(books: Iterable[Book], standards: set[int]) -> list[Book]:
    filtered = [book for book in books if book.standard in standards]
    return sorted(filtered, key=lambda book: (book.standard, book.code))


def output_path_for(book: Book, output_dir: Path) -> Path:
    class_dir = output_dir / f"class_{book.standard:02d}"
    if book.subject:
        class_dir = class_dir / slugify(book.subject)
    return class_dir / book.filename


def download_book(book: Book, output_dir: Path, force: bool = False) -> tuple[bool, str]:
    path = output_path_for(book, output_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and path.stat().st_size > 0 and not force:
        return True, "exists"

    temp_path = path.with_suffix(path.suffix + ".part")
    try:
        data = request_bytes(book.url, timeout=90)
        if not data.startswith(b"%PDF"):
            return False, "not a PDF response"

        temp_path.write_bytes(data)
        temp_path.replace(path)
        return True, f"downloaded {len(data) / (1024 * 1024):.1f} MB"
    except urllib.error.HTTPError as exc:
        return False, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(exc)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def parse_standards(raw: str) -> set[int]:
    raw = raw.strip().lower()
    if raw in {"all", "1-12"}:
        return set(range(1, 13))

    standards: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start, end = int(start_text), int(end_text)
            standards.update(range(start, end + 1))
        else:
            standards.add(int(part))

    invalid = sorted(value for value in standards if value < 1 or value > 12)
    if invalid:
        raise argparse.ArgumentTypeError(f"Invalid standards: {invalid}")
    return standards


def write_manifest(books: Iterable[Book], path: Path) -> None:
    payload = [
        {
            "code": book.code,
            "standard": book.standard,
            "title": book.title,
            "subject": book.subject,
            "language": book.language,
            "url": book.url,
        }
        for book in books
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download official NCERT textbook PDFs for Classes 1-12."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Download folder. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--standards",
        type=parse_standards,
        default=set(range(1, 13)),
        help="Classes to download, e.g. 'all', '1-12', '10', or '9,10,12'.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON manifest with NCERT book codes to download.",
    )
    parser.add_argument(
        "--write-manifest",
        type=Path,
        help="Write discovered book metadata to this JSON file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print what would be downloaded.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.4,
        help="Seconds to wait between downloads. Default: 0.4",
    )
    args = parser.parse_args()

    try:
        books_by_code = load_manifest(args.manifest) if args.manifest else discover_from_ncert()
    except Exception as exc:
        print(f"Could not discover NCERT books: {exc}", file=sys.stderr)
        print(
            "Try again with internet access, or pass --manifest with NCERT book codes.",
            file=sys.stderr,
        )
        return 2

    books = filter_books(books_by_code.values(), args.standards)
    if not books:
        print("No NCERT book codes were discovered for the requested classes.", file=sys.stderr)
        return 2

    if args.write_manifest:
        write_manifest(books, args.write_manifest)
        print(f"Wrote manifest: {args.write_manifest}")

    print(f"Found {len(books)} NCERT book(s) for Classes {sorted(args.standards)}")
    print(f"Output directory: {args.output_dir}")

    if args.dry_run:
        for book in books:
            print(f"[dry-run] Class {book.standard:02d} {book.code}: {book.url}")
        return 0

    ok_count = 0
    failed_count = 0
    for index, book in enumerate(books, start=1):
        path = output_path_for(book, args.output_dir)
        print(f"[{index}/{len(books)}] Class {book.standard:02d} {book.code} -> {path.name}")
        ok, message = download_book(book, args.output_dir, force=args.force)
        if ok:
            ok_count += 1
            print(f"  ok: {message}")
        else:
            failed_count += 1
            print(f"  failed: {message}")

        if args.delay > 0:
            time.sleep(args.delay)

    print("\nDownload summary")
    print(f"  Success: {ok_count}")
    print(f"  Failed:  {failed_count}")
    print(f"  Folder:  {args.output_dir}")
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
