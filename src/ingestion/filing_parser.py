"""
Filing parser: strips HTML boilerplate, chunks by Item headers,
detects template/boilerplate content.
"""
import re
import hashlib
from pathlib import Path
from bs4 import BeautifulSoup
from difflib import SequenceMatcher


# Common SEC boilerplate patterns to strip
_BOILERPLATE_PATTERNS = [
    r"UNITED STATES\s+SECURITIES AND EXCHANGE COMMISSION",
    r"FORM 8-K\s+CURRENT REPORT",
    r"Pursuant to Section 13 or 15\(d\)",
    r"Commission [Ff]ile [Nn]umber",
    r"SIGNATURES?\s+Pursuant to the requirements",
    r"Exhibit \d+",
]
_BOILERPLATE_RE = re.compile("|".join(_BOILERPLATE_PATTERNS), re.IGNORECASE)

# Item header pattern for chunking
_ITEM_RE = re.compile(
    r"(?:Item|ITEM)\s+(\d+\.\d+)\s*[.:\-—]?\s*(.*?)(?=\n|$)",
    re.IGNORECASE
)

# Template fingerprint bank (built during processing)
_template_hashes: set[str] = set()


def strip_html(raw_html: str) -> str:
    """Remove HTML tags and normalize whitespace."""
    soup = BeautifulSoup(raw_html, "lxml")

    # Remove script/style tags
    for tag in soup.find_all(["script", "style", "xbrl", "ix:nonfraction"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Normalize whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)

    return text.strip()


def _text_hash(text: str) -> str:
    """Fingerprint text for boilerplate detection."""
    # Normalize: lowercase, strip whitespace, take first 500 chars
    normalized = re.sub(r"\s+", "", text.lower())[:500]
    return hashlib.md5(normalized.encode()).hexdigest()


def detect_boilerplate(text: str) -> bool:
    """Check if text is likely boilerplate/template content."""
    if len(text.strip()) < 50:
        return True

    # Check against known SEC boilerplate
    boilerplate_hits = len(_BOILERPLATE_RE.findall(text))
    total_lines = max(1, text.count("\n") + 1)
    if boilerplate_hits / total_lines > 0.3:
        return True

    # Check against seen templates
    h = _text_hash(text)
    if h in _template_hashes:
        return True
    _template_hashes.add(h)

    return False


def chunk_filing(clean_text: str, accession: str, cik: int,
                 ticker: str, filed_at: str) -> list[dict]:
    """
    Split a cleaned filing into chunks by Item header.
    If no Item headers found, treat the whole filing as one chunk.
    """
    chunks = []

    # Find all Item headers
    items = list(_ITEM_RE.finditer(clean_text))

    if not items:
        # No item headers; single chunk
        text = clean_text.strip()
        if len(text) > 30:
            chunks.append({
                "chunk_id": f"{accession}_full",
                "accession": accession,
                "cik": cik,
                "ticker": ticker,
                "filed_at": filed_at,
                "item_type": "unknown",
                "clean_text": text,
                "is_boilerplate": detect_boilerplate(text),
                "char_count": len(text),
            })
        return chunks

    # Extract text between consecutive Item headers
    for i, match in enumerate(items):
        item_code = match.group(1)
        start = match.start()
        end = items[i + 1].start() if i + 1 < len(items) else len(clean_text)

        text = clean_text[start:end].strip()
        if len(text) < 30:
            continue

        # Remove the SEC boilerplate header portion
        lines = text.split("\n")
        content_lines = []
        for line in lines:
            if not _BOILERPLATE_RE.search(line):
                content_lines.append(line)
        text = "\n".join(content_lines).strip()

        if len(text) < 30:
            continue

        chunks.append({
            "chunk_id": f"{accession}_item{item_code}",
            "accession": accession,
            "cik": cik,
            "ticker": ticker,
            "filed_at": filed_at,
            "item_type": item_code,
            "clean_text": text,
            "is_boilerplate": detect_boilerplate(text),
            "char_count": len(text),
        })

    return chunks


def parse_filing(raw_path: str, accession: str, cik: int,
                 ticker: str, filed_at: str) -> list[dict]:
    """
    Full parse pipeline: read HTML, strip, chunk, detect boilerplate.
    """
    path = Path(raw_path)
    if not path.exists():
        return []

    raw_html = path.read_text(encoding="utf-8", errors="ignore")
    clean = strip_html(raw_html)
    chunks = chunk_filing(clean, accession, cik, ticker, filed_at)

    return chunks
