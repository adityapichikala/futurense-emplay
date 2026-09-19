"""Text normalization: unicode repair, whitespace collapse, boilerplate stripping.

PDF extraction in this corpus is noisy in predictable ways:

* ligatures / smart quotes / non-breaking spaces survive into the text layer
* running headers and footers repeat on every page
  ("Purchase Order Request for Proposals (PORFP)  Hardware Master Contract  3",
   "Page 4", "Dell Marketing LP. U.S. only. ...")
* hyphenation splits tokens across line breaks

Cleaning these *before* extraction stops boilerplate from being mistaken for
field values (e.g. a repeated footer line polluting ``company_name``).
"""

from __future__ import annotations

import re
import unicodedata

# --- unicode ---------------------------------------------------------------

_UNICODE_REPLACEMENTS = {
    "\u00a0": " ",  # nbsp
    "\u2007": " ",
    "\u202f": " ",
    "\u2009": " ",
    "\u200b": "",  # zero-width space
    "\u200c": "",
    "\u200d": "",
    "\ufeff": "",
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201b": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u2026": "...",
    "\ufb01": "fi",
    "\ufb02": "fl",
}

_WS_RE = re.compile(r"[ \t]+")
_MULTI_NL_RE = re.compile(r"\n{3,}")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")

#: Running headers/footers observed in this corpus (and common variants).
BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*Page\s+\d+\s*(\|\s*\d+\s*)?$", re.I),
    re.compile(r"^\s*\d+\s*\|\s*\d+\s*$"),
    re.compile(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", re.I),
    re.compile(r"^\s*Purchase Order Request for Proposals\s*\(PORFP\)\s*$", re.I),
    re.compile(r"^\s*Hardware Master Contract\s*$", re.I),
    re.compile(r"^\s*Dell Marketing LP\..*$", re.I),
    re.compile(r"^\s*Dallas ISD\s*$", re.I),
    re.compile(r"^\s*RFP\s+JA-207652.*$", re.I),
    re.compile(r"^\s*BidNet\s+Direct\s*$", re.I),
)


def fix_unicode(text: str) -> str:
    """NFKC-normalize then apply targeted punctuation repairs."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _UNICODE_REPLACEMENTS.items():
        if src in text:
            text = text.replace(src, dst)
    return text


def collapse_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    """Rejoin words split across lines ('informa-\\ntion' -> 'information').

    Only applied when both fragments are alphabetic, so legitimate hyphens
    such as 'Wi-Fi 6E' or part numbers are preserved.
    """
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def normalize_text(text: str, *, dehyphenate_words: bool = True) -> str:
    out = fix_unicode(text)
    if dehyphenate_words:
        out = dehyphenate(out)
    return collapse_whitespace(out)


#: Ariba/SAP-style PDFs emit form labels and values with no separator:
#: "BuyerALZATE, JASMINEEmailJALZATE@dallasisd.org".  Left alone, an email regex
#: happily swallows "JASMINEEmailJALZATE@..." as the address.  Re-inserting the
#: boundary is the cheapest correct fix.
#: The lookbehind is "(?<![^A-Za-z0-9])" - i.e. preceded by an alphanumeric
#: *or* by nothing at all.  Both cases occur:
#:   "BuyerALZATE"      -> label at the start of the string
#:   "JASMINEEmailJ..." -> label glued onto the previous value
_FIELD_WORD_RE = re.compile(
    r"(?<![^A-Za-z0-9])(Buyer|Email|Phone|Telephone|Fax|Address|Name|Date|Title|City)(?=[A-Z0-9])"
)


def deconcatenate_labels(text: str) -> str:
    """Split glued label/value pairs such as ``BuyerALZATE`` -> ``Buyer ALZATE``."""
    return _FIELD_WORD_RE.sub(lambda m: f" {m.group(1)} ", text).lstrip()


def is_boilerplate(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return True
    return any(p.match(stripped) for p in BOILERPLATE_PATTERNS)


def strip_boilerplate_lines(text: str) -> str:
    kept = [ln for ln in text.split("\n") if not is_boilerplate(ln)]
    return "\n".join(kept).strip()


def squish(text: str) -> str:
    """Single-line, single-spaced form - used for signatures and matching."""
    return " ".join(text.split())


#: Repeated header/footer detection -------------------------------------------------
def detect_running_lines(pages: list[list[str]], min_pages: int = 3) -> set[str]:
    """Lines that appear on >=60% of pages (and at least ``min_pages``) are boilerplate.

    This is corpus-agnostic: rather than hardcoding every issuer's footer, we
    learn it from repetition frequency.
    """
    if len(pages) < min_pages:
        return set()
    counts: dict[str, int] = {}
    for lines in pages:
        for line in {ln.strip() for ln in lines if ln.strip()}:
            counts[line] = counts.get(line, 0) + 1
    threshold = max(min_pages, int(0.6 * len(pages)))
    return {line for line, n in counts.items() if n >= threshold}
