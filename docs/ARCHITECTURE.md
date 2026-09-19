# Architecture & design rationale

How the pipeline is put together, and — more importantly — **why** certain decisions were
made. Most of these are responses to a specific failure observed on the corpus; each is
recorded here so the reasoning isn't lost.

## Flow

```
files → ingestion (PDF/HTML/DOCX → Block model, content-hash cached)
      → intelligence (classify, detect amendments, build precedence graph)
      → reasoning (chunk; retrieve per field; mine rules; merge across documents)
      → validation (normalize; score against gold set)
      → reporting (per-package JSON, per-document JSON, provenance report)
```

The single most important structural property: **every extracted value carries provenance**
(file, page, verbatim span, confidence, extraction pass). Nothing is asserted without a
citation, and a test enforces it.

## Decision 1 — Precedence is explicit, not implicit

Addenda outrank the portal, which outranks the master RFP
(`Addendum N = 100 + 10·N > portal 60 > master 40`). When two documents disagree the
higher-precedence value wins **and the disagreement is recorded** in `conflicts_resolved`
with both values named.

Without this, the due date would come from whichever document happened to be processed
last. On this corpus it matters concretely: the Dallas ISD master RFP says
`Solicitation Due 27-JUN-2024`, Addendum 2 says `July 9, 2024 at 2:00 PM CST`. The
pipeline returns 09-JUL and records 27-JUN as superseded.

Vendor-scoped fields (`model_no`, `part_no`, `company_name`) get +200 for vendor
documents, so a quote sheet outranks the agency's own PDF for those fields.

## Decision 2 — Dates compare by instant, never by string

`2:00 PM CST` and `3:00 PM EDT` are the same moment. Comparing strings would report a
conflict where none exists — and, worse, would make the correct answer depend on which
source you happened to believe. Values are normalized to ISO-8601 with an offset and
compared as UTC instants, so the addendum and the portal *corroborate* instead of
conflicting.

Naive timestamps (no offset in the source) are read in the issuer's timezone, inferred
from abbreviations in the issuer's **own** documents — portal notices are excluded from
that vote, because their timestamps always carry an explicit offset and would otherwise
outvote the issuer by sheer repetition.

US DST rules are implemented directly as a fallback when the `tzdata` package is absent
(common on Windows); `zoneinfo` is preferred when available.

## Decision 3 — Parsed documents are cached by content hash

pdfplumber's line analysis is ~99% of parse time (84s vs 0.8s for PyMuPDF text alone on a
62-page RFP). Parsed documents are cached under `.cache/ingest`, keyed by SHA-256 **plus a
loader version**, so changed files never read stale data and parser changes invalidate
entries instead of serving documents built by old code.

PyMuPDF's native `find_tables()` is 3× faster but was **rejected**: it does not detect the
Dallas ISD pricing grid that the schedule extraction depends on.

## Decision 4 — Field behaviour is configuration, not code

`configs/fields.yaml` declares each field's label synonyms, regex patterns, transforms and
merge strategy (`precedence | union | concat | longest | most_specific`). Supporting a new
issuer means editing YAML.

Two hard-won regex rules, both learned from real defects:

* **Avoid loose trailing quantifiers.** `[^.,;]{0,60}` and `(?:\s+\S+){0,3}` were the root
  cause of nearly every value-quality bug — they run into the next sentence and produce
  values like `FORM 1295 … the RFP states that "T`.
* **Glued tokens need context, not boundaries.** PDFs emit `XCTO Base210-BLYZ` (no space
  before a SKU) and `JASMINEEmailJALZATE@…` (label fused to value). A leading `\b` silently
  drops most SKUs; the fix is a context guard (`(?<![0-9)\-.])`), not a word boundary.

## Decision 5 — A document group is split by solicitation identity

Grouping by directory is not enough. Drop two bids into one folder — or upload them in a
single API call — and they merge into one package, collapsing accuracy to zero. Documents
are clustered by the solicitation number they cite; a single occurrence of `JA-…`, `BPM…`
or `E…P…` suffices, and **filenames count** (issuers name files after the solicitation).

Documents citing none (vendor spec sheets, legal appendices) are placed by similarity and
**reported**: any placement that relied on vocabulary rather than a shared identifier
appends a warning with its score and margin. See below.

## Known limitation — appendix placement

A one-page generic legal appendix shares more vocabulary with a 62-page RFP than with a
4-page PORFP, so `Contract_Affidavit.pdf` is attributed to the Dallas bid, giving
`additional_documentation_required` one entry it should not have.

Five scoring variants were tried (raw overlap, max cosine, mean cosine, shared-within-
cluster, contrastive) and none fixed it. Rather than pretend to confidence that doesn't
exist, the run reports it:

```
`Contract_Affidavit.pdf` cites no solicitation number and was placed by
vocabulary similarity (score 0.0777, margin 0.0095); verify this grouping
```

A margin under 0.01 means the assignment is genuinely a coin flip. **Surfacing uncertainty
beat tuning a heuristic that cannot be made reliable.**

## Decision 6 — Absence is a first-class answer

Ten gold-set expectations assert a field must be `null`. A pipeline that invents a
plausible value scores *worse* than one that returns nothing, because for procurement data
a fabricated due date is far more damaging than a missing one. Missing fields always carry
a human-readable `null_reason`.

## Testing philosophy

The gold set uses `contains` / `min_items` / `instant` / `absent` rather than exact
equality, so it measures whether the *fact* was captured rather than a specific phrasing.
That coarseness has a cost, though: several value-quality defects (corrupted emails,
`See more` UI chrome, truncated fragments) passed the gold checks while being visibly
wrong to a human. Those are now covered by dedicated invariant tests:

* no malformed email addresses
* no portal UI chrome in any value
* no truncated / unbalanced quotes
* every populated field has at least one citation
* every absent field has a reason

**Lesson:** a coarse metric proves the fact was found, not that the value is clean. Both
kinds of test are needed.
