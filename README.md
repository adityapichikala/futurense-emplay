# RFP Intelligence Extraction Platform

Structured information extraction from RFP packages (PDF + HTML + DOCX) with
**addendum-aware conflict resolution** and **per-field provenance**.

Given a folder of solicitation documents, the pipeline emits one JSON object per
bid package containing the 19 fields required by the assignment, where every
value carries a citation (file, page, verbatim quote) and a confidence score.

```
data/raw/bid1/  ─┐
                 ├─► ingestion ─► classification ─► retrieval ─► extraction ─► merge
data/raw/bid2/  ─┘        │             │               │            │           │
                     layout-aware   precedence     BM25+TF-IDF   per-doc     conflict
                     PDF/HTML/DOCX    graph          hybrid      candidates   resolution
                                                                                  │
                                                                    artifacts/*.json + report
```

---

## Why this is not a regex script

| Concern | How it is handled |
|---|---|
| **Supersession** | Addendum 2 changes the Dallas ISD due date to 09-JUL-2024 while the master RFP still says 27-JUN-2024. A `DocumentGraph` assigns explicit precedence (addendum > portal > master) and the merge stage records the override instead of silently picking one. |
| **Equivalent dates** | "2:00 PM CST" and "3:00 PM EDT" are the *same instant*. Values are compared by UTC instant, so they corroborate rather than conflict. |
| **Corrupted PDF order** | PyMuPDF span geometry is re-sorted into column-aware reading order; blocks keep `(page, bbox)` so every value is citable. |
| **Glued text** | Ariba PDFs emit `BuyerALZATE, JASMINE` and `Base210-BLYZ`. De-concatenation and boundary-aware SKU regexes recover them. |
| **Format heterogeneity** | Portal HTML grids, semi-structured RFP PDFs, vendor spec sheets and legal affidavits all reduce to one `Block` model. |
| **Graceful absence** | Missing fields return `null` with a human-readable reason. Nothing is guessed. |

---

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/run_pipeline.py                    # runs over data/raw, writes artifacts/
```

Output:

```
artifacts/extraction_bid1.json      # 19 fields + provenance for Dallas ISD (merged view)
artifacts/extraction_bid2.json      # 19 fields + provenance for Maryland (merged view)
artifacts/extraction_all.json       # consolidated
artifacts/per_document/*.json       # one file per input document (Pass 1, pre-merge)
artifacts/provenance_report.md      # citation table: field -> file -> page -> quote
artifacts/evaluation_report.json    # field-level accuracy vs the gold set
```

Two complementary views are produced because they answer different questions:

* **`per_document/`** — what each file says *on its own*. This is where the
  amendment story is visible: `Addendum 2` reports `due_date = 2024-07-09` while
  the master RFP still reports `2024-06-27`.
* **`extraction_<bid>.json`** — the resolved answer for the whole package after
  supersession, with every override recorded in `conflicts_resolved`.

Both also carry a supplementary **`pricing_schedule`** key (line → item, target
quantity, unit). Quantities exist only in the price grid — never in the prose —
and they are usually the most decision-relevant number in a solicitation:

```json
"pricing_schedule": {
  "Line 15.01": {"item": "Tier 1 Small Student Chromebook Laptop (Touch)",
                 "target_quantity": "50,000", "unit": "Each"},
  "Line 16.01": {"item": "Tier 3 Student Windows Laptop",
                 "target_quantity": "5,000", "unit": "Each"}
}
```

This is not one of the 19 assignment fields; it has no home in the field
contract, so it is reported alongside it rather than squeezed into
`product_specification`. Note that page 1 of the Dallas ISD schedule repeats
every line with quantity 1 — the extractor keeps the largest quantity seen per
line, so the real volumes (50,000 / 10,000 / 5,000) win.

### Optional: API service

```bash
make api                       # or: PYTHONPATH=src uvicorn rfp_extractor.api.main:app --reload
#  -> http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/extract \
  -F "files=@data/raw/bid1/Addendum 2 RFP JA-207652 Student and Staff Computing Devices.pdf" \
  -F "files=@data/raw/bid1/Student and Staff Computing Devices __SOURCING #168884__ - Bid Information - {3} _ BidNet Direct.html"
```

### Docker

```bash
make docker-build && make docker-run
```

> Note: the image could not be built in the development environment (the Docker
> daemon was not running), so `docker build` is **unverified**. The container's
> entrypoint command *is* verified: `PYTHONPATH=src uvicorn
> rfp_extractor.api.main:app` was started and served `/health`, `/fields` and
> `/openapi.json` successfully. `PYTHONPATH` is set inside the image — without
> it uvicorn cannot import `rfp_extractor` from `/app/src`.

---

## Architecture

```
src/rfp_extractor/
├── ingestion/      FileRouter -> PDF / HTML / DOCX loaders -> Block model
│   └── normalize.py   unicode repair, boilerplate stripping, de-concatenation
├── intelligence/   DocumentClassifier, AddendumDetector, DocumentGraph (precedence)
├── reasoning/      chunking, hybrid retriever, rule engine, 2-pass extractor
│   ├── rules.py        config-driven field mining with evidence spans
│   ├── spec_extractor.py  device tiers / line items -> structured specs
│   └── providers/      rule-based (default) and OpenAI (opt-in) providers
├── validation/     date/value normalization + gold-set evaluation harness
├── api/            FastAPI service
└── pipeline.py     orchestration
```

### Two-pass extraction

**Pass 1 (per document).** For each field the hybrid retriever selects the most
relevant chunks (BM25 for exact identifiers like `JA-207652`, TF-IDF vectors for
conceptual matches), then the rule engine mines those blocks. If retrieval finds
nothing, the engine falls back to a full scan so recall never silently depends
on the retriever. Aggregate fields (SKU lists) always get a full scan.

**Pass 2 (merge).** All per-document candidates are merged by precedence.
Disagreements become `ConflictResolution` records naming winner and superseded
values; agreements become corroborations that raise confidence.

### Configuration

Field behaviour lives in `configs/fields.yaml`, not in Python. To support a new
issuer, add patterns there:

```yaml
- name: due_date
  kind: datetime
  merge: precedence
  labels:
    - labels: ["Closing Date", "Solicitation Due"]
      confidence: 0.95
      transform: datetime
  patterns:
    - name: addendum_new_due_date
      regex: 'new\s+due\s+date\s+for\s+this\s+RFP\s+will\s+be\s+(?P<value>...)'
      confidence: 0.99
      doc_types: ["addendum"]
```

Merge strategies: `precedence | union | concat | longest | most_specific`.
Precedence weights are in the same file (`precedence.doc_types`).

---

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `RFP_PROVIDER` | `rule-based` | `openai` enables the LLM provider (see below) |
| `RFP_OPENAI_API_KEY` | – | API key for the OpenAI provider |
| `RFP_OPENAI_MODEL` | `gpt-4o-mini` | model for per-field extraction |
| `RFP_RETRIEVAL_TOP_K` | `6` | chunks retrieved per field |
| `RFP_EMBEDDING_BACKEND` | `tfidf` | `sentence-transformers` for dense vectors |
| `RFP_LOG_LEVEL` | `INFO` | logging verbosity |

No API key is required: the default provider is deterministic and offline.

### How the LLM provider is used

`RFP_PROVIDER=openai` does **not** replace the rule engine — it *augments* it.
For each document, a field is only sent to the model when no rule produced a
candidate above `min_confidence + 0.3`. The model's answer (default confidence
0.7) therefore fills gaps rather than competing with deterministic evidence the
pipeline already trusts, and Pass 2 merges both through the same precedence
graph, so an addendum still beats the master RFP.

Provider failures are isolated: a missing key, missing package, or API error
contributes no candidates and never aborts a run.

---

## Evaluation

`configs/gold_set.yaml` holds hand-labelled expectations per field
(`equals`, `contains`, `instant`, `min_items`, `absent`). The harness reports
per-field pass/fail so a regression is traceable to the rule that caused it.
[`docs/GOLD_SET_NOTES.md`](docs/GOLD_SET_NOTES.md) records the source document
and page behind every label, so any expectation can be challenged.

Current result on the bundled corpus:

```
Field-level accuracy: 100.0%
  bid1 (JA-207652):   20/20
  bid2 (BPM044557):   20/20
```

Run it with `python scripts/run_pipeline.py` (or `make test` for the full suite).

---

## Robustness

`tests/test_robustness.py` covers the failure modes a real corpus produces. The
rule is *degrade, never crash*: bad input is skipped or reported as null with a
reason, and good files in the same batch still extract.

| Input | Behaviour |
|---|---|
| Corrupt / truncated PDF | skipped with a warning |
| Zero-byte file | skipped with a warning |
| Unsupported extension (`.txt`, `.bin`) | no loader registered → skipped |
| HTML with no portal field grid | parsed, yields no labels |
| Empty directory | no packages, no error |
| Directory of only broken files | no packages, no error |
| Non-RFP document (e.g. a recipe page) | all fields null, each with a reason |
| Unicode filename | handled |

## Quality gates

```bash
make lint        # ruff
make typecheck   # mypy
make test        # pytest
make check       # all three
```

---

## Known limitations

* **Layout** is repaired heuristically; heavily multi-column pages with no
  horizontal gutter may still interleave.
* **Dense retrieval** is opt-in. The default TF-IDF backend is lexical, so
  paraphrase queries with no shared vocabulary will not match.
* **Timezones** are resolved from US DST rules when the `tzdata` package is
  absent (Windows). Install `tzdata` or use the Docker image for full IANA data.
* The **OpenAI provider** has never been run against the live API (no key in
  this environment). Its wiring — request shape, response parsing, retry,
  failure isolation, and the "only unanswered fields" gate — is covered by unit
  tests using an injected fake client, but a real model call is unverified.
* **`product_specification` for bid1 yields 10 device tiers.** The RFP defines 8
  blocks ending in "Minimum Requirements" plus 2 bare "Display Monitor" headings;
  the plan's estimate of 14 referred to pricing lines, a different structure
  that is not extracted as specifications.
