"""End-to-end extraction, conflict resolution and evaluation tests.

The conflict-resolution tests are the ones that matter most: they encode the
behaviour that distinguishes this pipeline from a naive extractor (which would
happily return the stale master-RFP due date).
"""

from __future__ import annotations

import re

from rfp_extractor.intelligence.graph import DocumentGraph
from rfp_extractor.models.schema import CANONICAL_FIELDS
from rfp_extractor.validation.evaluation import evaluate, load_gold_set


def test_canonical_field_contract_is_complete():
    assert len(CANONICAL_FIELDS) == 20
    for name in ("bid_number", "due_date", "model_no", "product_specification"):
        assert name in CANONICAL_FIELDS


def test_precedence_orders_addenda_above_master():
    graph = DocumentGraph()
    assert graph.weights["addendum"] > graph.weights["portal_notice"]
    assert graph.weights["portal_notice"] > graph.weights["master_rfp"]


def test_bid1_due_date_uses_addendum_two(consolidated):
    """Master RFP says 27-JUN-2024; Addendum 2 extends to 09-JUL-2024."""
    bid1 = consolidated.packages["bid1"]
    assert bid1.due_date is not None
    assert bid1.due_date.startswith("2024-07-09")


def test_bid1_due_date_conflict_is_recorded(consolidated):
    bid1 = consolidated.packages["bid1"]
    conflict = bid1.fields["due_date"].conflict
    assert conflict is not None
    assert "2024-06-27" in str(conflict.losers[0].normalized)


def test_bid1_fields(consolidated):
    bid1 = consolidated.packages["bid1"]
    assert bid1.bid_number == "JA-207652"
    assert "Student and Staff Computing Devices" in (bid1.title or "")
    assert "three (3) year" in (bid1.term_of_bid or "")
    assert "iSupplier" in (bid1.mfg_for_registration or "")
    assert bid1.model_no is None, "a buyer-side RFP has no vendor model numbers"
    assert bid1.company_name is None
    assert bid1.bid_number and bid1.due_date


def test_bid1_product_specification_is_tiered(consolidated):
    specs = consolidated.packages["bid1"].product_specification or {}
    assert len(specs) >= 5
    assert any("Chromebook" in key for key in specs)
    assert any("Desktop" in key for key in specs)


def test_bid2_fields(consolidated):
    bid2 = consolidated.packages["bid2"]
    assert bid2.bid_number == "BPM044557"
    assert "45 days" in (bid2.delivery_date or "")
    assert "Master Contract" in (bid2.contract_or_cooperative_to_use or "")
    assert "Mercury Affidavit" in (bid2.additional_documentation_required or "")
    assert "Dell" in (bid2.company_name or "")
    assert bid2.model_no is not None and len(bid2.model_no) >= 1


def test_bid2_model_and_part_numbers(consolidated):
    bid2 = consolidated.packages["bid2"]
    models = " ".join(bid2.model_no or []).lower()
    assert "latitude 5550" in models
    assert any("wd22tb4" in m.lower() for m in (bid2.model_no or []))
    assert bid2.part_no is not None and len(bid2.part_no) >= 20


def test_bid2_contact_information(consolidated):
    contact = (consolidated.packages["bid2"].contact_info or "").lower()
    assert "thawkins@treasurer.state.md.us" in contact
    assert "tamaira hawkins" in contact


def test_every_populated_field_has_evidence(consolidated):
    for package in consolidated.packages.values():
        for name, fv in package.fields.items():
            if fv.present:
                assert fv.evidence, f"{package.bid_id}.{name} has no citation"
                for ev in fv.evidence:
                    assert ev.file_name
                    assert ev.span.strip()


def test_absent_fields_report_a_reason(consolidated):
    for package in consolidated.packages.values():
        for name, fv in package.fields.items():
            if not fv.present:
                assert fv.null_reason, f"{package.bid_id}.{name} missing null_reason"


def test_gold_set_accuracy(consolidated):
    report = evaluate(consolidated)
    assert report.packages, "no packages evaluated"
    for package_score in report.packages:
        failures = [s.field for s in package_score.scores if not s.passed]
        assert not failures, f"{package_score.package} failed on {failures}"
    assert report.accuracy == 1.0


def test_per_document_extractions_cover_every_input_file(pipeline_run):
    """The assignment asks for structured data per provided document."""
    pipeline, _ = pipeline_run
    docs = pipeline.document_extractions
    assert len(docs) == 9, [d.file_name for d in docs]
    for doc in docs:
        assert doc.fields, f"{doc.file_name} produced no fields"
        for name, fv in doc.fields.items():
            if fv.present:
                assert fv.evidence, f"{doc.file_name}.{name} has no citation"


def test_addendum_two_carries_the_corrected_due_date(pipeline_run):
    """Per-document view is where the amendment story is visible."""
    pipeline, _ = pipeline_run
    addendum2 = next(
        d for d in pipeline.document_extractions if d.addendum_number == 2
    )
    assert addendum2.values["due_date"].startswith("2024-07-09")
    master = next(
        d for d in pipeline.document_extractions if d.doc_type == "master_rfp"
    )
    assert master.values["due_date"].startswith("2024-06-27")


def test_package_id_falls_back_to_bid_number_for_uploads():
    """Uploads land in a temp dir; the response key must stay meaningful."""
    from rfp_extractor.intelligence.graph import BidPackage
    from rfp_extractor.pipeline import Pipeline

    pkg = BidPackage(package_id="rfp_upload_kqg56qk2", bid_number="JA-207652")
    assert Pipeline._label_package("rfp_upload_kqg56qk2", pkg) == "ja-207652"
    # a real directory name is left alone
    plain = BidPackage(package_id="bid1", bid_number="JA-207652")
    assert Pipeline._label_package("bid1", plain) == "bid1"


_TLD = re.compile(
    r"\.(com|org|net|edu|gov|mil|us|uk|ca|au|de|fr|jp|in|io|co|info|biz)$", re.I
)
_EMAIL_RE = re.compile(r"[\w.%+\-]+@[\w.\-]+")


def test_no_malformed_emails_in_contact_info(consolidated):
    """PDF line-splits used to yield 'thawkins@treasurer.state.mdAgency'.

    The gold check only asserts that a *correct* address is present, so this
    junk passed while sitting in the deliverable.
    """
    for package in consolidated.packages.values():
        contact = package.contact_info or ""
        for found in _EMAIL_RE.findall(contact):
            assert _TLD.search(found), f"malformed address in {package.bid_id}: {found}"


def test_no_truncated_or_unbalanced_fragments(consolidated):
    """Values must not end mid-quote or trail off mid-sentence.

    Two real defects this catches: 'FORM 1295 ... the RFP states that "T'
    (dangling quote) and 'iSupplier portal ... This process takes'
    (start of the following sentence).
    """
    dangling = re.compile(r'["\u201c][^"\u201d]{0,3}$')
    for package in consolidated.packages.values():
        for name, fv in package.fields.items():
            value = fv.display
            if not isinstance(value, str) or not value:
                continue
            assert not dangling.search(value), f"{package.bid_id}.{name} ended mid-quote: {value!r}"


def test_no_portal_ui_chrome_in_values(consolidated):
    """BidNet appends a 'See more' affordance to truncated bodies."""
    for package in consolidated.packages.values():
        for name, fv in package.fields.items():
            value = fv.display
            if isinstance(value, str):
                assert "see more" not in value.lower(), f"{package.bid_id}.{name}"


def test_contact_info_excludes_vendor_boilerplate(consolidated):
    """contact_info is the buyer's contact, not the vendor's tax department."""
    contact = (consolidated.packages["bid2"].contact_info or "").lower()
    assert "dell.com" not in contact
    assert "thawkins@treasurer.state.md.us" in contact


def test_pricing_schedule_is_extracted(consolidated):
    """Quantities live only in the price grid - the plan calls for lines 1-20."""
    schedule = consolidated.packages["bid1"].pricing_schedule or {}
    assert len(schedule) >= 10, sorted(schedule)
    line = schedule.get("Line 15.01")
    assert line is not None, sorted(schedule)
    assert line["item"].startswith("Tier 1 Small Student Chromebook")
    assert line["target_quantity"] == "50,000"
    assert line["unit"] == "Each"


def test_pricing_schedule_keeps_the_largest_quantity(consolidated):
    """Page 1 repeats every line with quantity 1; the real volume is 50,000."""
    schedule = consolidated.packages["bid1"].pricing_schedule or {}
    assert schedule["Line 15.01"]["target_quantity"] == "50,000"
    assert schedule["Line 18.01"]["target_quantity"] == "10,000"


def test_pricing_schedule_handles_grids_without_a_line_column(consolidated):
    """The PORFP grid is Product/Qty with no Line column."""
    schedule = consolidated.packages["bid2"].pricing_schedule or {}
    assert len(schedule) >= 2, schedule
    items = " ".join(v["item"] for v in schedule.values()).lower()
    assert "latitude 5550" in items
    assert all(v["target_quantity"] == "30" for v in schedule.values())


def test_evaluation_matches_gold_by_bid_number_not_folder_name(consolidated):
    """Regression: a correct extraction used to score 0%.

    Package ids are directory names, so a flat folder (which splits into
    packages named after each solicitation) produced ids the gold set did not
    know, and every correct answer was scored as missing.
    """
    from rfp_extractor.models.schema import ConsolidatedResult

    renamed = ConsolidatedResult(
        packages={pkg.bid_number.lower(): pkg for pkg in consolidated.packages.values()},
        metadata=consolidated.metadata,
    )
    assert set(renamed.packages) == {"ja-207652", "bpm044557"}
    report = evaluate(renamed)
    assert {p.package for p in report.packages} == {"ja-207652", "bpm044557"}
    assert report.accuracy == 1.0


def test_gold_entries_declare_their_bid_number():
    gold = load_gold_set()
    for key, entry in gold.items():
        assert entry.get("bid_number"), f"gold entry {key} must declare bid_number"
        assert entry["expectations"], f"gold entry {key} has no expectations"


def test_coverage_reasonable(consolidated):
    for package in consolidated.packages.values():
        filled, total = package.coverage()
        assert total == len(CANONICAL_FIELDS)
        assert filled >= 12, f"{package.bid_id} only filled {filled}/{total}"


def test_pipeline_is_deterministic(bid1_dir, bid2_dir):
    """Re-running must produce identical values (idempotency)."""
    from rfp_extractor.pipeline import Pipeline

    first = Pipeline().run([bid1_dir, bid2_dir])
    second = Pipeline().run([bid1_dir, bid2_dir])
    for pid in first.packages:
        a = {k: v.normalized for k, v in first.packages[pid].fields.items()}
        b = {k: v.normalized for k, v in second.packages[pid].fields.items()}
        assert a == b
