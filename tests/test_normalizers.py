"""Date and value normalization - the foundation everything else depends on."""

from __future__ import annotations

from datetime import timedelta, timezone

import pytest

from rfp_extractor.validation.normalizers import (
    dedupe_preserving_order,
    parse_datetime,
    prune_subsumed,
    sentence_containing,
    strip_leading_label,
    to_iso_instant,
    to_list,
)

CENTRAL = timezone(timedelta(hours=-5))


@pytest.mark.parametrize(
    "raw,expected_iso",
    [
        # Ariba PDF header notation
        ("27-JUN-2024 14:00:00", "2024-06-27T14:00:00-05:00"),
        ("26-MAY-2024 08:00:00", "2024-05-26T08:00:00-05:00"),
        # Addendum prose ("2:00 PM CST" in July is really CDT)
        ("July 9, 2024 at 2:00 PM CST", "2024-07-09T14:00:00-05:00"),
        # PDF spacing artefact: "2024at 2:00PM"
        ("July 9, 2024at 2:00PM", "2024-07-09T14:00:00-05:00"),
        # BidNet portal notation
        ("07/09/2024 03:00 PM EDT", "2024-07-09T15:00:00-04:00"),
        ("06/10/2024 02:00 PM EDT", "2024-06-10T14:00:00-04:00"),
        # Bare date (PORFP form field)
        ("06/10/2024", "2024-06-10T00:00:00-05:00"),
        # Already ISO - must survive untouched
        ("2024-07-09T14:00:00-05:00", "2024-07-09T14:00:00-05:00"),
    ],
)
def test_parse_datetime(raw, expected_iso):
    result = parse_datetime(raw, default_tz=CENTRAL)
    assert result is not None, f"failed to parse {raw!r}"
    assert result.iso() == expected_iso


def test_equivalent_instants_are_not_a_conflict():
    """2:00 PM Central and 3:00 PM Eastern are the same moment."""
    assert to_iso_instant("July 9, 2024 at 2:00 PM CST") == to_iso_instant(
        "07/09/2024 03:00 PM EDT"
    )


def test_different_days_are_different_instants():
    assert to_iso_instant("June 27, 2024 at 2:00 PM CST") != to_iso_instant(
        "July 9, 2024 at 2:00 PM CST"
    )


@pytest.mark.parametrize("garbage", ["", "   ", "not a date", "N/A", "-"])
def test_non_dates_return_none(garbage):
    assert parse_datetime(garbage) is None


def test_to_list_splits_and_cleans():
    assert to_list("210-BLYZ, 379-BFNZ; 619-ARSB") == ["210-BLYZ", "379-BFNZ", "619-ARSB"]


def test_to_list_drops_placeholders():
    assert to_list("N/A, none, 210-BLYZ") == ["210-BLYZ"]


def test_prune_subsumed_removes_shorter_variants():
    assert prune_subsumed(["Latitude 5550", "Dell Latitude 5550", "WD22TB4"]) == [
        "Dell Latitude 5550",
        "WD22TB4",
    ]


def test_dedupe_preserving_order():
    assert dedupe_preserving_order(["a", "B", "a", "c"]) == ["a", "B", "c"]


def test_strip_leading_label():
    assert strip_leading_label("Description: Specifications include") == "Specifications include"
    assert strip_leading_label("Answer:Dallas ISD is anticipating") == "Dallas ISD is anticipating"


def test_sentence_containing_finds_whole_sentence():
    text = "Intro. Dallas ISD prefers responses to be submitted online. Next sentence."
    start = text.index("prefers")
    assert sentence_containing(text, start, start + 8) == (
        "Dallas ISD prefers responses to be submitted online."
    )
