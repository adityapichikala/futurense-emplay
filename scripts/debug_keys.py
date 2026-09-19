"""Compare flat-directory split against the directory-based baseline.

Any field that differs shows the real cost of orphan mis-assignment.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfp_extractor.models.schema import CANONICAL_FIELDS  # noqa: E402
from rfp_extractor.pipeline import Pipeline  # noqa: E402


def by_bid(result):
    return {p.bid_number: p for p in result.packages.values()}


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    baseline = by_bid(Pipeline().run(["data/raw/bid1", "data/raw/bid2"]))
    flat = by_bid(Pipeline().run(["tmp_flat"]))

    print("baseline bids:", sorted(baseline), " flat bids:", sorted(flat))
    for bid in sorted(baseline):
        other = flat.get(bid)
        print("=" * 80)
        print(bid, "->", "MISSING IN FLAT" if other is None else "present")
        if other is None:
            continue
        for field in CANONICAL_FIELDS:
            a, b = getattr(baseline[bid], field), getattr(other, field)
            if a != b:
                print(f"  DIFF {field}")
                print(f"      baseline: {str(a)[:100]}")
                print(f"      flat    : {str(b)[:100]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
