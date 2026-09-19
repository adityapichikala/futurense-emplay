"""Ad-hoc check: a flat directory containing documents from two different bids."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfp_extractor.pipeline import Pipeline  # noqa: E402
from rfp_extractor.validation.evaluation import evaluate  # noqa: E402


def main() -> int:
    logging.basicConfig(level=logging.WARNING)
    pipeline = Pipeline()
    result = pipeline.run(["tmp_flat"])
    print("packages:", list(result.packages))
    for pid, pkg in result.packages.items():
        print(f"  {pid}: bid={pkg.bid_number} docs={len(pkg.documents)} due={pkg.due_date}")
    if result.packages:
        print("ACC: %.1f%%" % (evaluate(result).accuracy * 100))
    else:
        print("no packages -> cannot evaluate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
