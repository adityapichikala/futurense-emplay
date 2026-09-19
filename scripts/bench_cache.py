"""Measure the ingestion cache: run the pipeline twice and compare wall time.

Also asserts the second (cached) run produces identical field values, which is
the property that makes caching safe.
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfp_extractor.models.schema import CANONICAL_FIELDS  # noqa: E402
from rfp_extractor.pipeline import Pipeline  # noqa: E402
from rfp_extractor.validation.evaluation import evaluate  # noqa: E402


def snapshot(result):
    return {
        pid: {f: getattr(pkg, f) for f in CANONICAL_FIELDS}
        for pid, pkg in result.packages.items()
    }


def main() -> int:
    logging.basicConfig(level=logging.ERROR)
    times = []
    snaps = []
    for run in (1, 2):
        start = time.time()
        result = Pipeline().run(["data/raw/bid1", "data/raw/bid2"])
        elapsed = time.time() - start
        times.append(elapsed)
        snaps.append(snapshot(result))
        print(f"run {run}: {elapsed:.1f}s   accuracy={evaluate(result).accuracy * 100:.0f}%")

    print(f"speedup: {times[0] / times[1]:.1f}x")
    print("results identical:", snaps[0] == snaps[1])
    if snaps[0] != snaps[1]:
        for pid in snaps[0]:
            for field in CANONICAL_FIELDS:
                if snaps[0][pid][field] != snaps[1][pid][field]:
                    print("  DIFF", pid, field)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
