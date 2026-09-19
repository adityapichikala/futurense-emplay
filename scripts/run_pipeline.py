"""CLI entry point: run the full pipeline over a corpus and emit artifacts.

Usage:
    python scripts/run_pipeline.py                       # data/raw
    python scripts/run_pipeline.py data/raw/bid1         # one package
    python scripts/run_pipeline.py --out artifacts       # custom output dir
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfp_extractor.logging_setup import configure_logging  # noqa: E402
from rfp_extractor.pipeline import Pipeline  # noqa: E402
from rfp_extractor.reporting import ArtifactWriter  # noqa: E402
from rfp_extractor.validation.evaluation import evaluate  # noqa: E402

log = logging.getLogger("rfp_extractor")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="RFP structured-extraction pipeline")
    parser.add_argument("paths", nargs="*", default=["data/raw"], help="files or directories")
    parser.add_argument("--out", default="artifacts", help="artifact output directory")
    parser.add_argument("--no-eval", action="store_true", help="skip the evaluation harness")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    configure_logging("WARNING" if args.quiet else "INFO")

    paths = args.paths or ["data/raw"]
    pipeline = Pipeline()
    result = pipeline.run(paths)

    evaluation = None if args.no_eval else evaluate(result)

    writer = ArtifactWriter(args.out)
    written = writer.write_all(result, evaluation, pipeline.document_extractions)

    print("\n" + "=" * 78)
    for package_id, package in result.packages.items():
        filled, total = package.coverage()
        print(f"{package_id}: {package.bid_number or '?'} - {filled}/{total} fields populated"
              f" - {len(package.conflicts_resolved)} conflict(s) resolved")
    if evaluation is not None:
        print(f"\nField-level accuracy: {evaluation.accuracy * 100:.1f}%")
        for pkg in evaluation.packages:
            print(f"  {pkg.package}: {pkg.passed}/{pkg.total}")
            for score in pkg.scores:
                if not score.passed:
                    print(f"    FAIL {score.field}: expected {score.expected!r}, got {score.actual!r}")
    print("\nArtifacts:")
    for name, path in written.items():
        print(f"  {name}: {path}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
