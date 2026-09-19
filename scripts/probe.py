"""Ad-hoc inspection helper: dump what ingestion produced for a file or directory.

Usage:
    python scripts/probe.py data/raw/bid1
    python scripts/probe.py data/raw/bid2/PORFP_-_Dell_Laptop_Final.pdf --text
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rfp_extractor.ingestion.router import FileRouter  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--text", action="store_true", help="print full text")
    ap.add_argument("--labels", action="store_true", help="print label->value map")
    ap.add_argument("--blocks", action="store_true", help="print block index")
    ap.add_argument("--limit", type=int, default=40)
    args = ap.parse_args()

    target = Path(args.target)
    router = FileRouter()
    docs = [router.load_one(target)] if target.is_file() else router.load_directory(target)

    for doc in docs:
        if doc is None:
            continue
        print("=" * 100)
        print(f"{doc.file_name}  [{doc.file_format.value}]  pages={doc.page_count}  blocks={len(doc.blocks)}")
        if args.labels or (not args.text and not args.blocks):
            if doc.label_values:
                print("-- label_values --")
                for k, v in doc.label_values.items():
                    print(f"   {k!r:55} -> {v!r}")
        if args.blocks:
            print("-- blocks --")
            for b in doc.blocks[: args.limit]:
                print(f"   [{b.block_id:12}] {b.kind.value:12} {b.text[:110]!r}")
        if args.text:
            print("-- text --")
            print(doc.full_text[: 4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
