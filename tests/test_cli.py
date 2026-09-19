"""End-to-end test of the documented CLI entry point.

`python scripts/run_pipeline.py` is the command in the README; the unit tests
all drive the Python API directly, so nothing guarded the script itself (arg
parsing, writer wiring, exit code).  Runs on a single small document to keep it
fast.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_pipeline.py"


def test_cli_writes_artifacts_to_a_custom_output_dir(tmp_path, bid2_dir):
    source = bid2_dir / "Mercury_Affidavit.pdf"
    out = tmp_path / "out"

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(source),
            "--out",
            str(out),
            "--no-eval",
            "--quiet",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    assert (out / "extraction_all.json").is_file()
    assert (out / "provenance_report.md").is_file()
    per_doc = out / "per_document"
    assert per_doc.is_dir() and any(per_doc.glob("*.json"))

    payload = json.loads((out / "extraction_all.json").read_text(encoding="utf-8"))
    assert payload["packages"], "expected at least one package"
    for package in payload["packages"].values():
        assert "fields" in package
        assert "pricing_schedule" in package


def test_cli_reports_zero_packages_for_an_empty_directory(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    out = tmp_path / "out"

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(empty), "--out", str(out), "--no-eval", "--quiet"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads((out / "extraction_all.json").read_text(encoding="utf-8"))
    assert payload["packages"] == {}


@pytest.mark.parametrize("flag", ["--help"])
def test_cli_help(flag):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), flag], cwd=REPO_ROOT, capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "usage" in result.stdout.lower()
