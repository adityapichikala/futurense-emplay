"""Shared pytest fixtures: repo paths and a cached corpus load.

Loading the 62-page master RFP is slow, so document fixtures are session-scoped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

RAW_DIR = REPO_ROOT / "data" / "raw"
BID1_DIR = RAW_DIR / "bid1"
BID2_DIR = RAW_DIR / "bid2"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def bid1_dir() -> Path:
    return BID1_DIR


@pytest.fixture(scope="session")
def bid2_dir() -> Path:
    return BID2_DIR


@pytest.fixture(scope="session")
def corpus_available() -> bool:
    return BID1_DIR.is_dir() and BID2_DIR.is_dir()


@pytest.fixture(scope="session")
def pipeline_run():
    """(pipeline, result) over the bundled corpus.

    Session-scoped because the 62-page master RFP takes ~40s to parse; the
    pipeline instance also carries the per-document extractions.
    """
    from rfp_extractor.pipeline import Pipeline

    pipeline = Pipeline()
    return pipeline, pipeline.run([BID1_DIR, BID2_DIR])


@pytest.fixture(scope="session")
def consolidated(pipeline_run):
    return pipeline_run[1]
