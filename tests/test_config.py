"""
tests/test_config.py — Smoke tests for nids.config.

These tests only check that the module imports correctly and that the
paths it defines are consistent. They do NOT require the dataset or models.
"""

from pathlib import Path

import pytest


def test_config_imports():
    """nids.config must be importable without errors."""
    from nids import config  # noqa: F401


def test_root_dir_is_directory():
    from nids.config import ROOT_DIR
    assert ROOT_DIR.is_dir(), f"ROOT_DIR does not exist: {ROOT_DIR}"


def test_data_dirs_are_under_root():
    from nids.config import DATA_DIR, MODELS_DIR, REPORTS_DIR, ROOT_DIR
    for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
        assert str(d).startswith(str(ROOT_DIR)), (
            f"{d} is not under ROOT_DIR ({ROOT_DIR})"
        )


def test_attack_categories_length():
    from nids.config import ATTACK_CATEGORIES
    # UNSW-NB15 has exactly 10 categories including Normal.
    assert len(ATTACK_CATEGORIES) == 10


def test_test_size_valid():
    from nids.config import TEST_SIZE
    assert 0.0 < TEST_SIZE < 1.0
