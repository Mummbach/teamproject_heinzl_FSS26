"""
config.py — RANGE_FILTERS regression + extract_section tests.

config.py has no leading digit/hyphen in its filename and no top-level I/O,
so it imports normally (no need for the conftest loader helper).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config


def test_range_filters_reject_zero_probe_disconnect_reading():
    # Regression for the 2026-07-16 fix (known_issue_zero_reading_artifacts):
    # a literal 0 is a monitor-disconnect artifact, not a real reading, for
    # these five vitals. If any of these lower bounds regress back to 0, a
    # single dropped-lead reading can silently dominate mean/min/last again.
    vitals_that_cannot_be_zero = ["heart_rate", "sbp", "dbp", "map", "spo2"]
    for vital in vitals_that_cannot_be_zero:
        lo, _hi = config.RANGE_FILTERS[vital]
        assert lo > 0, f"{vital} lower bound regressed to allowing 0"


def test_range_filters_bounds_are_physiologically_ordered():
    for feature, (lo, hi) in config.RANGE_FILTERS.items():
        assert lo < hi, f"{feature} has an invalid (lo, hi) range: ({lo}, {hi})"


def test_extract_section_returns_empty_for_missing_or_non_string():
    assert config.extract_section("FINDINGS: normal.", "IMPRESSION") == ""
    assert config.extract_section(None, "FINDINGS") == ""


def test_extract_section_extracts_named_section_up_to_next_header():
    text = "FINDINGS: Lungs clear.\nIMPRESSION: No acute process."
    assert config.extract_section(text, "FINDINGS") == "Lungs clear."
    assert config.extract_section(text, "IMPRESSION") == "No acute process."


def test_extract_section_is_case_insensitive_and_multiline():
    text = "findings:\nBilateral effusions,\nmild.\nIMPRESSION: stable."
    assert config.extract_section(text, "FINDINGS") == "Bilateral effusions,\nmild."
