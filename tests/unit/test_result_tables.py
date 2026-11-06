"""Tests of the table of the outcomes of a run."""
import pytest
from dashboard_core.result_tables import outcome_rows


def test_rows_are_most_frequent_first():
    assert outcome_rows({"11": 497, "00": 503}) == [
        {"outcome": "00", "shots": 503, "probability": 0.503},
        {"outcome": "11", "shots": 497, "probability": 0.497},
    ]


def test_equal_shots_are_sorted_by_outcome():
    rows = outcome_rows({"10": 5, "01": 5, "00": 5})
    assert [row["outcome"] for row in rows] == ["00", "01", "10"]


def test_probabilities_sum_to_one():
    rows = outcome_rows({"000": 3, "001": 7, "111": 90})
    assert sum(row["probability"] for row in rows) == pytest.approx(1.0, abs=1e-12)


def test_empty_counts_give_no_row():
    assert outcome_rows({}) == []
