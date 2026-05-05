"""Tests for functions declared in __main__."""

import pytest

import beave.__main__ as beave_main


@pytest.mark.parametrize(
    "normalized,thresholds,expected,match",
    [
        (
            True,
            [100.0, 90.0],
            beave_main.CommandError,
            "but values greater than or equal to 100.0 are provided. 100.0",
        ),  # Too high
        (
            True,
            [100.0, -90.0],
            beave_main.CommandError,
            "but values greater than or equal to 100.0 are provided. 100.0",
        ),  # Too high and too low
        (
            True,
            [101.0, -90.0],
            beave_main.CommandError,
            "but values greater than or equal to 100.0 are provided. 101.0",
        ),  # Too high and too low
        (
            True,
            [101.0, -90.0],
            beave_main.CommandError,
            "but values less than or equal to 0.0 are provided. -90.0",
        ),  # Too high and too low
        (
            True,
            [101.0, 90.0],
            beave_main.CommandError,
            "but values greater than or equal to 100.0 are provided. 101.0",
        ),  # Too high
        (
            True,
            [float("inf"), 100.0, -90.0],
            beave_main.CommandError,
            "but values less than or equal to 0.0 are provided. -90.0",
        ),  # Too high
        (
            True,
            [99.0, -90.0],
            beave_main.CommandError,
            "but values less than or equal to 0.0 are provided. -90.0",
        ),  # Too low
        (
            True,
            [99.0, 90.0],
            None,
            "",
        ),  # Too low
        (
            True,
            -1.0,
            beave_main.CommandError,
            "but values less than or equal to 0.0 are provided. -1.0",
        ),
        (
            True,
            100.0,
            beave_main.CommandError,
            "but values greater than or equal to 100.0 are provided. 100.0",
        ),
    ],
)
def test_verify_normalized_distance(normalized, thresholds, expected, match):
    """Test verify_normalized_distance raises the expected errors."""
    if expected:
        with pytest.raises(expected, match=match):
            beave_main.verify_normalized_distance(normalized, thresholds)
    else:
        assert beave_main.verify_normalized_distance(normalized, thresholds) is None
