"""Tests for functions declared in __main__."""

import pytest

import beave.__main__ as beave_main


@pytest.fixture()
def thresholds_validator():
    """Fixture for testing validation of thresholds passed to cluster."""
    thresh_validator = beave_main.ArgValidator()
    thresh_validator.add_validation_function(beave_main.verify_does_not_contain_infinity)
    thresh_validator.add_validation_function(beave_main.verify_normalized_distance)
    return thresh_validator


@pytest.fixture()
def threshold_validator():
    """Fixture for testing validation of thresholds passed to cluster."""
    thresh_validator = beave_main.ArgValidator()
    thresh_validator.add_validation_function(beave_main.verify_normalized_distance)
    return thresh_validator


@pytest.mark.parametrize(
    "thresholds,expected,match",
    [
        (
            [100.0, 90.0],
            beave_main.CommandError,
            "but values greater than or equal to 100.0 are provided. 100.0",
        ),  # Too high
        (
            [100.0, -90.0],
            beave_main.CommandError,
            "but values greater than or equal to 100.0 are provided. 100.0",
        ),  # Too high and too low
        (
            [101.0, -90.0],
            beave_main.CommandError,
            "but values greater than or equal to 100.0 are provided. 101.0",
        ),  # Too high and too low
        (
            [101.0, -90.0],
            beave_main.CommandError,
            "but values less than or equal to 0.0 are provided. -90.0",
        ),  # Too high and too low
        (
            [101.0, 90.0],
            beave_main.CommandError,
            "but values greater than or equal to 100.0 are provided. 101.0",
        ),  # Too high
        (
            [100.0, -90.0],
            beave_main.CommandError,
            "but values less than or equal to 0.0 are provided. -90.0",
        ),  # Too high
        (
            [float("inf"), 100.0, -90.0],
            beave_main.CommandError,
            "Sorry, infinity can not be used as a threshold.",
        ),  # Too high
        (
            [float("inf"), 150.0],
            beave_main.CommandError,
            "Sorry, infinity can not be used as a threshold.",
        ),  # Too high
        (
            [99.0, -90.0],
            beave_main.CommandError,
            "but values less than or equal to 0.0 are provided. -90.0",
        ),  # Too low
        (
            [99.0, 90.0],
            None,
            "",
        ),  # No error
    ],
)
def test_verify_normalized_distance_thresholds(thresholds_validator, thresholds, expected, match):
    """Test verify_normalized_distance raises the expected errors."""
    if expected:
        with pytest.raises(expected, match=match):
            thresholds_validator(thresholds)
    else:
        assert thresholds_validator(thresholds) is None


@pytest.mark.parametrize(
    "thresholds,expected,match",
    [
        (
            -1.0,
            beave_main.CommandError,
            "but values less than or equal to 0.0 are provided. -1.0",
        ),
        (
            100.0,
            beave_main.CommandError,
            "but values greater than or equal to 100.0 are provided. 100.0",
        ),
    ],
)
def test_verify_normalized_distance_threshold(threshold_validator, thresholds, expected, match):
    """Test verify_normalized_distance raises the expected errors."""
    if expected:
        with pytest.raises(expected, match=match):
            threshold_validator(thresholds)
    else:
        assert threshold_validator(thresholds) is None
