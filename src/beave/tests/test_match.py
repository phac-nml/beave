"""Tests for match.py."""

import pytest  # noqa: I001

from dataclasses import dataclass
import hashlib
from pathlib import Path

from beave import match

import numpy as np
import polars as pl


@pytest.mark.parametrize(
    "query,reference,expected",
    [
        (
            pl.DataFrame(
                {
                    "SampleID": ["1", "2", "3"],
                    "A": ["1", "2", "3"],
                    "B": ["1", "2", "3"],
                    "C": ["1", "2", "3"],
                }
            ),
            pl.DataFrame(
                {
                    "SampleID": ["4", "5", "6"],
                    "A": ["1", "2", "3"],
                    "B": ["1", "2", "3"],
                    "C": ["1", "2", "3"],
                }
            ),
            pl.DataFrame(
                {
                    "SampleID": ["1", "2", "3", "4", "5", "6"],
                    "A": ["1", "2", "3", "1", "2", "3"],
                    "B": ["1", "2", "3", "1", "2", "3"],
                    "C": ["1", "2", "3", "1", "2", "3"],
                }
            ),
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["1", "2", "3"],
                    "A": ["1", "2", "3"],
                    "B": ["1", "2", "3"],
                    "C": ["1", "2", "3"],
                }
            ),
            pl.DataFrame(
                {
                    "SampleID": ["4", "5", "6"],
                    "D": ["1", "2", "3"],
                    "E": ["1", "2", "3"],
                    "F": ["1", "2", "3"],
                }
            ),
            pl.exceptions.ShapeError,
        ),
    ],
)
def test_merge_query_and_reference(query, reference, expected):
    """Test Concatenation of query and reference profiles.

    This test verifies that a shape error is thrown if columns
    do not match. However mis-matching columns is handled prior to
    this process in the match.py main function.
    """
    if expected is pl.exceptions.ShapeError:
        with pytest.raises(expected):
            output: pl.DataFrame = match.merge_query_and_reference(query, reference)
    else:
        output: pl.DataFrame = match.merge_query_and_reference(query, reference)
        assert output.equals(expected)


@pytest.mark.parametrize(
    "input,sample_names,scaled,expected",
    [
        (
            np.array([[0, 1, 3], [0, 2, 6], [0, 3, 9]], dtype=np.float32),
            ["1", "2", "3", "4"],
            True,
            ["query_id\tref_id\tdist_scaled", "1\t2\t3.0", "1\t3\t6.0", "1\t4\t9.0", ""],
        ),
        (
            np.array([[0, 1, 3], [0, 2, 6], [0, 3, 9]], dtype=np.float32),
            ["1", "2", "3", "4"],
            False,
            ["query_id\tref_id\tdist_hamming", "1\t2\t3", "1\t3\t6", "1\t4\t9", ""],
        ),
    ],
)
def test_prepare_fast_match_outputs(tmp_path, input, sample_names, scaled, expected):
    """Write out calculated fast-match outputs."""
    file_out = tmp_path / "output.tsv"

    @dataclass
    class MatchArguments:
        output: Path
        scaled: bool
        delimiter: str

    input_args = MatchArguments(output=file_out, scaled=scaled, delimiter="\t")
    match.prepare_fast_match_outputs(input, sample_names, input_args)  # type: ignore[reportArgumentType]
    text = file_out.read_text().split("\n")
    assert text == expected


@pytest.mark.parametrize(
    "profiles,query_size,match_args,expected",
    [
        (
            np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.float32),
            1,
            match.MatchArguments(Path(""), Path(""), 1.0, 1, None, "\t", True, True, 0.0, Path("")),
            np.array([[0, 1, 0.0], [0, 2, 0.0]], dtype=np.float32),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ],
                dtype=np.float32,
            ),
            1,
            match.MatchArguments(
                Path(""), Path(""), 100.0, 1, None, "\t", True, True, 0.0, Path("")
            ),
            np.array(
                [
                    [0, 1, 0.0],
                    [0, 2, 10.0],
                    [0, 3, 20.0],
                    [0, 4, 30.0],
                    [0, 5, 40.0],
                    [0, 6, 50.0],
                    [0, 7, 60.0],
                    [0, 8, 70.0],
                    [0, 9, 80.0],
                    [0, 10, 90.0],
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ],
                dtype=np.float32,
            ),
            2,
            match.MatchArguments(
                Path(""), Path(""), 100.0, 1, None, "\t", True, True, 0.0, Path("")
            ),
            np.array(
                [
                    [0, 1, 0.0],
                    [0, 2, 10.0],
                    [0, 3, 20.0],
                    [0, 4, 30.0],
                    [0, 5, 40.0],
                    [0, 6, 50.0],
                    [0, 7, 60.0],
                    [0, 8, 70.0],
                    [0, 9, 80.0],
                    [0, 10, 90.0],
                    [1, 2, 10.0],
                    [1, 3, 20.0],
                    [1, 4, 30.0],
                    [1, 5, 40.0],
                    [1, 6, 50.0],
                    [1, 7, 60.0],
                    [1, 8, 70.0],
                    [1, 9, 80.0],
                    [1, 10, 90.0],
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ],
                dtype=np.float32,
            ),
            1,
            match.MatchArguments(
                Path(""), Path(""), 100.0, 1, None, "\t", True, False, 0.0, Path("")
            ),
            np.array(
                [
                    [0, 1, 0.0],
                    [0, 2, 1.0],
                    [0, 3, 2.0],
                    [0, 4, 3.0],
                    [0, 5, 4.0],
                    [0, 6, 5.0],
                    [0, 7, 6.0],
                    [0, 8, 7.0],
                    [0, 9, 8.0],
                    [0, 10, 9.0],
                ],
                dtype=np.float32,
            ),
        ),
        (
            np.array(
                [
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                    [1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
                    [1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
                    [1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                    [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
                    [1, 1, 1, 1, 1, 1, 2, 2, 2, 2],
                    [1, 1, 1, 1, 1, 1, 1, 2, 2, 2],
                    [1, 1, 1, 1, 1, 1, 1, 1, 2, 2],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 2],
                ],
                dtype=np.float32,
            ),
            1,
            match.MatchArguments(
                Path(""), Path(""), 100.0, 1, None, "\t", False, False, 0.0, Path("")
            ),
            np.array(
                [
                    [0, 1, 0.0],
                    [0, 2, 1.0],
                    [0, 3, 2.0],
                    [0, 4, 3.0],
                    [0, 5, 4.0],
                    [0, 6, 5.0],
                    [0, 7, 6.0],
                    [0, 8, 7.0],
                    [0, 9, 8.0],
                    [0, 10, 9.0],
                ],
                dtype=np.float32,
            ),
        ),
    ],
)
def test_run_fast_matching(profiles, query_size, match_args, expected):
    """Test for fast-matching outpouts."""
    output = match.run_fast_matching(profiles, query_size, match_args)
    np.testing.assert_allclose(
        output,
        expected,
    )  # floating point errors require allclose to be used


@pytest.mark.parametrize(
    "input,profile_width,expected_header",
    [
        (
            match.MatchArguments(
                Path("src/beave/tests/data/R1KC1K.head.tsv"),
                Path("src/beave/tests/data/R1KC1K.tail.tsv"),
                float("inf"),
                0,
                None,
                "\t",
                True,
                False,
                100.0,
                Path(""),
            ),
            1000,
            "query_id\tref_id\tdist_hamming",
        ),
        (
            match.MatchArguments(
                Path("src/beave/tests/data/R1KC1K.head.tsv"),
                Path("src/beave/tests/data/R1KC1K.tail.tsv"),
                float("inf"),
                2,
                None,
                "\t",
                True,
                True,
                100.0,
                Path(""),
            ),
            1000,
            "query_id\tref_id\tdist_scaled",
        ),
        (
            match.MatchArguments(
                Path("src/beave/tests/data/R1KC1K.head.tsv"),
                Path("src/beave/tests/data/R1KC1K.tail.tsv"),
                80.0,
                3,
                None,
                "\t",
                True,
                True,
                100.0,
                Path(""),
            ),
            1000,
            "query_id\tref_id\tdist_scaled",
        ),
        (
            match.MatchArguments(
                Path("src/beave/tests/data/R1KC1K.head.tsv"),
                Path("src/beave/tests/data/R1KC1K.tail.tsv"),
                float("inf"),
                3,
                None,
                "\t",
                True,
                False,
                200.0,
                Path(""),
            ),
            1000,
            "query_id\tref_id\tdist_hamming",
        ),
    ],
)
def test_match(monkeypatch, tmp_path, input, profile_width, expected_header):
    """Test of main match function."""
    output = tmp_path / "output.tsv"

    input.output = output
    monkeypatch.setattr(match, "MAX_ROWS_WRITE_BATCH", 10_000)
    match.match(input)
    data = [i for i in output.read_text().split("\n") if i != ""]
    if input.threshold is float("inf"):
        expected_output_size_no_filtering = 374250
        """
        Tests without filtering have an input size of 499 and a reference query size
        of 501. The expected number of comparisons can be calculated by first calculating
        the number of comparisons of all query samples against themselves.


        Query vs Query comparisons = (499(499-1))/2 # result is a triangle number
        Query vs Reference comparisons = 499 * 501 # not divided by two, as no self comparisons made

        Both numbers are summed:
        (499 * (499 - 1)) / 2 = 124,251
        (499 * 501) = 249,999
        number of comparisons = 124,251 + 249,999

        This assertion is not tested for all test inputs, as incorporating a threshold
        into the calculation is a more complicated.
        """
        assert len(data) - 1 == expected_output_size_no_filtering

    assert data[0] == expected_header
    for row in data[1:]:
        if not row:
            continue
        q, r, dist = row.split("\t")
        q = int(q)
        r = int(r)
        dist = float(dist)
        if input.scaled:
            expected = (abs(q - r) / profile_width) * 100.0
        else:
            expected = float(dist)
        assert dist <= input.threshold
        assert expected == pytest.approx(float(dist), rel=1e-6)


@pytest.mark.parametrize(
    "input",
    [
        match.MatchArguments(
            Path("src/beave/tests/data/HashesOnlyQuery.csv"),
            Path("src/beave/tests/data/HashesOnlyReference.csv"),
            float("inf"),
            0,
            None,
            ",",
            True,
            False,
            100.0,
            Path(""),
        ),
    ],
)
def test_match_all_match(tmp_path, input):
    """Verify the outputs match is correct with non-uniform inputs."""
    output = tmp_path / "output.tsv"
    input.output = output
    match.match(input)
    data = [i for i in output.read_text().split("\n") if i != ""]
    outputs_column_width = 299
    for row in data[1:]:
        q, r, dist = row.split(input.delimiter)
        hashed_reference_value = hashlib.md5(r.encode("utf8")).hexdigest()
        dist = int(dist)
        if q == hashed_reference_value:
            assert dist == 0
        else:
            assert dist == outputs_column_width


@pytest.mark.workflow("Run fast-matching")
def test_fast_match_run_outputs(workflow_dir):
    """Verify output of fast matching workflow test."""
    output_file = Path(workflow_dir, "output.tsv")
    assert output_file.exists()
    data = output_file.read_text().split("\n")
    assert data[0] == "query_id\tref_id\tdist_hamming"
    query_sample_ids = Path(workflow_dir, "src", "beave", "tests", "data", "R1KC1K.tail.tsv")
    query_ids = {
        int(i.split("\t")[0]) for i in query_sample_ids.read_text().split("\n")[1:] if i != ""
    }
    query_ids_read = set()
    for row in data[1:]:
        if not row:
            continue
        q, r, dist = row.split("\t")
        q = int(q)
        r = int(r)
        query_ids_read.add(q)
        dist = float(dist)
        assert dist == float(abs(q - r))
    assert query_ids_read == query_ids


@pytest.mark.workflow("Run fast-matching scaled")
def test_fast_match_run_outputs_scaled(workflow_dir):
    """Verify output of fast matching is correct with scaled outputs."""
    output_file = Path(workflow_dir, "output.tsv")
    assert output_file.exists()
    data = [i for i in output_file.read_text().split("\n") if i != ""]
    assert data[0] == "query_id\tref_id\tdist_scaled"
    query_sample_ids = Path(workflow_dir, "src", "beave", "tests", "data", "R1KC1K.sorted.head.tsv")
    reference_sample_ids = Path(
        workflow_dir, "src", "beave", "tests", "data", "R1KC1K.sorted.tail.tsv"
    )
    query_ids = {
        int(i.split("\t")[0]) for i in query_sample_ids.read_text().split("\n")[1:] if i != ""
    }
    ref_ids = {
        int(i.split("\t")[0]) for i in reference_sample_ids.read_text().split("\n")[1:] if i != ""
    }

    expected_labels = set()
    for v1 in range(0, len(query_ids)):
        for v2 in range(v1 + 1, len(ref_ids) + len(query_ids)):
            expected_labels.add((v1, v2))

    assert (
        len(expected_labels) == len(data) - 1
    )  # verify the correct number of comparisons are made

    query_ids_read = set()
    values = set()
    for row in data[1:]:
        if not row:
            continue
        q, r, dist = row.split("\t")
        q = int(q)
        r = int(r)
        query_ids_read.add(q)
        values.add((q, r))
        expected = float(abs(q - r) / 1000) * 100.0
        assert expected == pytest.approx(float(dist), rel=1e-6)
    assert query_ids_read == query_ids
    assert values == expected_labels  # verify the two sets are the same


@pytest.mark.workflow("Run fast-matching subset columns")
def test_fast_match_subset_columns(workflow_dir):
    """Verify output of fast matching is correct with scaled outputs.

    I have created a set of columns to pass to the program so that only the first 10 columns
    are saved.
    """
    output_file = Path(workflow_dir, "output.tsv")
    assert output_file.exists()
    data = output_file.read_text().split("\n")
    max_dist_possible = 10.0
    assert data[0] == "query_id\tref_id\tdist_hamming"
    for row in data[1:]:
        if not row:
            continue
        q, r, dist = row.split("\t")
        q = float(q)
        r = float(r)
        dist = float(dist)

        if q <= max_dist_possible and r <= max_dist_possible:
            assert dist == abs(q - r)
        elif q <= max_dist_possible or r <= max_dist_possible:
            assert dist == (max_dist_possible - min(q, r))
        else:
            assert dist == 0.0
