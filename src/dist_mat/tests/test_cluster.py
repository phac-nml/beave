"""Tests for cluter.py."""

import pytest  # noqa: I001

import dist_mat
from dist_mat import cluster
from dist_mat import transform_data as transform

import hashlib
from pathlib import Path

import polars as pl
from polars.testing.parametric import dataframes, column
import numpy as np
from numpy import typing as npt
import scipy
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.extra import numpy as nps


@pytest.fixture(scope="function")
def test_df() -> pl.DataFrame:
    """Example dataframe for the benchmark function."""
    return pl.DataFrame(
        {
            str(k): [hashlib.md5(str(i).encode("utf8")).hexdigest() for i in range(1000)]
            for k in range(300)
        }
    )


def test_benchmark_data_transformation_hashes(benchmark, test_df):
    """Benchmarks for different data transformation methods."""
    benchmark(transform.transform_data_hashes, test_df, 1.00)
    assert True


@pytest.mark.parametrize(
    "input,delimiter,threads,expected",
    [
        (
            Path("src/dist_mat/tests/data/simple_test_profiles.csv"),
            ",",
            1,
            pl.DataFrame(
                {
                    "SampleID": [str(1), str(2), str(3)],
                    "A": [str(1), str(4), str(7)],
                    "B": [str(2), str(5), str(8)],
                    "C": [str(3), str(6), str(9)],
                }
            ),
        ),
        (
            Path("src/dist_mat/tests/data/simple_test_profiles.tsv"),
            "\t",
            1,
            pl.DataFrame(
                {
                    "SampleID": ["1", "2", "3"],
                    "A": [str(1), str(4), str(7)],
                    "B": [None, str(5), str(8)],
                    "C": [str(3), str(6), str(9)],
                },
                strict=False,
            ),
        ),
    ],
)
def test_read_input_profiles(input, delimiter, threads, expected) -> None:
    """Tests for loading of the input profiles."""
    input_profiles = transform.read_input_profiles(input, delimiter, threads)
    assert input_profiles.equals(expected)


@pytest.mark.parametrize(
    "dataframe,threshold,expected,error",
    [
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 111, 111, 111],
                    "b": [111, 111, 0, 111],
                    "c": [111, 111, 0, 111],
                    "d": [111, 111, 0, 111],
                    "e": [111, 111, 0, 111],
                    "f": [111, 111, 0, 111],
                    "g": [111, 111, 0, 111],
                    "h": [111, 111, 0, 111],
                    "i": [111, 111, 0, 111],
                    "j": [111, 111, 0, 111],
                },
            ),
            0.89,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "d"],
                    "a": [111, 111, 111],
                    "b": [111, 111, 111],
                    "c": [111, 111, 111],
                    "d": [111, 111, 111],
                    "e": [111, 111, 111],
                    "f": [111, 111, 111],
                    "g": [111, 111, 111],
                    "h": [111, 111, 111],
                    "i": [111, 111, 111],
                    "j": [111, 111, 111],
                }
            ),
            None,
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 111, 111, 111],
                    "b": [111, 111, 111, 111],
                    "c": [111, 111, 111, 111],
                    "d": [111, 111, 111, 111],
                    "e": [111, 111, 111, 111],
                    "f": [111, 111, 111, 111],
                    "g": [111, 111, 111, 111],
                    "h": [111, 111, 111, 111],
                    "i": [111, 111, 111, 111],
                    "j": [111, 111, 0, 111],
                },
            ),
            0.09,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "d"],
                    "a": [111, 111, 111],
                    "b": [111, 111, 111],
                    "c": [111, 111, 111],
                    "d": [111, 111, 111],
                    "e": [111, 111, 111],
                    "f": [111, 111, 111],
                    "g": [111, 111, 111],
                    "h": [111, 111, 111],
                    "i": [111, 111, 111],
                    "j": [111, 111, 111],
                }
            ),
            None,
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 111, 111, 111],
                    "b": [111, 111, 0, 111],
                    "c": [111, 111, 0, 111],
                    "d": [111, 111, 0, 111],
                    "e": [111, 111, 0, 111],
                    "f": [111, 111, 0, 111],
                },
            ),
            0.75,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "d"],
                    "a": [111, 111, 111],
                    "b": [111, 111, 111],
                    "c": [111, 111, 111],
                    "d": [111, 111, 111],
                    "e": [111, 111, 111],
                    "f": [111, 111, 111],
                }
            ),
            None,
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 111, 111, 111],
                    "b": [111, 111, 0, 111],
                    "c": [111, 111, 0, 111],
                    "d": [111, 111, 0, 111],
                },
            ),
            1.00,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 111, 111, 111],
                    "b": [111, 111, 0, 111],
                    "c": [111, 111, 0, 111],
                    "d": [111, 111, 0, 111],
                }
            ),
            None,
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 111, 111, 111],
                    "b": [111, 111, 0, 111],
                    "c": [111, 111, 0, 111],
                    "d": [111, 111, 0, 111],
                },
            ),
            1.00,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 111, 111, 111],
                    "b": [111, 111, 0, 111],
                    "c": [111, 111, 0, 111],
                    "d": [111, 111, 0, 111],
                }
            ),
            None,
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 111, 111, 111],
                    "b": [111, 111, 111, 111],
                    "c": [111, 111, 111, 111],
                    "d": [111, 111, 111, 111],
                },
            ),
            0.00,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 111, 111, 111],
                    "b": [111, 111, 111, 111],
                    "c": [111, 111, 111, 111],
                    "d": [111, 111, 111, 111],
                }
            ),
            None,
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 111, 111, 111],
                    "b": [111, 111, 0, 111],
                    "c": [111, 111, 111, 111],
                    "d": [111, 111, 111, 111],
                },
            ),
            0.00,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "d"],
                    "a": [111, 111, 111],
                    "b": [111, 111, 111],
                    "c": [111, 111, 111],
                    "d": [111, 111, 111],
                }
            ),
            None,
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "a": [111, 0, 111, 111],
                    "b": [111, 111, 0, 111],
                    "c": [0, 111, 111, 111],
                    "d": [111, 111, 111, 0],
                },
            ),
            0.00,
            pl.DataFrame(
                {
                    "SampleID": [],
                    "a": [],
                    "b": [],
                    "c": [],
                    "d": [],
                }
            ),
            transform.AllColumnsFilteredError,
        ),
    ],
)
def test_filter_rows(dataframe, threshold, expected, error) -> None:
    """Test that filtering of rows is correct."""
    if error is None:
        filtered_data = transform.filter_rows(dataframe, threshold)
        assert filtered_data.equals(expected)
    else:
        with pytest.raises(error):
            filtered_data = transform.filter_rows(dataframe, threshold)


@given(
    df=dataframes(
        include_cols=[
            column("Subset1", dtype=pl.UInt32),
            column("Subset2", dtype=pl.UInt32),
        ],
        min_size=2,
        max_size=100,
        max_cols=50,
        min_cols=2,
    )
)
@settings(
    suppress_health_check=[
        HealthCheck.function_scoped_fixture,
        HealthCheck.too_slow,
        HealthCheck.data_too_large,
        HealthCheck.filter_too_much,
    ],
    max_examples=10,
)
def test_subset_columns(df: pl.DataFrame, tmp_path) -> None:
    """Tests for subsetting of columns."""
    output_path = tmp_path / "cols_keep.txt"
    output_path.write_text("SampleID\nSubset1\nSubset2\n")
    subset = transform.subset_columns(df, output_path, None)
    assert subset.columns[0] == "col0"  # Leftmost column should always be first
    assert set(subset.columns) == set(
        ["col0", "Subset1", "Subset2"]
    )  # set as order does not matter


@settings(derandomize=True)
@given(
    profiles=dataframes(
        cols=[
            column(
                "SampleID",
                dtype=pl.String,
                strategy=st.sampled_from(list(transform.REPLACE_CHARS.keys())),
            ),
            column(
                "QMarks2",
                dtype=pl.String,
                strategy=st.sampled_from(list(transform.REPLACE_CHARS.keys())),
            ),
            column(
                "QMarks3",
                dtype=pl.String,
                strategy=st.sampled_from(list(transform.REPLACE_CHARS.keys())),
            ),
            column(
                "Hashed",
                dtype=pl.String,
                strategy=st.sampled_from(
                    ["A"]
                ),  # 2683474508 -> A hashed and converted to np.uint32
            ),
        ],
        min_size=2,
        max_size=1000,
        allow_null=False,
    )
)
def test_prep_data(profiles: pl.DataFrame) -> None:
    """Tests for tranfomation of data."""
    array = np.zeros((profiles.height, 3), dtype=np.uint32)  # array should all be zeros
    for i in array:
        i[2] = np.uint32(2683474508)  # last value should be the hashed version of "A"
    output = cluster.prep_data(profiles, 1.00, transform.transform_data_hashes)
    np.testing.assert_equal(output, array)


@pytest.mark.parametrize(
    "data,threshold,expected",
    [
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": ["2", "2", "2", "2"],
                    "b": ["2", "2", None, "2"],
                    "c": ["2", "2", None, "2"],
                    "d": ["2", "2", None, "2"],
                    "e": ["2", "2", None, "2"],
                    "f": ["2", "2", None, "2"],
                    "g": ["2", "2", None, "2"],
                },
                schema={
                    "SampleID": pl.String,
                    "A": pl.Categorical,
                    "b": pl.Categorical,
                    "c": pl.Categorical,
                    "d": pl.Categorical,
                    "e": pl.Categorical,
                    "f": pl.Categorical,
                    "g": pl.Categorical,
                },
            ),
            1.00,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": [
                        1,  # value corrends to a hashed "2"
                        1,
                        1,
                        1,
                    ],
                    "b": [
                        1,
                        1,
                        0,
                        1,
                    ],
                    "c": [
                        1,
                        1,
                        0,
                        1,
                    ],
                    "d": [
                        1,
                        1,
                        0,
                        1,
                    ],
                    "e": [
                        1,
                        1,
                        0,
                        1,
                    ],
                    "f": [
                        1,
                        1,
                        0,
                        1,
                    ],
                    "g": [
                        1,
                        1,
                        0,
                        1,
                    ],
                },
            ),
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": ["2", "2", "2", "2"],
                    "b": ["2", "2", None, "2"],
                    "c": ["2", "2", None, "2"],
                    "d": ["2", "2", None, "2"],
                    "e": ["2", "2", None, "2"],
                    "f": ["2", "2", None, "2"],
                    "g": ["2", "2", None, "2"],
                },
                schema={
                    "SampleID": pl.String,
                    "A": pl.Categorical,
                    "b": pl.Categorical,
                    "c": pl.Categorical,
                    "d": pl.Categorical,
                    "e": pl.Categorical,
                    "f": pl.Categorical,
                    "g": pl.Categorical,
                },
            ),
            0.00,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "d"],
                    "A": [
                        1,  # value corrends to a hashed "2"
                        1,
                        1,
                    ],
                    "b": [
                        1,
                        1,
                        1,
                    ],
                    "c": [
                        1,
                        1,
                        1,
                    ],
                    "d": [
                        1,
                        1,
                        1,
                    ],
                    "e": [
                        1,
                        1,
                        1,
                    ],
                    "f": [
                        1,
                        1,
                        1,
                    ],
                    "g": [
                        1,
                        1,
                        1,
                    ],
                },
            ),
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": ["2", "2", "2", "2"],
                    "b": ["2", "2", None, "2"],
                    "c": ["2", "2", None, "2"],
                    "d": ["2", "2", None, "2"],
                    "e": ["2", "2", None, "2"],
                    "f": ["2", "2", None, "2"],
                    "g": ["2", "2", None, "2"],
                },
                schema={
                    "SampleID": pl.String,
                    "A": pl.Categorical,
                    "b": pl.Categorical,
                    "c": pl.Categorical,
                    "d": pl.Categorical,
                    "e": pl.Categorical,
                    "f": pl.Categorical,
                    "g": pl.Categorical,
                },
            ),
            0.25,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "d"],
                    "A": [
                        1,  # value corrends to a hashed "2"
                        1,
                        1,
                    ],
                    "b": [
                        1,
                        1,
                        1,
                    ],
                    "c": [
                        1,
                        1,
                        1,
                    ],
                    "d": [
                        1,
                        1,
                        1,
                    ],
                    "e": [
                        1,
                        1,
                        1,
                    ],
                    "f": [
                        1,
                        1,
                        1,
                    ],
                    "g": [
                        1,
                        1,
                        1,
                    ],
                },
            ),
        ),
    ],
)
def test_transform_data(data, threshold, expected):
    """Tests for mapping tranformation and filtering of data."""
    out = transform.transform_data_categorical_encoding(data, threshold)
    assert out.shape == expected.shape  # verify shape as map values will change on each run


@pytest.mark.parametrize(
    "data,threshold,expected",
    [
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": ["2", "2", "2", "2"],
                    "b": ["2", "2", "?", "2"],
                    "c": ["2", "2", "", "2"],
                    "d": ["2", "2", " ", "2"],
                    "e": ["2", "2", "_", "2"],
                    "f": ["2", "2", "-", "2"],
                    "g": ["2", "2", "0", "2"],
                },
            ),
            1.00,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": [
                        10486959400714174283,  # value corrends to a hashed "2"
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "b": [
                        10486959400714174283,
                        10486959400714174283,
                        0,
                        10486959400714174283,
                    ],
                    "c": [
                        10486959400714174283,
                        10486959400714174283,
                        0,
                        10486959400714174283,
                    ],
                    "d": [
                        10486959400714174283,
                        10486959400714174283,
                        0,
                        10486959400714174283,
                    ],
                    "e": [
                        10486959400714174283,
                        10486959400714174283,
                        0,
                        10486959400714174283,
                    ],
                    "f": [
                        10486959400714174283,
                        10486959400714174283,
                        0,
                        10486959400714174283,
                    ],
                    "g": [
                        10486959400714174283,
                        10486959400714174283,
                        0,
                        10486959400714174283,
                    ],
                },
            ),
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": ["2", "2", "2", "2"],
                    "b": ["2", "2", "?", "2"],
                    "c": ["2", "2", "", "2"],
                    "d": ["2", "2", " ", "2"],
                    "e": ["2", "2", "_", "2"],
                    "f": ["2", "2", "-", "2"],
                    "g": ["2", "2", "0", "2"],
                },
            ),
            0.00,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "d"],
                    "A": [
                        10486959400714174283,  # value corrends to a hashed "2"
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "b": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "c": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "d": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "e": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "f": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "g": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                },
            ),
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": ["2", "2", "2", "2"],
                    "b": ["2", "2", "?", "2"],
                    "c": ["2", "2", "", "2"],
                    "d": ["2", "2", " ", "2"],
                    "e": ["2", "2", "_", "2"],
                    "f": ["2", "2", "-", "2"],
                    "g": ["2", "2", "0", "2"],
                },
            ),
            0.25,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "d"],
                    "A": [
                        10486959400714174283,  # value corrends to a hashed "2"
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "b": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "c": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "d": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "e": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "f": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                    "g": [
                        10486959400714174283,
                        10486959400714174283,
                        10486959400714174283,
                    ],
                },
            ),
        ),
    ],
)
def test_transform_data_hashes(data, threshold, expected):
    """Tests for hashing of data."""
    out = transform.transform_data_hashes(data, threshold)
    assert out.equals(expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        (
            cluster.LinkageMetric.SINGLE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
        ),
        (
            cluster.LinkageMetric.COMPLETE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 2.82842712, 3]], dtype=float),
        ),
        (
            cluster.LinkageMetric.AVERAGE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 2.12132034, 3]], dtype=float),
        ),
    ],
)
def test_compute_linkage_matrix(method, expected):
    """Tests from scipy for computing linkage matrix."""
    input_array = np.array([1.41421356, 2.82842712, 1.41421356])
    output = cluster.compute_linkage_matrix(input_array, method)
    np.testing.assert_allclose(output, expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        (
            cluster.LinkageMetric.COMPLETE,
            [[0.0, 1.0, 1.0, 2.0], [2.0, 4.0, 4.0, 3.0], [3.0, 5.0, 8.0, 4.0]],
        ),
        (
            cluster.LinkageMetric.SINGLE,
            [[0.0, 1.0, 1.0, 2.0], [2.0, 4.0, 3.0, 3.0], [3.0, 5.0, 5.0, 4.0]],
        ),
    ],
)
def test_compute_linkage_matrix_integers(method, expected):
    """Tests for computation of linkage matrix by scipy using a known input.

    Average linkage is not tested with integers as decimals are created due to
    proportional averaging being required.
    """
    input_array = np.array([1, 3, 8, 4, 7, 5])
    output = cluster.compute_linkage_matrix(input_array, method)
    np.testing.assert_equal(expected, output)


@pytest.mark.parametrize(
    "linkage,thresholds,labels,expected_columns",
    [
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            [1.0],
            ["A", "B", "C"],
            ["SampleID", "level_1.0", "denovo_address"],
        ),
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            [2.0, 1.0, 0.0],
            ["A", "B", "C"],
            ["SampleID", "level_2.0", "level_1.0", "level_0.0", "denovo_address"],
        ),
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            [float(i) for i in range(100, 0, -1)],
            ["A", "B", "C"],
            [
                "SampleID",
                *[f"level_{float(i)}" for i in range(100, 0, -1)],
                "denovo_address",
            ],
        ),
    ],
)
def test_assign_clusters_columns(linkage, thresholds, labels, expected_columns):
    """Tests that outputs of assign clusters are correct."""
    actual_columns = cluster.assign_clusters(linkage, thresholds, labels).columns
    assert actual_columns == expected_columns


@pytest.mark.parametrize(
    "linkage,thresholds,labels,expected",
    [
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            [4.0],
            ["A", "B", "C"],
            pl.DataFrame(
                {
                    "SampleID": ["A", "B", "C"],
                    "level_4.0": [1, 1, 1],
                    "denovo_address": ["1", "1", "1"],
                }
            ),
        ),
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            [4.0, 0.01],
            ["A", "B", "C"],
            pl.DataFrame(
                {
                    "SampleID": ["A", "B", "C"],
                    "level_4.0": [1, 1, 1],
                    "level_0.01": [1, 2, 3],
                    "denovo_address": ["1.1", "1.2", "1.3"],
                }
            ),
        ),
    ],
)
def test_assign_clusters_(linkage, thresholds, labels, expected):
    """Test of fcluster assignments."""
    out = cluster.assign_clusters(linkage, thresholds, labels)
    assert out.equals(expected)


@pytest.mark.parametrize(
    "linkage,branchlength_type,expected",
    [
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            cluster.BranchLengthType.COPHENETIC,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
        ),
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            cluster.BranchLengthType.PATRISTIC,
            np.array([[0, 1, 0.70710678, 2], [2, 3, 0.70710678, 3]], dtype=float),
        ),
        (
            np.array([[0, 1, 3, 2], [2, 3, 3, 3]], dtype=float),
            cluster.BranchLengthType.PATRISTIC,
            np.array([[0, 1, 1.5, 2], [2, 3, 1.5, 3]], dtype=float),
        ),
        (
            np.array([[0, 1, 3, 2], [2, 3, 3, 3]], dtype=float),
            cluster.BranchLengthType.COPHENETIC,
            np.array([[0, 1, 3, 2], [2, 3, 3, 3]], dtype=float),
        ),
        (
            np.array([[0, 1, 0, 2], [2, 3, 3, 3]], dtype=float),
            cluster.BranchLengthType.PATRISTIC,
            np.array([[0, 1, 0, 2], [2, 3, 1.5, 3]], dtype=float),
        ),
    ],
)
def test_convert_branch_lengths(linkage, branchlength_type, expected):
    """Test for conversiong of branchlengths to cophenetic and patristic distances."""
    assert np.allclose(cluster.convert_branch_lengths(linkage, branchlength_type), expected)


@pytest.mark.parametrize(
    "profiles,count_missing,scaled,expected",
    [
        (
            np.array(
                [
                    [np.uint32(1), transform.MISSING_VALUE],
                    [transform.MISSING_VALUE, np.uint32(1)],
                ]
            ),
            False,
            True,
            np.array([np.float32(100.0)]),
        ),
        (
            np.array(
                [
                    [np.uint32(1), transform.MISSING_VALUE],
                    [transform.MISSING_VALUE, np.uint32(1)],
                ]
            ),
            True,
            True,
            np.array([np.float32(100.0)]),
        ),
        (
            np.array(
                [
                    [transform.MISSING_VALUE, transform.MISSING_VALUE],
                    [transform.MISSING_VALUE, transform.MISSING_VALUE],
                ]
            ),
            False,
            True,
            np.array([np.float32(100.0)]),
        ),
    ],
)
def test_calc_dists(profiles, count_missing, scaled, expected):
    """Test distance calculation output is correct."""
    output = dist_mat.calc_dists(profiles, 1, scaled, count_missing)
    np.testing.assert_equal(output, expected)


@given(
    arr=nps.arrays(
        dtype=np.uint32,
        shape=(1000, 100),
    )
)
def test_calc_dists_fuzzing_hypothesis_no_infinites(arr):
    """Tests to make sure calc_dists always returns a finite answer."""
    output = np.isfinite(dist_mat.calc_dists(arr, 1, True, False))
    assert np.all(output)


@pytest.mark.parametrize(
    "input,scaled,count_missing",
    [
        (Path("tests/R1KC1K.tsv"), True, True),
        (Path("tests/R1KC1K.tsv"), False, True),
        (Path("tests/R1KC1K.tsv"), False, True),
    ],
)
def test_calc_dists_file_inputs(input, scaled, count_missing):
    """Test inputs of calc dists is correct with known input."""
    profiles: pl.DataFrame = cluster.read_input_profiles(input, "\t", 1)
    dists: npt.NDArray = cluster.compute_dists(profiles, count_missing, scaled, 1)
    matrix: npt.NDArray = scipy.spatial.distance.squareform(
        dists
    )  # conversion to squareform so iteration of the matrix is simpler as we do not need to
    # calculate the column and row index from the output condensed array.
    for i in range(0, profiles.height):
        sample1: int = int(profiles.item(i, "sample"))
        for f in range(0, profiles.height):
            sample2: int = int(profiles.item(f, "sample"))
            if scaled:
                dist: float = (abs(sample1 - sample2) / float(profiles.height)) * 100.0
                assert dist == pytest.approx(matrix[i][f], rel=1e-6)
            else:
                assert float(abs(sample1 - sample2)) == pytest.approx(matrix[i][f], rel=1e-6)


@pytest.mark.parametrize(
    "linkage,sample_ids,expected",
    [
        (
            np.asarray([[0, 1, 3.0, 2], [3, 2, 4.0, 3]], dtype=np.float64),
            ["l0", "l1", "l2"],
            "((l0:3.0,l1:3.0):1.0,l2:4.0);",
        ),
        (
            np.asarray([[0, 1, 3.0, 2], [2, 3, 2.0, 2], [4, 5, 4.0, 4]], dtype=np.float64),
            ["l0", "l1", "l2", "l3"],
            "((l0:3.0,l1:3.0):1.0,(l2:2.0,l3:2.0):2.0);",
        ),
        (
            np.asarray(
                [[0, 1, 2.0, 2], [5, 3, 3.5, 2], [6, 4, 4.0, 2], [7, 2, 6.0, 4]], dtype=np.float64
            ),
            ["l0", "l1", "l2", "l3", "l4"],
            "((((l0:2.0,l1:2.0):1.5,l3:3.5):0.5,l4:4.0):2.0,l2:6.0);",
        ),
    ],
)
def test_linkage_matrix_to_nwk(linkage, sample_ids, expected):
    """Test for converting linkage matrix to a newick tree.

    These tests are adpated from the original pull request implementing
    the to newick fucntion in a scipy PR.

    https://github.com/scipy/scipy/pull/17329/changes
    """
    assert cluster.linkage_matrix_to_nwk(linkage, sample_ids) == expected


def test_get_subset_columns():
    """Test for get_subset_columns."""
    cols = transform.get_subset_columns(Path("src/dist_mat/tests/data/test_columns.txt"))
    assert cols == {"sample", "col1", "col2", "col3"}
