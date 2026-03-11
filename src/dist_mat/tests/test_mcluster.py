import pytest
import dist_mat.mcluster as mc
import polars as pl
from polars.testing.parametric import dataframes, column
from hypothesis import given, settings, HealthCheck, strategies as st
import numpy as np
import pathlib as p


@pytest.mark.parametrize(
    "input,columns_keep,delimiter,threads,expected",
    [
        (
            p.Path("src/dist_mat/tests/data/test_profiles.csv"),
            None,
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
            p.Path("src/dist_mat/tests/data/test_profiles.tsv"),
            None,
            "\t",
            1,
            pl.DataFrame(
                {
                    "SampleID": ["1", "2", "3"],
                    "A": [str(1), str(4), str(7)],
                    "B": ["", str(5), str(8)],
                    "C": [str(3), str(6), str(9)],
                },
                strict=False,
            ),
        ),
    ],
)
def test_read_input_profiles(input, columns_keep, delimiter, threads, expected) -> None:
    """
    Tests for ingestion of the input profiles

    TODO add tests for nulls and other types
    """

    input_profiles = mc.read_input_profiles(input, columns_keep, delimiter, threads)
    assert input_profiles.equals(expected)


@pytest.mark.parametrize(
    "dataframe,threshold,expected",
    [
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": [111, 111, 111, 111],
                    "b": [111, 111, 0, 111],
                    "c": [111, 111, 0, 111],
                    "d": [111, 111, 0, 111],
                },
            ),
            0.75,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "d"],
                    "A": [111, 111, 111],
                    "b": [111, 111, 111],
                    "c": [111, 111, 111],
                    "d": [111, 111, 111],
                }
            ),
        )
    ],
)
def test_filter_rows(dataframe, threshold, expected) -> None:
    assert mc.filter_rows(dataframe, threshold).equals(expected)


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
    d = tmp_path / "cols_keep.txt"
    d.write_text("SampleID\nSubset1\nSubset2\n")
    subset = mc.subset_columns(df, d)
    assert subset.columns[0] == "col0"  # Leftmost column should always be first
    assert set(subset.columns) == frozenset(
        ["col0", "Subset1", "Subset2"]
    )  # set as order does not matter


@given(
    profiles=dataframes(
        cols=[
            column(
                "SampleID",
                dtype=pl.String,
                strategy=st.sampled_from(list(mc.REPLACE_CHARS.keys())),
            ),
            column(
                "QMarks2",
                dtype=pl.String,
                strategy=st.sampled_from(list(mc.REPLACE_CHARS.keys())),
            ),
            column(
                "QMarks3",
                dtype=pl.String,
                strategy=st.sampled_from(list(mc.REPLACE_CHARS.keys())),
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
    array = np.zeros((profiles.height, 3), dtype=np.uint32)  # array should all be zeros
    for i in array:
        i[2] = np.uint32(2683474508)  # last value should be the hashed version of "A"
    output = mc.prep_data(profiles)
    assert np.array_equal(output, array)


@pytest.mark.parametrize(
    "method,expected",
    [
        (
            mc.LinkageMetrics.SINGLE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
        ),
        (
            mc.LinkageMetrics.COMPLETE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 2.82842712, 3]], dtype=float),
        ),
        (
            mc.LinkageMetrics.AVERAGE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 2.12132034, 3]], dtype=float),
        ),
    ],
)
def test_comp_linkage_matrix(method, expected):
    input_array = np.array([1.41421356, 2.82842712, 1.41421356])
    output = mc.comp_linkage_matrix(input_array, method)
    assert np.allclose(output, expected)


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
    output = mc.assign_clusters(linkage, thresholds, labels).columns
    assert output == expected_columns


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
    out = mc.assign_clusters(linkage, thresholds, labels)
    assert out.equals(expected)


@pytest.mark.parametrize(
    "linkage,bl_type,expected",
    [
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            mc.BranchLengths.PATRISTIC,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
        ),
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            mc.BranchLengths.COPHENETIC,
            np.array([[0, 1, 2.82842712, 2], [2, 3, 2.82842712, 3]], dtype=float),
        ),
    ],
)
def test_convert_branch_lengths(linkage, bl_type, expected):
    assert np.allclose(mc.convert_branch_lengths(linkage, bl_type), expected)
