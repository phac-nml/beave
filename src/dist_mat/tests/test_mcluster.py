import pytest

import pathlib as p

import dist_mat
from dist_mat import mcluster

import polars as pl
from polars.testing.parametric import dataframes, column
import numpy as np
from numpy import typing as npt
import scipy
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.extra import numpy as nps


@pytest.mark.parametrize(
    "input,columns_keep,delimiter,threads,expected",
    [
        (
            p.Path("src/dist_mat/tests/data/simple_test_profiles.csv"),
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
            p.Path("src/dist_mat/tests/data/simple_test_profiles.tsv"),
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
    Tests for loading of the input profiles
    """

    input_profiles = mcluster.read_input_profiles(
        input, columns_keep, delimiter, threads
    )
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
        ),
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
            0.00,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": [111, 111, 111, 111],
                    "b": [111, 111, 0, 111],
                    "c": [111, 111, 0, 111],
                    "d": [111, 111, 0, 111],
                }
            ),
        ),
        (
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": [111, 111, 111, 111],
                    "b": [111, 111, 111, 111],
                    "c": [111, 111, 111, 111],
                    "d": [111, 111, 111, 111],
                },
            ),
            1.00,
            pl.DataFrame(
                {
                    "SampleID": ["a", "b", "c", "d"],
                    "A": [111, 111, 111, 111],
                    "b": [111, 111, 111, 111],
                    "c": [111, 111, 111, 111],
                    "d": [111, 111, 111, 111],
                }
            ),
        ),
    ],
)
def test_filter_rows(dataframe, threshold, expected) -> None:
    filtered_data = mcluster.filter_rows(dataframe, threshold)
    assert filtered_data.equals(expected)


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
    output_path = tmp_path / "cols_keep.txt"
    output_path.write_text("SampleID\nSubset1\nSubset2\n")
    subset = mcluster.subset_columns(df, output_path)
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
                strategy=st.sampled_from(list(mcluster.REPLACE_CHARS.keys())),
            ),
            column(
                "QMarks2",
                dtype=pl.String,
                strategy=st.sampled_from(list(mcluster.REPLACE_CHARS.keys())),
            ),
            column(
                "QMarks3",
                dtype=pl.String,
                strategy=st.sampled_from(list(mcluster.REPLACE_CHARS.keys())),
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
    output = mcluster.prep_data(profiles, 0.00)
    assert np.array_equal(output, array)


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
def test_transform_data(data, threshold, expected):
    out = mcluster.transform_data(data, threshold)
    assert out.equals(expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        (
            mcluster.LinkageMetrics.SINGLE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
        ),
        (
            mcluster.LinkageMetrics.COMPLETE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 2.82842712, 3]], dtype=float),
        ),
        (
            mcluster.LinkageMetrics.AVERAGE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 2.12132034, 3]], dtype=float),
        ),
    ],
)
def test_compute_linkage_matrix(method, expected):
    input_array = np.array([1.41421356, 2.82842712, 1.41421356])
    output = mcluster.compute_linkage_matrix(input_array, method)
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
    actual_columns = mcluster.assign_clusters(linkage, thresholds, labels).columns
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
    out = mcluster.assign_clusters(linkage, thresholds, labels)
    assert out.equals(expected)


@pytest.mark.parametrize(
    "linkage,branchlength_type,expected",
    [
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            mcluster.BranchLengths.PATRISTIC,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
        ),
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            mcluster.BranchLengths.COPHENETIC,
            np.array([[0, 1, 2.82842712, 2], [2, 3, 2.82842712, 3]], dtype=float),
        ),
    ],
)
def test_convert_branch_lengths(linkage, branchlength_type, expected):
    assert np.allclose(
        mcluster.convert_branch_lengths(linkage, branchlength_type), expected
    )


@pytest.mark.parametrize(
    "profiles,count_missing,scaled,expected",
    [
        (
            np.array(
                [
                    [np.uint32(1), mcluster.MISSING_VALUE],
                    [mcluster.MISSING_VALUE, np.uint32(1)],
                ]
            ),
            False,
            True,
            np.array([np.float32(100.0)]),
        ),
        (
            np.array(
                [
                    [np.uint32(1), mcluster.MISSING_VALUE],
                    [mcluster.MISSING_VALUE, np.uint32(1)],
                ]
            ),
            True,
            True,
            np.array([np.float32(100.0)]),
        ),
        (
            np.array(
                [
                    [mcluster.MISSING_VALUE, mcluster.MISSING_VALUE],
                    [mcluster.MISSING_VALUE, mcluster.MISSING_VALUE],
                ]
            ),
            False,
            True,
            np.array([np.float32(100.0)]),
        ),
    ],
)
def test_calc_dists(profiles, count_missing, scaled, expected):
    output = dist_mat.calc_dists(profiles, 1, scaled, count_missing)
    assert np.array_equal(output, expected)


@given(
    arr=nps.arrays(
        dtype=np.uint32,
        shape=(1000, 100),
    )
)
def test_calc_dists_fuzzing_hypothesis(arr):
    output = np.isfinite(dist_mat.calc_dists(arr, 1, True, False))
    assert np.all(output)


@pytest.mark.parametrize(
    "input,scaled,count_missing",
    [
        (p.Path("tests/R1KC1K.tsv"), True, True),
        (p.Path("tests/R1KC1K.tsv"), False, True),
        (p.Path("tests/R1KC1K.tsv"), False, True),
    ],
)
def test_calc_dists_file_inputs(input, scaled, count_missing):
    profiles: pl.DataFrame = mcluster.read_input_profiles(input, None, "\t", 1)
    dists: npt.NDArray = mcluster.compute_dists(profiles, count_missing, scaled, 1)
    matrix: npt.NDArray = scipy.spatial.distance.squareform(
        dists
    )  # conversion to squareform so iteration of the matrix is simpler as we do not need to calculate the
    # column and row index from the output condensed array.
    #
    for i in range(0, profiles.height):
        sample1: int = int(profiles.item(i, "sample"))
        for f in range(0, profiles.height):
            sample2: int = int(profiles.item(f, "sample"))
            if scaled:
                dist: float = (abs(sample1 - sample2) / float(profiles.height)) * 100.0
                assert dist == pytest.approx(matrix[i][f], rel=1e-6)
            else:
                assert float(abs(sample1 - sample2)) == pytest.approx(
                    matrix[i][f], rel=1e-6
                )
