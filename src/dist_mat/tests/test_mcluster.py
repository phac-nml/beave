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
                    "SampleID": [1, 2, 3],
                    "A": [1, 4, 7],
                    "B": [2, 5, 8],
                    "C": [3, 6, 9],
                }
            ),
        )
    ],
)
def test_read_input_profiles(input, columns_keep, delimiter, threads, expected) -> None:
    """
    Tests for ingestion of the input profiles

    TODO add tests for nulls and other types
    """
    assert mc.read_input_profiles(input, columns_keep, delimiter, threads).equals(
        expected
    )


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


# @st.composite
# def zero_charactars(draw):
#    return draw(st.lists(mc.REPLACE_CHARS.keys()))


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
        (
            mc.LinkageMetrics.CENTROID,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 2.12132034, 3]], dtype=float),
        ),
        (
            mc.LinkageMetrics.MEDIAN,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 2.12132034, 3]], dtype=float),
        ),
        (
            mc.LinkageMetrics.WARD,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 2.44948974, 3]], dtype=float),
        ),
    ],
)
def test_comp_linkage_matrix(method, expected):
    input_array = np.array([1.41421356, 2.82842712, 1.41421356])
    output = mc.comp_linkage_matrix(input_array, method)
    assert np.allclose(output, expected)
