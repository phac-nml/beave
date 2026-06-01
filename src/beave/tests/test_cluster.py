"""Tests for cluter.py."""

import pytest  # noqa: I001

import beave
from beave import cluster
from beave.declarations import BranchType, ClusterArguments, DefaultArguments, LinkageMetric
from beave import transform_data as transform

import hashlib
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from polars.testing.parametric import dataframes, column
import numpy as np
from numpy import typing as npt
import scipy
from hypothesis import given, settings, HealthCheck, strategies as st
from hypothesis.extra import numpy as nps


@pytest.fixture(scope="session")
def test_df() -> pl.DataFrame:
    """Example dataframe for the benchmark function."""
    return pl.DataFrame(
        {
            str(k): [hashlib.md5(str(i + k).encode("utf8")).hexdigest() for i in range(1000)]
            for k in range(800)
        }
    )


def test_benchmark_data_transformation_hashes(benchmark, test_df):
    """Benchmarks for different data transformation methods."""
    output_args = DefaultArguments("\t", True, True, None, 0.0, 1, Path(""))
    benchmark(transform.transform_data_hashes, test_df, 1.00, output_args)
    assert True


def test_benchmark_read_input_profiles(benchmark, test_df, tmp_path):
    """Benchmarks for different data transformation methods."""
    tmp_file = tmp_path / "output_data.tsv"
    test_df.write_csv(tmp_file, separator="\t")

    def read_input_profiles(input_file: Path, delimiter: str, threads: int):
        """Test case where each column is its own category."""
        profiles = pl.read_csv(
            input_file,
            separator=delimiter,
            n_threads=threads,
            has_header=True,
            raise_if_empty=True,
            infer_schema=False,
        )

        profiles = profiles.with_columns(
            pl.all()
            .exclude(profiles.columns[0])
            .replace(old=list(transform.REPLACE_CHARS.keys()), new=None)
        )
        # Remove rows which are all empty e.g. caused by new lines at the end of files
        profiles = profiles.filter(~pl.all_horizontal(pl.all().is_null()))
        """
        This expression creates a categorical mapping for each loci in the input file. Polars
        has deprecated the StringCache feature (1.41.0) in order to improve performance.
        They have also added namespaces and the ability to set the physical underlying data type
        of the categories used.

        Categories are global, meaning if a category has the same name, namespace
        and physical type they are the same.
        """
        profiles = profiles.with_columns(
            pl.col(col).cast(pl.Categorical(pl.Categories(physical=pl.UInt32, name=col)))
            for col in profiles.columns[1:]  # create categories dynamically
        )
        transform.verify_dataframe_integrity(profiles)

        return profiles

    def runtime_test():
        result = read_input_profiles(tmp_file, "\t", 2)
        cluster_args = ClusterArguments(
            delimiter="\t",
            cores=2,
            normalize_distance=True,
            count_missing=True,
            filter_threshold=1,
            columns_path=None,
            input_file=tmp_file,
            output_directory=tmp_path / "test",
            linkage_method=LinkageMetric.AVERAGE,
            thresholds=[0.99],
            branch_type=BranchType.COPHENETIC,
            matrix=False,
        )
        cluster.compute_dists(result, cluster_args)

    benchmark(runtime_test)

    assert True


def test_benchmark_read_input_profiles_simple_method(benchmark, test_df, tmp_path):
    """Benchmarks old read_input_profiles implementation from polars.

    This method is faster however it will cap the number of uniqure profiles to uint32 max
    across the whole dataframe, while the other method allows for uint32 max unique alleles per
    a column. I do not know if we would ever hit that limit in regular use but it is worth
    considering the limitation.
    """
    tmp_file = tmp_path / "output_data.tsv"
    test_df.write_csv(tmp_file, separator="\t")

    def read_input_profiles(input_file: Path, delimiter: str, threads: int) -> pl.DataFrame:
        category = pl.Categories(physical=pl.UInt32, name="test_32")
        profiles = pl.read_csv(
            input_file,
            separator=delimiter,
            n_threads=threads,
            has_header=True,
            raise_if_empty=True,
            infer_schema=False,
        )
        profiles = profiles.with_columns(
            pl.all()
            .exclude(profiles.columns[0])
            .replace(old=list(transform.REPLACE_CHARS.keys()), new=None)
        )
        # Remove rows which are all empty e.g. caused by new lines at the end of files
        profiles = profiles.filter(~pl.all_horizontal(pl.all().is_null()))
        profiles = profiles.with_columns(
            pl.all().exclude(profiles.columns[0]).cast(pl.Categorical(category))
        )

        transform.verify_dataframe_integrity(profiles)
        return profiles

    def runtime_test():
        result = read_input_profiles(tmp_file, "\t", 2)
        cluster_args = ClusterArguments(
            delimiter="\t",
            cores=2,
            normalize_distance=True,
            count_missing=True,
            filter_threshold=1,
            columns_path=None,
            input_file=tmp_file,
            output_directory=tmp_path / "test",
            linkage_method=LinkageMetric.AVERAGE,
            thresholds=[0.99],
            branch_type=BranchType.COPHENETIC,
            matrix=False,
        )
        cluster.compute_dists(result, cluster_args)

    benchmark(runtime_test)
    assert True


def test_verify_categories_raises_compute_error(test_df, tmp_path):
    """Verify categories reaises an error if number of unique values exceeded."""
    tmp_file = tmp_path / "output_data.tsv"
    test_df.write_csv(tmp_file, separator="\t")

    def read_input_profiles(input_file: Path, delimiter: str, threads: int) -> pl.DataFrame:
        category = pl.Categories(physical=pl.UInt8, name="test_small")
        profiles = pl.read_csv(
            input_file,
            separator=delimiter,
            n_threads=threads,
            has_header=True,
            raise_if_empty=True,
            infer_schema=False,
        )
        profiles = profiles.with_columns(
            pl.all()
            .exclude(profiles.columns[0])
            .replace(old=list(transform.REPLACE_CHARS.keys()), new=None)
        )
        # Remove rows which are all empty e.g. caused by new lines at the end of files
        profiles = profiles.filter(~pl.all_horizontal(pl.all().is_null()))
        profiles = profiles.with_columns(
            pl.all().exclude(profiles.columns[0]).cast(pl.Categorical(category))
        )

        transform.verify_dataframe_integrity(profiles)
        return profiles

    with pytest.raises(pl.exceptions.ComputeError):
        _ = read_input_profiles(tmp_file, "\t", 2)


@pytest.mark.parametrize(
    "input,delimiter,threads,expected",
    [
        (
            Path("src/beave/tests/data/simple_test_profiles.csv"),
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
            Path("src/beave/tests/data/simple_test_profiles.tsv"),
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
    output_args = DefaultArguments("\t", True, True, None, 0.0, 1, Path(""))
    if error is None:
        filtered_data = transform.filter_rows(dataframe, threshold, output_args)
        assert filtered_data.equals(expected)
    else:
        with pytest.raises(error):
            filtered_data = transform.filter_rows(dataframe, threshold, output_args)


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
    output_args = DefaultArguments("\t", True, True, None, 0.0, 1, Path(""))
    array = np.zeros((profiles.height, 3), dtype=np.uint32)  # array should all be zeros
    for i in array:
        i[2] = np.uint32(2683474508)  # last value should be the hashed version of "A"
    output, _ = cluster.prep_data(profiles, 1.00, transform.transform_data_hashes, output_args)
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
    output_args = DefaultArguments("\t", True, True, None, 0.0, 1, Path(""))
    out = transform.transform_data_categorical_encoding(data, threshold, output_args)
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
    output_args = DefaultArguments("\t", True, True, None, 0.0, 1, Path(""))
    out = transform.transform_data_hashes(data, threshold, output_args)
    assert out.equals(expected)


@pytest.mark.parametrize(
    "method,expected",
    [
        (
            LinkageMetric.SINGLE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
        ),
        (
            LinkageMetric.COMPLETE,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 2.82842712, 3]], dtype=float),
        ),
        (
            LinkageMetric.AVERAGE,
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
            LinkageMetric.COMPLETE,
            [[0.0, 1.0, 1.0, 2.0], [2.0, 4.0, 4.0, 3.0], [3.0, 5.0, 8.0, 4.0]],
        ),
        (
            LinkageMetric.SINGLE,
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
            cluster.BranchType.COPHENETIC,
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
        ),
        (
            np.array([[0, 1, 1.41421356, 2], [2, 3, 1.41421356, 3]], dtype=float),
            cluster.BranchType.PATRISTIC,
            np.array([[0, 1, 0.70710678, 2], [2, 3, 0.70710678, 3]], dtype=float),
        ),
        (
            np.array([[0, 1, 3, 2], [2, 3, 3, 3]], dtype=float),
            cluster.BranchType.PATRISTIC,
            np.array([[0, 1, 1.5, 2], [2, 3, 1.5, 3]], dtype=float),
        ),
        (
            np.array([[0, 1, 3, 2], [2, 3, 3, 3]], dtype=float),
            cluster.BranchType.COPHENETIC,
            np.array([[0, 1, 3, 2], [2, 3, 3, 3]], dtype=float),
        ),
        (
            np.array([[0, 1, 0, 2], [2, 3, 3, 3]], dtype=float),
            cluster.BranchType.PATRISTIC,
            np.array([[0, 1, 0, 2], [2, 3, 1.5, 3]], dtype=float),
        ),
    ],
)
def test_convert_branch_lengths(linkage, branchlength_type, expected):
    """Test for conversiong of branchlengths to cophenetic and patristic distances."""
    assert np.allclose(cluster.convert_branch_lengths(linkage, branchlength_type), expected)


@pytest.mark.parametrize(
    "profiles,count_missing,normalized,expected",
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
            np.array([np.float32(1.0)]),
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
            np.array([np.float32(1.0)]),
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
            np.array([np.float32(1.0)]),
        ),
    ],
)
def test_calc_dists(profiles, count_missing, normalized, expected):
    """Test distance calculation output is correct."""
    output = beave.calc_dists(profiles, 1, normalized, count_missing)
    np.testing.assert_equal(output, expected)


@given(
    arr=nps.arrays(
        dtype=np.uint32,
        shape=(1000, 100),
    )
)
def test_calc_dists_fuzzing_hypothesis_no_infinites(arr):
    """Tests to make sure calc_dists always returns a finite answer."""
    output = np.isfinite(beave.calc_dists(arr, 1, True, False))
    assert np.all(output)


@pytest.mark.parametrize(
    "input,normalized,count_missing",
    [
        (Path("src/beave/tests/data/R1KC1K.tsv"), True, True),
        (Path("src/beave/tests/data/R1KC1K.tsv"), False, True),
        (Path("src/beave/tests/data/R1KC1K.tsv"), False, True),
    ],
)
def test_calc_dists_file_inputs(input, normalized, count_missing):
    """Test inputs of calc dists is correct with known input."""
    profiles: pl.DataFrame = cluster.read_input_profiles(input, "\t", 1)

    @dataclass
    class ClusterArguments:
        cores: int
        count_missing: bool
        normalize_distance: bool
        filter_threshold: float

    input_args = ClusterArguments(1, count_missing, normalized, 1)

    dists, _ = cluster.compute_dists(profiles, input_args)  # type: ignore[reportArgumentType]
    matrix: npt.NDArray = scipy.spatial.distance.squareform(
        dists
    )  # conversion to squareform so iteration of the matrix is simpler as we do not need to
    # calculate the column and row index from the output condensed array.
    for i in range(0, profiles.height):
        sample1: int = int(profiles.item(i, "sample"))
        for f in range(0, profiles.height):
            sample2: int = int(profiles.item(f, "sample"))
            if normalized:
                dist: float = abs(sample1 - sample2) / float(profiles.height)
                assert dist == pytest.approx(matrix[i][f], rel=1e-6)
            else:
                assert float(abs(sample1 - sample2)) == pytest.approx(matrix[i][f], rel=1e-6)


def test_prepare_matrix():
    """Tests for the prepare_matrix function."""
    test_array: npt.NDArray = np.array([1, 2, 3, 4, 5, 6])
    labels: pl.Series = pl.Series(["1", "2", "3", "4"])
    output_df: pl.DataFrame = cluster.prepare_matrix(test_array, labels)
    exepected_df: pl.DataFrame = pl.from_numpy(
        np.array([[0, 1, 2, 3], [1, 0, 4, 5], [2, 4, 0, 6], [3, 5, 6, 0]]),
        schema=labels.to_list(),
    )
    exepected_df = exepected_df.insert_column(0, labels)
    assert output_df.equals(exepected_df)


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
    cols = transform.get_subset_columns(Path("src/beave/tests/data/test_columns.txt"))
    assert cols == {"sample", "col1", "col2", "col3"}


@pytest.mark.workflow("Run cluster pass filter data")
def test_cluster_pass_filter_data(workflow_dir):
    """Verify the outputs of the filtered data are correct."""
    # passed parameter of more than 20% is empty it gets filtered
    # input 1000 rows and columns means 80% of data get filtered
    output_length = 1000
    expected_output_lengths = (
        output_length * 0.2 + 1
    )  # <= comparison so we should have one extra value present
    clusters = Path(workflow_dir, "clusters.tsv")
    values_kept = [int(i.split("\t")[0]) for i in clusters.read_text().split("\n")[1:] if i]
    assert len(values_kept) == expected_output_lengths
    for i in values_kept:
        assert i >= expected_output_lengths
    matrix = Path(workflow_dir, "matrix.tsv")
    lines = [
        [int(f) if f.isdigit() else f for f in i.split("\t")]
        for i in matrix.read_text().split("\n")
        if i
    ]
    header = lines[0]
    for i in range(1, len(lines)):
        sample1 = header[i]  # sample1 position
        for f in range(1, len(lines)):
            sample2 = header[f]
            dist = float(abs(sample1 - sample2))  # type: ignore
            assert dist == float(lines[i][f])

    filtered_samples = [
        i for i in Path(workflow_dir, "FilteredProfiles.txt").read_text().split("\n") if i
    ][1:]  # get all values except for column header
    out_length = output_length - expected_output_lengths
    assert len(filtered_samples) == out_length
    for value in filtered_samples:
        assert int(value) < out_length
