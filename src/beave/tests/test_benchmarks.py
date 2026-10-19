"""Benchmarks for cluster.py.

Polars environment variables for threads are not set here to control the
number of threads used. As polars configures this at import time and pytest
does not support configuring a temporary environment variable before module imports.
"""

import pytest  # noqa: I001

from beave import cluster
from beave.declarations import BranchType, ClusterArguments, DefaultArguments, LinkageMetric
from beave import transform_data as transform

import hashlib
from pathlib import Path

import polars as pl


@pytest.fixture(scope="module")
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
    threads = 2
    tmp_file = tmp_path / "output_data.tsv"
    test_df.write_csv(tmp_file, separator="\t")

    def read_input_profiles(input_file: Path, delimiter: str):
        """Test case where each column is its own category."""
        profiles = pl.read_csv(
            input_file,
            separator=delimiter,
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
        result = read_input_profiles(tmp_file, "\t")
        cluster_args = ClusterArguments(
            delimiter="\t",
            cores=threads,
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
            matrix_only=False,
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
    threads = 2

    def read_input_profiles(input_file: Path, delimiter: str) -> pl.DataFrame:
        category = pl.Categories(physical=pl.UInt32, name="test_32")
        profiles = pl.read_csv(
            input_file,
            separator=delimiter,
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
        result = read_input_profiles(tmp_file, "\t")
        cluster_args = ClusterArguments(
            delimiter="\t",
            cores=threads,
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
            matrix_only=False,
        )
        cluster.compute_dists(result, cluster_args)

    benchmark(runtime_test)
    assert True


def test_verify_categories_raises_compute_error(test_df, tmp_path):
    """Verify categories raises an error if number of unique values exceeded."""
    tmp_file = tmp_path / "output_data.tsv"
    test_df.write_csv(tmp_file, separator="\t")
    threads = 2

    def read_input_profiles(input_file: Path, delimiter: str, threads: int) -> pl.DataFrame:
        category = pl.Categories(physical=pl.UInt8, name="test_small")
        profiles = pl.read_csv(
            input_file,
            separator=delimiter,
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
        _ = read_input_profiles(tmp_file, "\t", threads)
