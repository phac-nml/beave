"""Contains arguments passed to run functions to prevent circular imports."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class LinkageMetric(StrEnum):
    """Linkage options that can be passed to scipy."""

    SINGLE = "single"
    AVERAGE = "average"
    COMPLETE = "complete"


class BranchType(StrEnum):
    """Branch length types used for converting tree branch metrics."""

    PATRISTIC = "patristic"
    COPHENETIC = "cophenetic"


@dataclass(slots=True)
class DefaultArguments:
    """Global arguments used across parsers."""

    delimiter: str
    count_missing: bool
    normalize_distance: bool
    columns_path: Path | None
    filter_threshold: float
    cores: int
    output_directory: Path


@dataclass(slots=True)
class ClusterArguments(DefaultArguments):
    """CLI arguments for clustering program."""

    input_file: Path
    thresholds: list[float]
    linkage_method: str
    branch_type: BranchType
    matrix: bool
    matrix_only: bool


@dataclass(slots=True)
class MatchArguments(DefaultArguments):
    """CLI arguments for match process."""

    query: Path
    reference: Path
    threshold: float
