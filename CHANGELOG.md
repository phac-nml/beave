# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.0]

### Added

- Added option to generate a distance matrix only. [PR 45](https://github.com/phac-nml/beave/pull/45)

### Changed

- Updated polars to version 2.0.0. [PR 50](https://github.com/phac-nml/beave/pull/50)

- Optimized Hamming distance hot loop in distance calculation. [PR 42](https://github.com/phac-nml/beave/pull/42)

- Reverted Hamming distance hot loop in distance calculation. [PR 43](https://github.com/phac-nml/beave/pull/43)
  - Benchmarking revealed inefficient vectorization of instructions leading to slightly longer runtimes.

### Removed

- Dropped C++ CLI option removing `syncstream` dependency causing issues in Mac builds. [PR 42](https://github.com/phac-nml/beave/pull/42)
