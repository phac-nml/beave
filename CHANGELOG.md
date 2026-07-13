# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Optimized Hamming distance hot loop in distance calculation. [PR 42](https://github.com/phac-nml/beave/pull/42)

- Reverted change in Hamming distance hot loop in distance calculation. [PR 43](https://github.com/phac-nml/beave/pull/43)
  - Change did not result in substantial performance improvement and may automatic SIMD vectorization.

### Removed

- Dropped C++ CLI option removing `syncstream` dependency causing issues in Mac builds. [PR 42](https://github.com/phac-nml/beave/pull/42)
