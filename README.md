# `beave`

- [Introduction](#introduction)
  - [Contact](#contact)
- [Compatibility](#compatibility)
- [Install](#install)
  - [Get Started](#get-started)
  - [Python](#python)
    - [Without Conda](#without-conda)
    - [With Conda](#with-conda)
- [Getting Started](#getting-started)
  - [Using Python](#using-python)
    - [Usage](#usage)
    - [Configuration and Settings](#configuration-and-settings)
    - [Data Input](#data-input)
    - [Data Output](#data-output)
  - [Using C++ Binary](#using-c++-binary)
    - [Usage](#usage)
    - [Configuration and Settings](#configuration-and-settings)
    - [Data Input](#data-input)
    - [Data Output](#data-output)
- [Troubleshooting and FAQs](#troubleshooting-and-faqs)
- [Other Information](#other-information)
- [Legal and Compliance Information](#legal-and-compliance-information)
- [Updates and Release Notes](#updates-and-release-notes)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Introduction

## Python CLI

A very Canadian utility for genomic clustering and distance querying.

This program is under active development and is used for creating distance matrices from allelic profiles, or for comparing groups of isolates against multiple. This program is similar to [cgmlst-dists](https://github.com/tseemann/cgmlst-dists) from Torstein Tseeman and [gas](https://github.com/phac-nml/genomic_address_service) from the Public Health Agency of Canada.

## Contact

[Matthew Wells] : <matthew.wells@phac-aspc.gc.ca>

## Compatibility

`beave` has only been tested on Linux, any system that supports G++ can compile the program. As only the C++ 23 standard library is used, the program may be able to be compiled on Windows system.

This program relies heavily on the compiler to optimize the program and add SIMD instructions, it is recommended to compile the program on your local computer to get the full benefit of the potential instruction sets your CPU may offer. Compilation using AVX-512 instruction sets has been tested, however in our testing the programs performance degrades likely due to throttling by the CPU.

To build the `beave` Python package, you will first need to install dependencies listed in the pyproject.toml file. Python version 3.13 or greater is required, along with scikit-build-core and the nanobind Python package.

Python runtime dependencies include numpy >= 2.4.0 and polars >= 1.40.1 and scipy >= 1.17.0.

## Install

### Get Started

Start by pulling the repository.

`git clone https://github.com/phac-nml/beave`

### Python

#### Without Conda

To build and install the Python package you must have the following python packages, `scikit-build-core` and `nanobind` which can be installed with `pip install nanobind scikit-build-core[pyproject]`.

Users can run simply run `pip install .` to install the package.

Developers can run `pip install -ve .[dev]` or `pip install -Ceditable.rebuild=true -ve .[dev]`. Further examples can be found in the nanobind documentation here: [nanobind packaging](https://nanobind.readthedocs.io/en/latest/packaging.html). The `--no-build-isolation` is not included as running `uv pip install` with the `--no-build-isolation` flag tells uv to ignore the `build-system.requires` section from the `pyproject.toml`.

To build a wheel that can be distributed instead of installed, simply run `pip wheel .`

#### With Conda

1. Pull the GitHub repository as described above.

2. Create the Conda environment by running `conda env create -f environment-linux.yml` for linux and `conda env create -f ./environment-osx.yml` for mac.

3. Activate the environment with: `conda activate beave`

4. Run `pip install .`, if you wish to install development dependencies or run tests please run `pip install -ve .[dev]`

## Getting Started

#### Usage

The main help message for the program is shown below:

```Bash
>>> beave -h
usage: beave [-h] [--cores CORES] [--delimiter DELIMITER] [--columns-subset COLUMNS_SUBSET] [--count-missing] [--normalize-distance]
             [--filter-threshold FILTER_THRESHOLD] [--verbose] [--version]
             {cluster,match} ...

A very Canadian utility for genomic clustering and distance querying.

positional arguments:
  {cluster,match}       Select a program to run.
    cluster             Run denovo clustering.
    match               Run fast matching.

options:
  -h, --help            show this help message and exit
  --cores, -c CORES     Specify the number of threads to be used. [default 12]
  --delimiter DELIMITER
                        Input alleles delimiter. [default \t] (default: )
  --columns-subset, -s COLUMNS_SUBSET
                        A file containing a single column of the column names to subset from the passed allele profiles. (default: None)
  --count-missing, -m   Count missing values as differences. (default: False)
  --normalize-distance, -n
                        Compute the normalized distance. Distance is presented as a percentage, or a value between [0.0-1.0] (default:
                        False)
  --filter-threshold, -f FILTER_THRESHOLD
                        Exclude samples from analysis if they are missing more than the specified percentage of data. Must be between
                        [0.0-100.0]. [default 100.0] (default: 1.0)
  --verbose             Display logger debug messages. (default: False)
  --version, -v         show program's version number and exit
```

To run _de-novo_ clustering use the `cluster` option. The long form options for cluster are shown below:

```Bash
>>> beave cluster --help

usage: beave cluster [-h] [--cores CORES] [--delimiter DELIMITER] [--columns-subset COLUMNS_SUBSET] [--count-missing]
                     [--normalize-distance] [--filter-threshold FILTER_THRESHOLD] [--verbose] --input INPUT [--output OUTPUT]
                     --thresholds THRESHOLDS [THRESHOLDS ...] [--linkage-method {single,average,complete}]
                     [--branch-type {patristic,cophenetic}]

options:
  -h, --help            show this help message and exit
  --cores, -c CORES     Specify the number of threads to be used. [default 12]
  --delimiter DELIMITER
                        Input alleles delimiter. [default \t]
  --columns-subset, -s COLUMNS_SUBSET
                        A file containing a single column of the column names to subset from the passed allele profiles.
  --count-missing, -m   Count missing values as differences.
  --normalize-distance, -n
                        Compute the normalized distance. Distance is presented as a percentage, or a value between [0.0-1.0]
  --filter-threshold, -f FILTER_THRESHOLD
                        Exclude samples from analysis if they are missing more than the specified percentage of data. Must be between
                        [0.0-100.0]. [default 100.0]
  --verbose             Display logger debug messages.
  --input, -i INPUT     Input alleles.
  --output, -o OUTPUT   Output directory for generated tree and clusters, directory will be treated if does not exist. [default:
                        /home/CSCScience.ca/mwells/Development/beave]
  --thresholds, -t THRESHOLDS [THRESHOLDS ...]
                        List of threshold values to use.
  --linkage-method, -l {single,average,complete}
                        Hierarchical clustering linkage to use. [default: average]
  --branch-type, -b {patristic,cophenetic}
                        Determine how to display tree lenghts in the Newick file. [default cophenetic]
  --matrix              Write the computed distance matrix to a file in the output directory called 'matrix.tsv'.
>>> # Example programs
>>> beave cluster --input src/beave/tests/data/R1KC1K.2-zeroes.does-not-exist.csv -o out -l average -nm -b cophenetic -c 2 -t 0.9 0.5 --delimiter , --matrix
>>> beave cluster --input src/beave/tests/data/R1KC1K.tsv -l average -b cophenetic -c 1 --thresholds 10 9 8
```

The match argument may be used to compute pairwise distances between a group of query samples against a group of reference samples. The parameters for running match are described below:

```Bash
>>> beave match --help
usage: beave match [-h] [--cores CORES] [--delimiter DELIMITER] [--columns-subset COLUMNS_SUBSET] [--count-missing] [--normalize-distance] [--filter-threshold FILTER_THRESHOLD] [--verbose] --reference REFERENCE --query QUERY [--threshold THRESHOLD] [--output OUTPUT]

options:
  -h, --help            show this help message and exit
  --cores, -c CORES     Specify the number of threads to be used. (default 12)
  --delimiter DELIMITER
                        Input alleles delimiter. (default \t)
  --columns-subset, -s COLUMNS_SUBSET
                        A file containing a single column of the column names to subset from the passed allele profiles.
  --count-missing, -m   Count missing values as differences.
  --normalize-distance, -n
                        Compute the normalized distance. Distance is presented as a percentage, or a value between [0.0-1.0]
  --filter-threshold, -f FILTER_THRESHOLD
                        Exclude samples from analysis if they are missing more than the specified percentage of data. Must be between [0.0-1.0]. (default 1.0)
  --verbose             Display logger debug messages.
  --reference, -r REFERENCE
                        Profiles to compare against. Query samples will be included in comparisons.
  --query, -q QUERY     Profiles containing new-samples for comparisons.
  --threshold, -t THRESHOLD
                        Only report distances below specified threshold. (default: infinity)
  --output, -o OUTPUT   Output directory for calculated distances. (default: beave)

>>> # Example programs
>>> beave match -q src/beave/tests/data/R1KC1K.head.tsv -r src/beave/tests/data/R1KC1K.tail.tsv -nm --verbose
>>> beave match -q src/beave/tests/data/R1KC1K.head.tsv -r src/beave/tests/data/R1KC1K.tail.tsv -t 101 -l average -c 8
```

#### Data Input

The inputs for this program must be tabular, any delimiter is supported as long is it is a single character. The first column of the file must contain no duplicates or missing values. The columns are not inspected to verify unique values only, so duplicate column names will be name mangled and treated as another unique column. The characters "?", " ", "", "-", "\_", and "0" are treated as missing values by the program unless the `-c` option is added to the program. All other values are treated as a valid alleles. Example inputs can be found in the `tests` folder. Thresholds are always converted to float values, however you can specify either integers not just decimals.

When running `match`, the query and reference profiles will be merged by the program. If duplicate ID's are detected an error will be raised by the program.

#### Data Output

##### Cluster Outputs

The program outputs a Newick-Format file containing the tree generated by whichever linkage metric is selected, the sample IDs and their addresses are put out in a separate file specified by the user in TSV format. Addresses are delimited by an '.'.

Example of cluster outputs:

| SampleID    | level_10.0 | level_9.0 | denovo_address |
| ----------- | ---------- | --------- | -------------- |
| CoolSample  | 1          | 2         | 1.2            |
| CoolSample2 | 2          | 1         | 2.1            |

##### Match Outputs

The output of `match` is a single file showing the query sample, the reference sample and distance.

The general structure of the `match` output:

| query_id | ref_id | dist\_{hamming,normalized} |
| -------- | ------ | -------------------------- |
| 1        | 2      | 4                          |
| 1        | 3      | 8                          |
| 1        | 4      | 10                         |

# Troubleshooting and FAQs

- There may be issues with the Python build system, please create a GitHub issue for any issues identified.

- If there are duplicate identifiers with different profiles. The distance between the two values will still be reported as no duplicates detection is performed currently. Future iterations may add this functionality.

- `-ffast-math` is enabled during compilation to prevent sub-normals. This leads to some error in floating point operations, the affect of this is being evaluated and this compiler flag may be removed after further testing is performed.

# Legal and Compliance Information

Copyright Government of Canada [2026]

Written by: National Microbiology Laboratory, Public Health Agency of Canada

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this work except in compliance with the License. You may obtain a copy of the License at:

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.

# Updates and Release Notes

Please see the `CHANGELOG.md`.
