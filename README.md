![PyPI](https://img.shields.io/pypi/v/beave?label=pypi%20package)
![Conda](https://img.shields.io/conda/dn/bioconda/beave)

# Beave

Beave is an open-source bioinformatics utility for genomic clustering and distance querying. This program may be used to create distance matrices from allelic profiles, or to compare groups of isolates against multiple reference sequences.

# Table of Contents

- [Beave](#beave)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
  * [Compatibility](#compatibility)
  * [Bioconda](#bioconda)
  * [pip](#pip)
  * [Conda](#conda)
- [Getting Started](#getting-started)
  * [Usage](#usage)
    + [cluster](#cluster)
    + [match](#match)
    + [matrix](#matrix)
  * [Input](#input)
  * [Output](#output)
    + [Cluster Outputs](#cluster-outputs)
    + [Match Outputs](#match-outputs)
- [Troubleshooting](#troubleshooting)
- [Contact](#contact)
- [Legal](#legal)

# Installation

## Compatibility

Beave only supports Linux distributions. However, as only the C++ 23 standard library and Python libraries are used, in principle any environment that supports G++ and Python may compile and run the program. Since this program relies heavily on the compiler to optimize the program and add SIMD instructions, it is recommended to compile the program on your local computer to get the full benefit of the potential instruction sets your CPU may offer.

To build the `beave` Python package, you will first need to install the dependencies listed in the `pyproject.toml` file. Python version `3.13` or greater is required, along with `scikit-build-core` and the `nanobind` Python package.

Additionally, the following Python runtime dependencies are required:
- `numpy>=2.4.0`
- `polars>= 1.40.1`
- `scipy>=1.17.0`

## Bioconda

Currently, Bioconda only supports Linux. If you wish to install the software on OSX, you will need to install `pypi` to build the project from the source code. To install with Bioconda on Linux, run the following code:

`conda install -c bioconda beave`

## pip

Beave may be installed from `pypi` by running the following command:

`pip install beave`

Beave may alternatively be installed from source by downloading the project and running the following command in the source code directory:

`pip install .`

## Conda

The project may be installed within a Conda environment (without using Bioconda) by downloading the project source and creating a Conda environment on Linux as follows:

`conda env create -f environment-linux.yml`

or on Mac as follows:

`conda env create -f ./environment-osx.yml`

The envinronment can then be activated with:

`conda activate beave`

and the project can be installed with `pip` as follows:

`pip install .`

# Getting Started

## Usage

The main help message for the program is shown below:

```Bash
>>> beave -h
usage: beave [-h] [--cores CORES] [--delimiter DELIMITER] [--columns-subset COLUMNS_SUBSET] [--count-missing] [--normalize-distance]
             [--filter-threshold FILTER_THRESHOLD] [--verbose] [--version]
             {cluster,match} ...

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

### cluster

To run _de-novo_ clustering use the `cluster` utility. The parameters for running `cluster` are shown below:

```Bash
>>> beave cluster --help

usage: beave cluster [-h] [--cores CORES] [--delimiter DELIMITER] [--columns-subset COLUMNS_SUBSET] [--count-missing] [--normalize-distance] [--filter-threshold FILTER_THRESHOLD] [--verbose] --input INPUT [--output OUTPUT] --thresholds THRESHOLDS [THRESHOLDS ...]
                     [--linkage-method {single,average,complete}] [--branch-type {patristic,cophenetic}] [--matrix]

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
                        Exclude samples from analysis if they are missing more than the specified percentage of data. Must be between [0.0-100.0]. (default 100.0)
  --verbose             Display logger debug messages.
  --input, -i INPUT     Input alleles. (required)
  --output, -o OUTPUT   Output directory for generated tree and clusters, directory will be created if does not exist. (default: beave)
  --thresholds, -t THRESHOLDS [THRESHOLDS ...]
                        List of threshold values to use. (required)
  --linkage-method, -l {single,average,complete}
                        Hierarchical clustering linkage to use. (default: average)
  --branch-type, -b {patristic,cophenetic}
                        Determine how to display tree lenghts in the Newick file. (default cophenetic)
  --matrix              Write the computed distance matrix to a file in the output directory called 'matrix.tsv'.

>>> # Example programs
>>> beave cluster --input src/beave/tests/data/R1KC1K.2-zeroes.does-not-exist.csv -o out -l average -nm -b cophenetic -c 2 -t 0.9 0.5 --delimiter , --matrix
>>> beave cluster --input src/beave/tests/data/R1KC1K.tsv -l average -b cophenetic -c 1 --thresholds 10 9 8
```

### match

The `match` utility may be used to compute pairwise distances between a group of query samples against a group of reference samples. The parameters for running `match` are described below:

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
                        Exclude samples from analysis if they are missing more than the specified percentage of data. Must be between [0.0-100.0]. (default 100.0)
  --verbose             Display logger debug messages.
  --reference, -r REFERENCE
                        Profiles to compare against. Query samples will be included in comparisons. (required)
  --query, -q QUERY     Profiles containing new-samples for comparisons. (required)
  --threshold, -t THRESHOLD
                        Only report distances below specified threshold. (default: infinity)
  --output, -o OUTPUT   Output directory for calculated distances, directory will be created if does not exist. (default: beave)

>>> # Example programs
>>> beave match -q src/beave/tests/data/R1KC1K.head.tsv -r src/beave/tests/data/R1KC1K.tail.tsv -nm --verbose
>>> beave match -q src/beave/tests/data/R1KC1K.head.tsv -r src/beave/tests/data/R1KC1K.tail.tsv -t 101 -l average -c 8
```

When running `match`, the query and reference profiles will be merged by the program. If duplicate ID's are detected an error will be raised by the program.

### matrix

In some instances you may have no need to perform clustering and simply want a distance matrix for other downstream purposes. The matrix utility can perform this task and generate a matrix or all pairwise distances in a molten format. For example:

`SampleID_1, SampleID_2, dist_{hamming,normalized}`

The parameters for running `matrix` are described below:

```Bash
>>> beave matrix --help
usage: beave matrix [-h] [--cores CORES] [--delimiter DELIMITER] [--columns-subset COLUMNS_SUBSET] [--count-missing]
                    [--normalize-distance] [--filter-threshold FILTER_THRESHOLD] [--verbose] --input INPUT [--output OUTPUT]

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
                        Exclude samples from analysis if they are missing more than the specified percentage of data. Must be between
                        [0.0-100.0]. (default 100.0)
  --verbose             Display logger debug messages.
  --input, -i INPUT     Input alleles. (required)
  --output, -o OUTPUT   Output directory for generated tree and clusters, directory will be created if does not exist. (default: .)
  --molten              Write the final matrix in molten format.

>>> beave matrix -i src/beave/test/data/R1KC1K.head.tsv -nm --verbose -o output
>>> beave matrix -i src/beave/test/data/R1KC1K.head.tsv --verbose -o output --molten
```

## Input

The inputs must be provided in tabular format. Most delimiters are supported as long as they are a single character. The first column of the file must contain no duplicates or missing values. The columns are not inspected to verify uniqueness and duplicated columns will be loaded incorrectly as unique columns. The characters `?`, ` `(space), (blank), `-`, `\_`, and `0` are treated as missing values by the program unless the `--count-missing` option is specified. All other values are treated as a valid alleles. Example inputs can be found in the `tests` directory. Thresholds are always converted to float values, however both integers and floating point numbers may be provided.

## Output

### Cluster Outputs

The program outputs a Newick-format file containing the tree generated by whichever linkage metric is selected. The sample IDs and their addresses are put out in a separate file specified by the user in TSV format. Addresses are delimited by an `.`. The following is an example of the clustering output:

| SampleID    | level_10.0 | level_9.0 | denovo_address |
| ----------- | ---------- | --------- | -------------- |
| CoolSample  | 1          | 2         | 1.2            |
| CoolSample2 | 2          | 1         | 2.1            |

### Match Outputs

The output of `match` is a single file showing the query sample, reference sample, and distance. The following is an example of the match output:

| query_id | ref_id | dist\_{hamming,normalized} |
| -------- | ------ | -------------------------- |
| 1        | 2      | 4                          |
| 1        | 3      | 8                          |
| 1        | 4      | 10                         |

# Troubleshooting

- If you encounter any issues installing or running the program, please create a GitHub issue for any issues identified.

- If there are duplicate identifiers with different profiles. The distance between the two values will still be reported, as no duplicate detection is performed currently. Future iterations may add this functionality.

- `-ffast-math` is enabled during compilation to prevent sub-normals. This leads to some error in floating point operations, the affect of this is being evaluated and this compiler flag may be removed in the future.

# Contact

[Matthew Wells] : <matthew.wells@phac-aspc.gc.ca>

[Eric Marinier] : <eric.marinier@phac-aspc.gc.ca>

# Legal

Copyright Government of Canada 2026

Written by: National Microbiology Laboratory, Public Health Agency of Canada

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this work except in compliance with the License. You may obtain a copy of the License at:

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions and limitations under the License.
