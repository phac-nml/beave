# `dist-mat`

- [Introduction](#introduction)
  - [Citation](#citation)
  - [Contact](#contact)
- [Install](#install)
  - [Compatibility](#compatibility)
- [Getting Started](#getting-started)
  - [Usage](#usage)
  - [Configuration and Settings](#configuration-and-settings)
  - [Data Input](#data-input)
  - [Data Output](#data-output)
- [Troubleshooting and FAQs](#troubleshooting-and-faqs)
- [Other information](#other-information)
- [Legal and Compliance Information](#legal-and-compliance-information)
- [Updates and Release Notes](#updates-and-release-notes)

<small><i><a href='http://ecotrust-canada.github.io/markdown-toc/'>Table of contents generated with markdown-toc</a></i></small>

# Introduction

This program is under active development and is used for creating distance matrices from allelic profiles, or for comparing groups of isolates against multiple. This program is similar to [cgmlst-dists](https://github.com/tseemann/cgmlst-dists) from Torstein Tseeman, and uses test data from his original program.

## Citation

_Include how to cite the tool_

## Contact

[NAME] : <FAKE@phac-aspc.gc.ca>

## Install

## Pull the Repository

`git pull --recurse-submodules https://github.com/mattheww95/dist-mat`

## Building C++ cli

This program is written entirely in C++ 23, the only dependencies are a g++ compiler and CMake. Catch2 is required for testing however the library is only required for testing and is managed by CMake, this means an internet connection is required when first building the program.

To build the program pull the latest branch and follow the proceeding instructions:

```
cd ./dist-mat
mkdir build && cd build
cmake .. -DTARGET_GROUP=release
make -j4
```

This will compiler a release build and the assembled binary will be available in the `release` directory created by CMake. This will be located in the build directory. The resulting binary can be copied into a `bin` directory your system path can find it.

To run tests follow the build instructions below (it is presumed you are in the `build` directory created in the previous step already):

```
cmake .. -DTARGET_GROUP=test
make -j4
make test

```

Unit tests are run by Catch2 however all tests are orchestrated by CTest which is typically bundled with CMake.

To create a debug build follow the next steps (it is presumed you are still in the debug directory):

```
cmake .. # The debug build is the default
make -j4
```

The output binary will be in the debug directory.

## Building the Python Package

### Without Conda

To build and install the python package you must have the following python dependencies, `scikit-build-core` and `nanobind` which can be installed with `pip install nanobind scikit-build-core[pyproject]`.

Developers can run `pip install --no-build-isolation -ve .` or `pip install --no-build-isolation -Ceditable.rebuild=true -ve .`. Further examples can be found in the nanobind documentation here [nanobind packaging](https://nanobind.readthedocs.io/en/latest/packaging.html).

To build a wheel that can be distributed instead of installed simply run `pip wheel .`

### With Conda

1. Pull the github repository as described above.

2. Create the conda environment by running `conda env create -f environment.yml`

3. Activate the environment with : `conda activate dist-mat`

4. Enjoy

# Compatibility

`dist-mat` has only been tested on linux, any system that supports g++ can compile the program. As only the C++ 23 standard library is used, the program may be able to be compiled on windows system.

This program relies heavily on the compiler to optimize the program and add SIMD instructions, it is recommended too compile the program on your local computer to get the full benefit of the potential instruction sets your CPU may offer especially if AVX-512 instructions are available.

To build the python package `dist-mat` dependencies are listed in the `pyproject.toml`, python version 3.13 or greater is required, along with scikit-build-core and the nanobind python package.

Runtime dependencies only include numpy >= 2.4.0 and polars >= 1.38.1 and scipy >= 1.17.0. These packages are not required for building the program however.

# Getting Started

## Usage

The help message for the program can be brought up by executing as shown below, (-h|--help) can be used to print the help message at any time as well:

```
$ dist-mat
No args passed
Subcommands:
 matrix - Create distance matrix with an input profile.
 fast-match - Compare a set of profiles to a set of query profiles.

Examples:
dist-mat matrix -i profiles.tsv -t 4 -sc > output.tsv
dist-mat fast-match -i qprofiles.tsv -r profiles.tsv -t 4 -sc > output.tsv
```

The program will display the top-level help messages with the example commands, and warn the user that no arguments have been passed.

Note - The program supports multi-threading but is intended for use on a single CPU therefore the maximum number of CPU cores that can be specified currently is 256.

### Creating a distance matrix

The help message for commands is displayed below along with default settings. Flag options can be specified in a sequence as a single letter string e.g. "-sc" or sperately "-s -c". When passing characters such as delimiters you may have to specify ANSI C quoted strings in bash (e.g to specify a tab delimiter $'\t'). The default values are specified in the help message.

```
$ dist-mat matrix

Subcommands:
 matrix - Create distance matrix with an input profile.
 fast-match - Compare a set of profiles to a set of query profiles.

Examples:
dist-mat matrix -i profiles.tsv -t 4 -sc > output.tsv
dist-mat fast-match -i qprofiles.tsv -r profiles.tsv -t 4 -sc > output.tsv

Command Options

 --input| -i:
 Input file file of profiles. [required]
 --threads| -t:
 How many threads to run. default = 1 [optional]
 --missing| -m:
 Specify the charactar to use for missing values. default = 0 [optional]
 --delimiter| -d:
 Delimiter for table. default = \t [optional]
 --scaled| -s:
 Calculate a scaled distance metric. [flag]
 --count-missing| -c:
 Include missing values in count of differences. [flag]
```

### Running Fast Matching

The help message for fast-matching is shown below. The one value that differs to generating a distance matrix is the addition of the `-r` flag to specify a set of reference profiles.

```
Subcommands:
 matrix - Create distance matrix with an input profile.
 fast-match - Compare a set of profiles to a set of query profiles.

Examples:
dist-mat matrix -i profiles.tsv -t 4 -sc > output.tsv
dist-mat fast-match -i qprofiles.tsv -r profiles.tsv -t 4 -sc > output.tsv

Command Options

 --input| -i:
 Input file file of profiles. [required]
 --reference| -r:
 Reference profiles to use for fast matching. [required]
 --threads| -t:
 How many threads to run. default = 1 [optional]
 --missing| -m:
 Specify the charactar to use for missing values. default = 0 [optional]
 --delimiter| -d:
 Delimiter for table. default = \t [optional]
 --scaled| -s:
 Calculate a scaled distance metric. [flag]
 --count-missing| -c:
 Include missing values in count of differences. [flag]
```

### Explanation of flags

- `-t|--threads`: Specify the number of threads passed to the program, if you specify more threads than profiles to use. Then only 1 thread will be used.
- `-m|--missing`: This parameter allows you to specify a character to specify marking certain allele values as uncalled. The default value is '0', but '?', '-' etc. can be passed.
- `-d|-delimiter`: Specify the delimiter used by the input file, a tab character is the default but simple specify ',' to use a comma instead, any single character can be used.
- `-s|--scaled`: Use the scaled distance instead of hamming, this metric will report the hamming distance normalized by the number of comparisons made.
- `-c|--count-missing`: Specify this value to treat missing allele calls as values, by default comparisons to missing allele calls are not made however enabling this value to count missing values as difference will greatly increase the speed of the program.

A detailed description of the help flags for each program is provided below.

## Data Input

Input data needs to be a flat tabular file, the left most column is treated as the sample columns and all subsequent columns are the alleles. The header line is for the most part ignored but is required, if the first line in your table is sample infromation it will be treated as the file header.

Example inputs can be found in the `data` directory of the repository.

## Data Output

_Explanation of how to interpret and/or export data. Include an example table of output if applicable with columns explained._

# Troubleshooting and FAQs

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
