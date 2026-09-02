# 🧪 ChemReporter

[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Python Version](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit)
[![Tests and Linters](https://github.com/instadeepai/chemreporter/actions/workflows/tests_and_linters.yaml/badge.svg?branch=main)](https://github.com/instadeepai/chemreporter/actions/workflows/tests_and_linters.yaml)
![badge](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/M4rieBlu/f2fe3c924dd33b0aa1c17aa49f522e5e/raw/pytest-coverage-comment.json)

📚 Full documentation: https://instadeepai.github.io/chemreporter

## 👀 Overview

**ChemReporter** is a framework that
converts molecular and materials datasets into a unified,
queryable representation, and exports the result directly into training
data for Machine Learning Interatomic Potentials (MLIPs).

It operates in three decoupled stages:

1. **Process** — parse raw source datasets into a partitioned Apache Parquet
   repository, the Query Database, enriched with structural, physical, and
   chemical metadata.
2. **Query** — filter and sample the Query Database via the CLI using
   SQL-like selection criteria, from simple physical constraints (e.g.,
   number of atoms, force magnitude) to custom, user-defined strategies.
3. **Export** — stream the selected subset into an HDF5 file, ready for
   direct use in modern MLIP training frameworks.

ChemReporter currently supports five source datasets:
[**OMOL25**](https://arxiv.org/pdf/2505.08762),
[**OC20**](https://pubs.acs.org/doi/10.1021/acscatal.0c04525),
[**OMAT24**](https://arxiv.org/html/2410.12771v1),
[**OMC25**](https://www.nature.com/articles/s41597-026-06628-2), and
[**ODAC**](https://pubs.acs.org/doi/10.1021/acscentsci.3c01629). See
[Supported Source Datasets](https://instadeepai.github.io/chemreporter/source_datasets/source_datasets.html)
for details on each one.

ChemReporter is released under the Apache License 2.0.

## 📦 Installation

ChemReporter requires Python 3.11 or newer. Install the package with `pip`:

```
pip install chemreporter
```

This gives you the `chemreporter` command-line tool along with the Python
library. See the
[Installation guide](https://instadeepai.github.io/chemreporter/guides/installation.html)
for more details, including the development install.

## 🚀 Quick Start

Once installed, the `chemreporter` CLI gives you three commands, one for
each stage of the workflow. Each command is configured via a YAML
configuration file, passed with the `-c` flag:

```
chemreporter process -c /path/to/process.yaml
chemreporter query -c /path/to/query.yaml
chemreporter export -c /path/to/export.yaml
```

See the [CLI guide](https://instadeepai.github.io/chemreporter/guides/cli.html)
for a full walkthrough of each command and its configuration options.

## 🧭 Next Steps

The [full documentation](https://instadeepai.github.io/chemreporter) covers
everything in more depth, including:

- **[CLI guide](https://instadeepai.github.io/chemreporter/guides/cli.html)** —
  a detailed walkthrough of the `process`, `query`, and `export` commands,
  plus how to write custom I/O plugins for other storage backends.
- **[Configuration reference](https://instadeepai.github.io/chemreporter/reference/configs.html)**
  — every field of the `process`, `query`, and `export` YAML configs, backed
  by their Pydantic schemas.
- **[Query examples](https://instadeepai.github.io/chemreporter/reference/query_examples.html)**
  — common SQL-like filtering patterns, from basic property filters to
  drug-likeness heuristics.
- **[Query Database schema](https://instadeepai.github.io/chemreporter/reference/database_schema.html)**
  and **[Units and physical quantities](https://instadeepai.github.io/chemreporter/reference/units.html)**
  — every field you can query on, and the units it is stored in.
- **[Export schema](https://instadeepai.github.io/chemreporter/reference/hdf5_schema.html)**
  — the internal layout of the exported HDF5 files, and how they plug into the
  [`mlip`](https://github.com/instadeepai/mlip) training library.
- **[Supported source datasets](https://instadeepai.github.io/chemreporter/source_datasets/source_datasets.html)**
  — details and a computational setup comparison for each supported source
  dataset.


## 📚 Citing our work

We kindly request that you cite our white paper when using this library:

Marie Bluntzer, Jules Tilly, Christoph Brunken
*ChemReporter: A Framework for Curating and Exporting Large-Scale Chemical Datasets for MLIP Training*
arXiv, 2026, arXiv:2608.16418

### BibTeX

```bibtex
@misc{bluntzer2026chemreporterframeworkcuratingexporting,
  title={ChemReporter: A Framework for Curating and Exporting Large-Scale Chemical Datasets for MLIP Training},
  author={Marie Bluntzer and Jules Tilly and Christoph Brunken},
  year={2026},
  eprint={2608.16418},
  archivePrefix={arXiv},
  primaryClass={physics.chem-ph},
  url={https://arxiv.org/abs/2608.16418},
}
