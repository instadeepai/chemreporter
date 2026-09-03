# Installation

ChemReporter requires Python 3.10 or newer. Most users only need the CLI and
library to process, query, and export chemical datasets (see [CLI](cli.md)); if you
plan to contribute to ChemReporter itself, use the development install
instead.

## Standard installation

Install the package with `pip`:

```bash
pip install chemreporter
```

This gives you the `chemreporter` command-line tool along with the Python
library.

## Development install (via uv)

If you would like to install ChemReporter from source, it can be done with
[`uv`](https://docs.astral.sh/uv/), which also sets up the test, lint, and docs
tooling.

```bash
# Clone the repository
git clone https://github.com/instadeepai/chemreporter.git
cd chemreporter

# Install with all development dependency groups
uv sync --all-groups

# Activate the environment
source .venv/bin/activate
```
