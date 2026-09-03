# Changelog

## Release 0.1.0

Initial public (open-source) release of ChemReporter.

- Three-stage `chemreporter` CLI — `process`, `query`, and `export` — each
  configured via a YAML file, for converting molecular and materials
  datasets into training data for Machine Learning Interatomic Potentials
  (MLIPs).
- **Process**: parses raw source datasets into a partitioned Apache Parquet
  Query Database, enriched with structural, physical, and chemical metadata.
- **Query**: filters and samples the Query Database using SQL-like
  selection criteria, from simple physical constraints to custom,
  user-defined sampling strategies.
- **Export**: streams a selected subset into an HDF5 file compatible with
  the [`mlip`](https://github.com/instadeepai/mlip) library.
- Support for five source datasets: OMOL25, OC20, OMAT24, OMC25, and ODAC.
- Full documentation, including CLI and configuration guides, a Query
  Database schema reference, an HDF5 export schema reference, and
  per-dataset usage notes.
- Apache License 2.0.
