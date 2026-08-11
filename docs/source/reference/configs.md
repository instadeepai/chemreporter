# Configuration

This page documents the configuration files used by the `chemreporter` CLI. Each
CLI command (`process`, `query`, `export`) reads a YAML config file, which is
validated at runtime against a corresponding schema. These schemas are defined
in the `chemreporter` library as Pydantic classes, linked in the table below;
follow those links to the [Config Schemas API](config-schemas-api) section
further down this page for a detailed, auto-generated reference of every field,
its type, and its default value. These example configs can be found in the
[ChemReporter GitHub repository](https://github.com/instadeepai/chemreporter/tree/main/example_configs),
and their content is also shown further below on this page.

| Command | Example config file | Schema |
|---------|-------------|--------|
| `process` | `example_configs/process.yaml` | {class}`chemreporter.config_schemas.ProcessDBConfig` |
| `query` | `example_configs/query.yaml` | {class}`chemreporter.config_schemas.QueryDBConfig` |
| `export` | `example_configs/export.yaml` | {class}`chemreporter.config_schemas.ExportHDF5Config` |


## Example YAML config for "process" step

The config below is passed to `chemreporter process -c <config>.yaml` and
controls how raw source datasets are ingested into the Query Database.

```{literalinclude} ../../../example_configs/process.yaml
:language: yaml
```

Note that the field `graph_based_processing` is validated by
{class}`chemreporter.config_schemas.GraphBasedProcessingConfig`.

## Example YAML config for "query" step

The config below is passed to `chemreporter query -c <config>.yaml` and
controls how the Query Database is filtered into a data subset.

```{literalinclude} ../../../example_configs/query.yaml
:language: yaml
```

For query syntax and filtering examples, see [Query Examples](query_examples.md).

## Example YAML config for "export" step

The config below is passed to `chemreporter export -c <config>.yaml` and
controls how the filtered subset is written out to an HDF5 file.

```{literalinclude} ../../../example_configs/export.yaml
:language: yaml
```

(config-schemas-api)=
## Config Schemas API

```{eval-rst}
.. automodule:: chemreporter.config_schemas
   :members:
   :show-inheritance:
   :undoc-members:
```
