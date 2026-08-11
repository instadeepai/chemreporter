# Query Examples

The YAML config file for the `query` step accepts SQL-like query conditions
(without the `WHERE` keyword). Note that the file does not have to be named
`query.yaml` — any filename works, as long as it is passed via `-c` to the
CLI. Below is the checked-in example config, which can also be found in the
[ChemReporter GitHub repository](https://github.com/instadeepai/chemreporter/tree/main/example_configs):

```{literalinclude} ../../../example_configs/query.yaml
:language: yaml
```

For context: We build the SQL query internally as:
`SELECT entry_key FROM table WHERE <query>`

## Basic Filtering

These simple examples show how to filter structures by basic dataset
properties, such as the number of atoms in a structure or which source
subset it belongs to. Note that fields like `subset` are populated
differently — or not at all — depending on the source dataset; see
[Crucial Implementation Details](crucial-implementation-details)
before relying on them across datasets.

```yaml
# Filter by number of atoms
query: "num_atoms > 10 AND num_atoms < 50"
# Filter by subset
query: "subset = 'spice'"
# Multiple subsets
query: "subset IN ('spice', 'anix', 'biomolecules')"
```

## Property-based Filtering

Beyond simple structural counts, you can also filter on computed
physicochemical properties, such as lipophilicity, polarity, molecular
weight, or overall net charge.

```yaml
# Lipophilic molecules
query: "logp > 2.0 AND logp < 5.0"
# High polar surface area
query: "tpsa > 100"
# Light molecules with few atoms
query: "molecular_weight < 200 AND num_atoms < 15"
# Charged molecules
query: "net_charge != 0"
```

## Reactivity Filters

For reaction datasets, you can select structures based on their role along a
reaction pathway, for example to isolate transition states.

```yaml
# Select only transition states
query: "is_transition_state = true"
```

## Combined Filters

Real-world use cases often require combining several conditions into a
single, more elaborate query. The examples below showcase common combined
filters, ranging from element-based restrictions to a typical drug-likeness
heuristic.

```yaml
# molecules with element CHNO (and nothing else)
# (but CHN CHO .. are ok  )
query: |
  atomic_symbols  ~ '^[CHNO]+$'
  AND num_atoms > 3
# Large organic molecules
query: |
  num_atoms > 30
  AND molecular_weight > 300
  AND graph_properties_candidate = true
# Clean database (no calculation errors)
query: |
  error_graph_properties = false
  AND graph_properties_candidate = true
  AND subset IN ('spice', 'anix')
# Lipinski rule of 5 approximation
query: |
  molecular_weight BETWEEN 150 AND 500
  AND logp BETWEEN -0.5 AND 5
  AND tpsa BETWEEN 20 AND 130
  AND num_atoms BETWEEN 10 AND 50
  AND error_graph_properties = false
```
