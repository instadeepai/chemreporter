# Sampling

Once a `query` step has filtered the query database down to the rows you
care about — see [Query Examples](../reference/query_examples.md) for the
SQL filter syntax — its `sampling` block lets you pick a subset of those
rows to keep. In basic usage, `method` is set to `random` or `all`; for
more advanced use cases, it can instead point at a Python file to run your
own selection strategy — for example, greedy diversity sampling based on
Tanimoto distance, shown below as a worked example.

## Basic Usage

The `sampling` block controls both how many of the filtered rows are kept
and how they are chosen. `method` selects the strategy — either a built-in
name or a path to a plugin file — and the rest of the block supplies
whatever that strategy needs.

For the built-in strategies, that's just `n_samples` and an optional
`seed`:

```yaml
sampling:
  n_samples: 100000
  method: random
  seed: 42 # for reproducibility
```

`method: random` draws a uniform random subset of size `n_samples`;
`method: all` skips sampling entirely and keeps every filtered row. To use
your own selection logic instead, point `method` at a Python file and pass
it whatever columns and keyword arguments it needs:

```yaml
sampling:
  method: /path/to/custom_sampling.py
  required_columns: ["num_atoms", "energy"]
  kwargs:
    some_kwarg: 42
```

- `method` — `random`, `all`, or a path to a Python file defining a
  `custom_sampling_function` function.
- `n_samples` — number of rows to keep. Applies to `random` and to custom
  samplers that respect it; omit it (or set it to `null`) to let a custom
  sampler decide how many rows to keep on its own.
- `seed` — random seed used only for `method: random`.
- `required_columns` — extra column name(s) to load and pass to a custom
  sampler alongside `entry_key`, in this example, number of atoms and energy.
  Ignored by the built-in methods.
- `kwargs` — any additional keyword arguments forwarded to a custom
  sampler's `custom_sampling_function` function.

## Writing a Sampling Plugin

Create a standard Python file (e.g. `my_sampler.py`) that defines a top-level
`custom_sampling_function` function:

```python
# my_sampler.py
import polars as pl


def custom_sampling_function(
    frame: pl.LazyFrame | pl.DataFrame,
    n_samples: int | None,
    some_kwarg: int,
) -> list[str]:
    """Return the entry_key of every selected row.

    `some_kwarg` is an example for a keyword argument (here, not actually used below).
    """
    collected = frame.collect() if isinstance(frame, pl.LazyFrame) else frame
    return collected["entry_key"].to_list()[:n_samples]
```

`frame` is the filtered query result (with `entry_key` plus whatever columns
you requested via `required_columns`); `custom_sampling_function` must return the
list of selected `entry_key` values. The CLI loads the file with
`runpy.run_path` and raises `FileNotFoundError`/`ValueError` if the file or
the `custom_sampling_function` function is missing. The function name and
signature shown above must be followed exactly — the CLI calls
`custom_sampling_function(frame, n_samples, **kwargs)` positionally, so
renaming it or reordering its parameters will break the plugin.

```{note}
The plugin mechanism mirrors the "Custom I/O Plugins" section of the
[CLI guide](cli.md): a plain Python file loaded dynamically, no package
installation required.
```

## Example: Greedy Tanimoto Sampling

[`greedy_sampling.py`](https://github.com/instadeepai/chemreporter/blob/main/sampling_examples/greedy_sampling.py)
is primarily meant as a worked example of a custom sampling plugin: it
shows how to write a `custom_sampling_function` that streams through a
large filtered result set in chunks and selects rows based on a pairwise
distance computed between them. That said, it is also directly usable if
greedy diversity sampling fits your use case, not just a reference to
copy from.

The plugin greedily selects a chemically diverse subset based on molecular
fingerprint bit vectors: to compute fingerprint-based similarity,
we use the Tanimoto index[^tanimoto_paper]:
starting from one seed structure, it repeatedly adds the
next structure whose Tanimoto distance to every structure already selected
exceeds a threshold, streaming through the filtered rows in chunks so it
scales to large databases. It relies on [JAX](https://docs.jax.dev/) for
the distance computation, so make sure `jax` is installed in your
environment; it is not a dependency of ChemReporter itself.

Here is an example `sampling` config using it:

```yaml
sampling:
  method: /path/to/greedy_sampling.py
  required_columns: fingerprint_bits
  n_samples: null
  kwargs:
    seed: 42
    chunk_size: 25000
    min_distance_threshold: 0.65
```

- `n_samples: null` keeps adding structures until no remaining candidate
  clears the threshold, rather than stopping at a fixed count. Set it to an
  integer to cap the subset size at that count instead.
- `required_columns: fingerprint_bits` — `fingerprint_bits` is not itself a
  query database column; it's a special preset name that expands to every
  individual `fingerprint_*` bit column (`fingerprint_1`, `fingerprint_2`,
  ...) needed to compute Tanimoto distances, so you don't have to list them
  all yourself.
- `min_distance_threshold` — minimum Tanimoto **distance** (`1 -
  similarity`) a candidate must have to *every* structure already selected.
  Raise it for a more diverse (and smaller) subset, lower it to keep more
  structures.
- `chunk_size` — number of rows processed per streaming batch. Lower it if
  you are memory-constrained.
- `seed` — used to pick the random starting structure when you don't specify
  a `start_idx` yourself.

[^tanimoto_paper]: **Why Tanimoto?** Bajusz et al. demonstrated that the
Tanimoto index, alongside Dice and Cosine metrics, consistently yields optimal
similarity rankings for binary molecular fingerprints across unbiased compound
datasets ([Bajusz et al., 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4456712/)).
