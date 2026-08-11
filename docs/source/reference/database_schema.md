# Query Database Schema

This page lists every field of the Query Database that can be referenced in a
`query` step's filter expression (see [Query Examples](query_examples.md) for
example syntax). This table is generated directly from the
[`chemreporter.query_database_tools.table_schemas`](https://github.com/instadeepai/chemreporter/tree/main/src/chemreporter/query_database_tools/table_schemas.py)
module.

<!-- schema-fields -->

For more detailed information about the units used for each field, see
[Units and Physical Quantities](units.md).

## Known caveats

`num_water_molecules` might be slightly overestimated: compounds containing an
R-OHH group are counted as containing one water molecule. Such cases are
assumed to be sufficiently rare, so this approximation is not currently
corrected for.
