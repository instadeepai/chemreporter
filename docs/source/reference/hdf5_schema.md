# Export Schema

When you run `chemreporter export`, the resulting HDF5 file is structured
specifically for training Machine Learning Interatomic Potentials (MLIPs).
The file contains a flat hierarchy of groups, where each group represents a
single chemical system (an "entry") keyed by its unique `entry_key`. This
format is directly compatible with the
[`mlip`](https://github.com/instadeepai/mlip) library for training
interatomic potentials.

Below is the internal layout of the generated HDF5 file:

```text
/
├── {entry_key_1}/
│   ├── attributes:
│   │   ├── energy (float, eV) - The raw total DFT energy
│   ├── datasets:
│   │   ├── elements (1D array of integers) - Atomic numbers (Z)
│   │   ├── positions (2D array of floats) - Atomic coordinates in Ångströms (Å)
│   │   ├── forces (2D array of floats) - Atomic forces in eV/Å
│   │   ├── pbc (1D array of booleans) - Periodic boundary conditions [x, y, z]
|   |   ├── charge (int) - the net_charge of the system  (if available)
|   |   ├── spin_multiplicity (int or float)  (if available)
│   │   └── stress (optional, 3x3 array of floats) - Stress tensor (if available)
│   │   ├── {other scalar fields from the source database (depend on source database)}
│   └── extras/
│       └── datasets: (optional, present if `extras_fields` were requested during export)
│           └── {other array fields from the query database}
│
├── {entry_key_2}/
│   └── ...
└── ...
```

## Details on the exported fields

1. **`energy`**: This is the **raw total DFT energy** taken directly from
   the source dataset, not the cohesive or atomization energy. MLIP training
   pipelines will generally need to compute or subtract isolated atom
   reference energies (E0) during preprocessing or within the training loop.
2. **`elements`**: A 1D integer array containing the atomic number (Z) of
   each atom in the structure.
3. **`pbc`**: The periodic boundary conditions, stored as a length-3 boolean
   array indicating whether the system is periodic along the x, y, and z
   directions, respectively.
4. **`stress`**: The stress tensor, if available, is always exported as a full
   3x3 matrix (not a 6-element Voigt vector).
5. Note: some fields depend on the source dataset (e.g., in OMOL25 both Löwdin and
   Mulliken partial charges are exported. Please refer to
   [Source Datasets](../source_datasets/source_datasets.md) for details on
   dataset-specific implementations.
6. **`extras`**: If you specified `extras_fields` in your `export.yaml`
   config. Any field of the query database can be exported, (such as `smiles`,
   `net_charge`, `database_name`, etc.). They will be
   stored in a nested `extras` group. Scalar values (strings, ints, floats)
   are stored as HDF5 attributes on the `extras` group, while array-like
   values are stored as HDF5 datasets within that group.
