# Units and Physical Quantities

This page describes the units of measurement used throughout ChemReporter.
All units are handled by ASE (Atomic Simulation Environment) and remain
consistent across the entire processing pipeline, from raw source datasets
to the final exported HDF5 files.

## Standard Units

ChemReporter uses the following standard unit for each physical quantity:

### Energy
- **Electronvolt (eV)** — used in ASE database files and throughout the
  codebase for all energy values.

### Distance
- **Ångström (Å)** — used for atomic positions and distances.

### Force
- **eV/Å** — used to compute `net_force_norm` and `max_force_norm`.

### Molecular Weight
- **g/mol (atomic mass units)** — computed from the atomic masses in the
  ASE `Atoms` object.

### Dipole Moment
- **Debye** — used to express the magnitude of the dipole moment.

## Unit Conversions

ChemReporter relies on the unit conversion constants provided by ASE, so all
physical quantities are automatically handled in the standard units listed
above. For more details, see the
[ASE Units Documentation](https://docs.ase-lib.org/ase/units.html#module-ase.units).
