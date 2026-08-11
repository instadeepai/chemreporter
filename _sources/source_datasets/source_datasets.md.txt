# Supported Source Datasets

ChemReporter currently supports the following source datasets:

- **OMOL25** – Molecular systems, including bond-breaking, computed with hybrid DFT.
- **OC20** – Periodic surface systems for adsorption and catalysis studies.
- **OMAT24** – Periodic bulk inorganic materials.
- **OMC25** – Periodic organic molecular crystals, relaxed with dispersion-corrected
  DFT.
- **ODAC** – Periodic MOF systems for direct air capture (DAC) applications.

Each dataset is handled by its own reader implementation, and these
implementations are not interchangeable: they differ in how properties like
`subset` are populated, which fields are read as-is from the source, and
which are computed by ChemReporter. Read
[Crucial Implementation Details](#crucial-implementation-details) to
understand these differences before processing and querying datasets.

---

(appendix-dft-setup-details)=
## Computational Setup Details

The table below compares the computational details used across the
supported source datasets. Column headers link to the primary reference
paper for each dataset.

| Setting | [OMOL25](https://arxiv.org/pdf/2505.08762) | [OC20](https://pubs.acs.org/doi/10.1021/acscatal.0c04525) | [OMAT24](https://arxiv.org/html/2410.12771v1) | [OMC25](https://www.nature.com/articles/s41597-026-06628-2) | [ODAC](https://pubs.acs.org/doi/10.1021/acscentsci.3c01629) |
|----------|------------|----------|------------|------------|----------|
| **DFT Code** | ORCA 6.0.0 | VASP | VASP | VASP | Not reported |
| **System Type** | Molecular (non-periodic) | Periodic (surfaces) | Periodic (bulk materials) | Periodic (molecular crystals) | Periodic MOFs (DAC) |
| **Basis Representation** | Gaussian | Plane waves | Plane waves | Plane waves | Not reported |
| **Basis Set** | def2-TZVPD | — | — | — | — |
| **Pseudopotentials / ECPs** | def2 ECPs (elements 1–83 supported) | PAW | PAW | PAW | Not reported |
| **Plane-Wave Cutoff** | — | 350 eV | 520 eV (assuming MP default) | 520 eV | Not reported |
| **Exchange–Correlation** | ωB97M-V | RPBE (GGA) | PBE (GGA) | PBE (GGA) | PBE (GGA) |
| **Dispersion Treatment (Correction)** | VV10 nonlocal correlation | D3 | Hubbard +U | Grimme D3 | D3 (PBE-D3) |
| **Spin Treatment** | UKS ; RKS also evaluated on subset| Spin-polarized when required | Spin-polarized | As set by relaxation workflow | Spin-polarized, initial magnetic moment +1 for all atoms |
| **Integral / Acceleration** | RI-J, COSX | — | — | — | Not reported |
| **Grid / k-Points** | DEFGRID3 (590 XC / 302 COSX angular points) | Monkhorst–Pack mesh | MP default k-point density | Via atomate2 `RelaxSetGenerator` | Not reported |
| **Convergence Settings** | Tight SCF; `thresh=1e-12`; `tcut=1e-13` | Force < 0.03 eV/Å or ≤200 ionic steps | Force < 0.03 eV/Å or ≤200 ionic steps | Max per-atom residual force < 0.001 eV/Å or ≤1,500 ionic steps; total energy tolerance 0.001 meV | Not reported |



(crucial-implementation-details)=
## Crucial Implementation Details

Each source dataset is read by a dedicated implementation, selected via the
`database_format` field of the `process` config
(see [Configuration](../reference/configs.md)). This maps to the following
formats and readers:

| `database_format` | Dataset | File format | Reader |
|---|---|---|---|
| `aselmdb_omol` | OMOL25 | `*.aselmdb` | FairChem `AseDBDataset` |
| `aselmdb_omat` | OMAT24 | `*.aselmdb` | FairChem `AseDBDataset` |
| `aselmdb_oc` | OC20 | `*.aselmdb` | FairChem `AseDBDataset` |
| `aselmdb_odac` | ODAC23 | `*.aselmdb` | FairChem `AseDBDataset` |
| `aselmdb_omc` | OMC25 | `*.aselmdb` | FairChem `AseDBDataset` |
| `xyz_oc` | OC20 (extxyz) | `*.extxyz.xz` | `ase.io` + `.pkl`/`.txt` auxiliary files |
| `xyz` | generic extxyz | `*.xyz` | `ase.io` |

### Common Behavior Across Datasets

Beyond the per-dataset differences covered below, a few behaviors are shared
by every implementation:

- **`database_name` affects more than display.** It is lowercased and `_` is
  replaced by `-`, and it becomes part of every `entry_key`. For OC20 and
  OMAT24 it *also* carries the subset (see below), so a bare `oc20` or
  `omat24` raises `SourceDatabaseReaderUsageError`.
- **DFT metadata** (`basis_set`, `functional`, `correction_term`) is not read
  from the data itself; it comes verbatim from the `source_database_metadata`
  field of your `process` config (see [Configuration](../reference/configs.md))
  and is applied to every entry processed from that source database. It is
  optional, so leaving it out silently yields empty metadata columns.
- **Charge and spin are either read from the source or enforced.**
  OMOL25 and generic `xyz` store these values explicitly and
  they are read verbatim. All other datasets (OC20, OMAT24, OMC25, ODAC23)
  enforce `net_charge = 0` (neutral) and `spin_multiplicity = 1`
  (closed-shell) where applicable.
- **`subset` drives graph-feature skipping.** The `subsets_skip_list` list,
  set under `graph_based_processing` in your `process` config, is matched
  against `subset` to decide which entries skip graph-feature computation —
  so it has no effect for formats that never populate `subset`.
- **Remote sources** (`s3://`, …) are supported out of the box: files are
  downloaded into a temporary directory shard by shard and deleted again once
  each shard has been processed. See
  [Custom I/O Plugins](custom-io-plugins) if you need to customize
  this behavior.

### Dataset-Specific Details

#### OMOL25 (`aselmdb_omol`)

- **Subsets come from the data.** `subset` is read directly from
  `atoms.info["data_id"]`, so splits such as `spice`, `biomolecules`, and
  `elytes` appear automatically — no naming convention is required on your
  side.
- `net_charge` (from `charge`), `spin_multiplicity` (from `spin`), `num_atoms`,
  and `composition` are all read straight from the source — this is the only
  dataset where all four fields are authoritative rather than inferred.
- **Reactivity metadata is parsed from free text.** For the `rgd`,
  `reactivity`, `ani1`, `trans1x`, and `metal_complexes` subsets, the
  free-text `source` field (and, for metal complexes, `reference_source`) is
  parsed into `reaction_id`, `reaction_pathway_id`, `reaction_step_idx`,
  `is_reactant`, and `is_product`. These columns stay null for every other
  subset. Unrecognized `source` strings yield nulls rather than raising, so
  partially-null reactivity columns are expected, not a bug.
- The `metal_complexes` subset **renames itself** to
  `reactivity_metal_complexes`, `ground_state_metal_complexes`, or
  `failed_metal_complexes` — filter on those names rather than on
  `metal_complexes` itself.
- The generic `reactivity` subset also **renames its own `subset`**, based on
  the first path segment of the free-text `source` field — known values
  include `ani1xbb`, `pmechdb`, and `rmechdb` — so filter on the resulting
  name rather than on `reactivity` itself.
- The HDF5 export automatically includes `subset`, `charge`,
  `spin_multiplicity`, `num_atoms`, `mulliken_charges`, and `lowdin_charges`.

#### OC20 (`aselmdb_oc`, `xyz_oc`)

- **The subset isn't in the data — it's in the name.** `subset` is derived as
  the last token of `database_name` after splitting on non-alphanumeric
  characters, so you *must* set `database_name` to something like
  `oc20-s2ef` or `oc20-is2re`. Only the last token is kept, so
  `oc20-s2ef-all` yields subset `all`.
- **Needs supplementary files alongside the shards**, for both `aselmdb_oc`
  and `xyz_oc`: a single `oc20_data_mapping.pkl` in the same directory, plus
  a per-shard auxiliary file (`<shard>.txt` or `<shard>.txt.xz`) providing
  `sid`, `frame_number`, and `ref_energy`. A `sid` missing from the pickle
  raises an error, as does a shard missing from the mapping.
- **Charge and spin are enforced.** `net_charge` is fixed at
  `0` (periodic slab supercells are neutral by construction), and
  `spin_multiplicity` is fixed at `1` (closed-shell).
- **OC20 gets extra analysis**  This includes
  `catalyst_adsorbate_smiles` (OpenBabel is ran on the adsorbate alone),
  Note that adsorbate net charges are constrained within the range of -3 to +3,
  with the implementation prioritising neutral states whenever viable.
  `catalyst_num_adsorbate_atoms`, `catalyst_num_bulk_atoms`,
  `catalyst_substrate_height` (measured along the true surface normal, not
  the z-axis), `catalyst_xyz_adsorbate_is_valid` (whether the adsorbate sits
  above the slab and within its in-plane footprint, vdW-padded and
  minimum-image aware), and `is_molecular_structure_valid` (adsorbate plus
  nearby surface atoms). It also adds `catalyst_bulk_id`,
  `catalyst_adsorbate_id`, `catalyst_bulk_symbols`,
  `catalyst_adsorbate_symbols`, `catalyst_miller_index`,
  `catalyst_reference_energy`, `catalyst_relaxation_frame_idx`, and the
  decoded labels `catalyst_class` (intermetallics / metalloids / non-metals /
  halides) and `catalyst_anomaly` (no_anomaly / adsorbate_dissociation /
  adsorbate_desorption / surface_reconstruction / uninteracting_hydrogen).
- All `catalyst_*` columns exist only for OC20 — they are null for every
  other dataset.
- The HDF5 export automatically includes `num_atoms`.

#### OMAT24 (`aselmdb_omat`)

- **The subset comes from the name, not the data.** `task_type` is initially
  mapped to `subset`, but this is then overwritten with everything after the
  first delimiter of `database_name`, joined back together with `-` (so
  `omat24-rattled-1000` becomes `rattled-1000`). Process one subset directory
  at a time and name it accordingly — a bare `omat24` raises an error.
- `composition` is taken from `composition_reduced`, so — unlike every other
  dataset, where it reflects the full cell content — it is the *reduced*
  chemical formula.
- `num_atoms` is recomputed. `net_charge` is enforced at `0` (bulk cells are
  neutral by construction). `spin_multiplicity` is not set for OMAT24.
- The HDF5 export automatically includes `subset`, and `stress`.

#### OMC25 (`aselmdb_omc`)

- **No subset at all.** Nothing is mapped or derived, so `subset` always
  stays null — use `database_name` instead to distinguish between OMC25
  runs.
- Everything else is derived from geometry: `composition` (full formula) and
  `num_atoms`. `net_charge` is enforced at `0` and `spin_multiplicity` at
  `1` (closed-shell)
- For OMC25, set `source_database_metadata` in your `process` config to
  `basis_set: PAW-PW`, `functional: PBE`, and `correction_term: D3` (see the
  [Computational Setup Details](#appendix-dft-setup-details) table above).
- The HDF5 export automatically includes `stress` (as a 3x3 matrix).

#### ODAC23 (`aselmdb_odac`)

- **No subset**
- `num_atoms` and `composition` are computed. `net_charge` is enforced at
  `0` and `spin_multiplicity` at `1` (closed-shell).
- No extra analysis columns and no extra HDF5 fields are automatically exported.

#### Generic extxyz (`xyz`)

- This is the escape hatch for your own data: `subset` is read from
  `config_type`, `net_charge` from `total_charge`, `spin_multiplicity` from
  `spin`, and `num_atoms` from `num_atoms`. `energy` and `forces` are
  rehydrated into a `SinglePointCalculator` when present in `info`/`arrays`.
- Nothing else is derived, so including those four keys in your extxyz
  headers is the cheapest way to get a fully populated Query Database.
- The HDF5 export automatically includes `subset`, `charge`, `spin_multiplicity`, and `num_atoms`.

### What to check before querying

- Is `subset` populated for your dataset? For OMC25 and ODAC23 it isn't — filter
  on `database_name` / `split_name` instead.
- For OC20 and OMAT24, did you encode the subset in `database_name`?
- Are `net_charge` / `spin_multiplicity` authoritative or enforced for the
  dataset you're mixing in? OMOL25 and generic `xyz` read them from the
  source (ground truth). OC20, OMC25, and ODAC23 enforce `0` / `1` for all
  entries. OMAT24 enforces `net_charge = 0` but leaves `spin_multiplicity`
  null. Mixing datasets means mixing ground truth with enforced constants.
- Dataset-specific columns (`reaction_*`, `catalyst_*`, `stress`,
  `mulliken_charges`, `lowdin_charges`) are null outside their source
  dataset — filtering on them silently drops every other dataset.
