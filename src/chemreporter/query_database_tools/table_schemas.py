# Copyright 2026 InstaDeep Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import polars as pl

_MOTIF_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@cache
def _motif_codes_from_json(filename: str) -> tuple[str, ...]:
    path = _MOTIF_DATA_DIR / filename
    with path.open(encoding="utf-8") as f:
        raw: dict[str, dict[str, str]] = json.load(f)
    return tuple(sorted(raw.keys()))


AMINO_ACID_CODES = _motif_codes_from_json("amino_acids_L.json")
NUCLEOBASE_CODES = _motif_codes_from_json("nucleobases.json")


@dataclass
class SchemaField:
    """One column in the query-database Parquet schema."""

    name: str
    polars_type: pl.DataType
    description: str = "no description"
    unit: str | None = None


def get_schema_dict(schema: list[SchemaField]) -> dict[str, pl.DataType]:
    """Get the schema dictionary.

    Returns:
        A dictionary of the fields in the schema.
    """
    return {field.name: field.polars_type for field in schema}


BASE_SCHEMA = [
    SchemaField("key", pl.String, "Unique identifier for each structure"),
    SchemaField("entry_key", pl.String, "Unique identifier for each structure"),
    SchemaField(
        "database_name",
        pl.String,
        "Name of the source database (set when processing)",
    ),
    SchemaField("split_name", pl.String, "Split name (set when processing)"),
    SchemaField("basis_set", pl.String, "Basis set used for the DFT calculation"),
    SchemaField("functional", pl.String, "Functional used for the DFT calculation"),
    SchemaField("correction_term", pl.String, "Dispersion or post-DFT correction term"),
    SchemaField(
        "subset",
        pl.String,
        "Dataset subset name (for example spice, anix, biomolecules)",
    ),
    SchemaField("composition", pl.String, "Brutto formula of the structure"),
    SchemaField("net_charge", pl.Int64, "Net molecular charge", unit="e"),
    SchemaField("spin_multiplicity", pl.Float64, "Spin multiplicity", unit="unitless"),
    SchemaField("num_atoms", pl.Int64, "Number of atoms"),
    SchemaField("atomic_numbers", pl.List(pl.Int64), "Atomic numbers by atom"),
    SchemaField(
        "energy",
        pl.Float64,
        "Raw DFT total energy (not cohesive/atomization energy)",
        unit="eV",
    ),
]

CALCULATED_SCHEMA = [
    SchemaField(
        "dipole_moment_magnitude",
        pl.Float64,
        "Magnitude of dipole moment",
        unit="Debye",
    ),
    SchemaField("net_force_norm", pl.Float64, "Norm of net force vector", unit="eV/Å"),
    SchemaField(
        "max_force_norm", pl.Float64, "Maximum force magnitude on any atom", unit="eV/Å"
    ),
    SchemaField(
        "atomic_symbols",
        pl.String,
        "Unique element symbols in alphabetical order",
    ),
    SchemaField("molecular_weight", pl.Float64, "Molecular weight", unit="g/mol"),
    SchemaField(
        "is_molecular_structure_valid",
        pl.Boolean,
        "Historic structure check (hydrogen coordination)",
    ),
    SchemaField("num_water_molecules", pl.Int64, "Number of water molecules"),
]

AMINO_ACID_RESIDUE_SCHEMA = [
    SchemaField(
        f"num_{code.lower()}",
        pl.Int64,
        (
            f"RDKit substructure count for L-{code} residue motif "
            "(SMARTS from amino_acids_L.json)"
        ),
    )
    for code in AMINO_ACID_CODES
] + [
    SchemaField(
        "is_protein",
        pl.Boolean,
        "True when the sum of amino-acid motif counts exceeds 3",
    ),
]

NUCLEOBASE_RESIDUE_SCHEMA = [
    SchemaField(
        f"num_{code.lower()}",
        pl.Int64,
        (
            f"RDKit substructure count for nucleobase {code} "
            "(patterns from nucleobases.json)"
        ),
    )
    for code in NUCLEOBASE_CODES
] + [
    SchemaField(
        "is_nucleobase",
        pl.Boolean,
        "True when the sum of nucleobase motif counts exceeds 3",
    ),
]

GRAPH_PROP_SCHEMA = [
    SchemaField(
        "graph_properties_candidate",
        pl.Boolean,
        "Whether the structure is suitable for graph-based analysis",
    ),
    SchemaField(
        "error_graph_properties",
        pl.Boolean,
        "Whether graph property calculation failed",
    ),
    SchemaField(
        "logp",
        pl.Float64,
        "Partition coefficient (lipophilicity)",
        unit="unitless",
    ),
    SchemaField(
        "tpsa",
        pl.Float64,
        "Topological polar surface area",
        unit="Å²",
    ),
    SchemaField("smiles", pl.String, "SMILES representation"),
]

FINGERPRINTS_SCHEMA = [
    SchemaField(f"fingerprint_{i}", pl.Int64, f"Morgan fingerprint bit {i}")
    for i in range(1024)
]

REACTION_SCHEMA = [
    SchemaField("reaction_id", pl.String, "Reaction identifier"),
    SchemaField("reaction_pathway_id", pl.Int64, "Pathway identifier (if applicable)"),
    SchemaField("is_reactant", pl.Boolean, "Whether the structure is a reactant"),
    SchemaField("is_product", pl.Boolean, "Whether the structure is a product"),
    SchemaField(
        "is_transition_state", pl.Boolean, "Whether the structure is a transition state"
    ),
    SchemaField("reaction_step_idx", pl.Int64, "Index of the reaction step"),
]

CATALYST_SCHEMA = [
    SchemaField("catalyst_bulk_id", pl.String, "Catalyst bulk identifier"),
    SchemaField("catalyst_adsorbate_id", pl.String, "Catalyst adsorbate identifier"),
    SchemaField("catalyst_bulk_symbols", pl.String, "Element symbols in the bulk"),
    SchemaField(
        "catalyst_adsorbate_symbols", pl.String, "Element symbols in the adsorbate"
    ),
    SchemaField("catalyst_adsorbate_smiles", pl.String, "SMILES of the adsorbate"),
    SchemaField(
        "catalyst_num_adsorbate_atoms",
        pl.Int64,
        "Number of atoms in the adsorbate",
    ),
    SchemaField("catalyst_num_bulk_atoms", pl.Int64, "Number of atoms in the bulk"),
    SchemaField(
        "catalyst_reference_energy",
        pl.Float64,
        "Reference energy of the catalyst system",
        unit="eV",
    ),
    SchemaField("catalyst_class", pl.String, "Catalyst class label"),
    SchemaField("catalyst_anomaly", pl.String, "Catalyst anomaly label"),
    SchemaField(
        "catalyst_substrate_height",
        pl.Float64,
        "Substrate height",
        unit="Å",
    ),
    SchemaField(
        "catalyst_miller_index",
        pl.List(pl.Int64),
        "Miller indices (h, k, l) of the surface",
    ),
    SchemaField("catalyst_relaxation_frame_idx", pl.Int64, "Relaxation frame index"),
    SchemaField(
        "catalyst_xyz_adsorbate_is_valid",
        pl.Boolean,
        "Whether adsorbate coordinates pass the OC20 xyz validity check",
    ),
]

FULL_SCHEMA = (
    BASE_SCHEMA
    + CALCULATED_SCHEMA
    + AMINO_ACID_RESIDUE_SCHEMA
    + NUCLEOBASE_RESIDUE_SCHEMA
    + GRAPH_PROP_SCHEMA
    + FINGERPRINTS_SCHEMA
    + REACTION_SCHEMA
    + CATALYST_SCHEMA
)
