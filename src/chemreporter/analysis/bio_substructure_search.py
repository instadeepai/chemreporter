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
"""RDKit substructure search for biological motifs (L-amino acids, nucleobases).

Each motif family is defined by a JSON file under ``chemreporter/data/``. Amino-acid
entries use a single SMARTS string. Nucleobase entries use one or more SMILES/SMARTS
strings (from Open Babel fragments of PDB nucleotides); multiple patterns are merged
with maximal-set dedupe so the same residue is not counted twice.
"""

from __future__ import annotations

import json
import logging
from functools import cache
from pathlib import Path
from typing import Any

from rdkit import Chem

logger = logging.getLogger("chemreporter")

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"

AMINO_ACIDS_JSON = "amino_acids_L.json"
NUCLEOBASES_JSON = "nucleobases.json"

PROTEIN_RESIDUE_THRESHOLD = 3
NUCLEOBASE_MOTIF_THRESHOLD = 2


@cache
def _pattern_string_lists(json_filename: str) -> dict[str, list[str]]:
    path = _DATA_DIR / json_filename
    with path.open(encoding="utf-8") as f:
        raw: dict[str, dict[str, str | list[str]]] = json.load(f)
    out: dict[str, list[str]] = {}
    for code, entry in raw.items():
        s = entry["smarts"]
        out[code] = [s] if isinstance(s, str) else list(s)
    return out


@cache
def _query_mol_lists(json_filename: str) -> dict[str, list[Chem.Mol]]:
    out: dict[str, list[Chem.Mol]] = {}
    for code, patterns in _pattern_string_lists(json_filename).items():
        mols: list[Chem.Mol] = []
        for smarts in patterns:
            q = Chem.MolFromSmarts(smarts)
            if q is None:
                logger.warning(
                    "RDKit could not parse SMARTS for %s in %s: %s",
                    code,
                    json_filename,
                    smarts,
                )
                continue
            mols.append(q)
        out[code] = mols
    return out


def _maximal_match_atom_sets(
    mol: Chem.Mol, queries: list[Chem.Mol]
) -> list[frozenset[int]]:
    """Keep only maximal atom-index sets among all substructure matches.

    Drops strict subsets so one residue is not counted twice across patterns.

    Returns:
        List of frozensets of matched atom indices, one per counted occurrence.
    """
    sets: list[frozenset[int]] = []
    for q in queries:
        for m in mol.GetSubstructMatches(q, uniquify=True):
            sets.append(frozenset(m))
    maximal: list[frozenset[int]] = []
    for s in sorted(sets, key=len, reverse=True):
        if any(s < t for t in maximal):
            continue
        maximal = [t for t in maximal if not (t < s)]
        maximal.append(s)
    return maximal


def _num_field(code: str) -> str:
    return f"num_{code.lower()}"


def _count_smarts_in_mol(
    mol: Chem.Mol | None,
    *,
    json_filename: str,
    is_biomolecule_key: str,
    threshold: int,
) -> dict[str, Any]:
    table = _pattern_string_lists(json_filename)
    codes = sorted(table.keys())
    counts: dict[str, Any] = {_num_field(c): 0 for c in codes}
    counts[is_biomolecule_key] = False
    if mol is None:
        return counts

    queries = _query_mol_lists(json_filename)
    total = 0
    for code in codes:
        qs = queries.get(code) or []
        if not qs:
            continue
        key = _num_field(code)
        n = len(_maximal_match_atom_sets(mol, qs))
        counts[key] = n
        total += n

    counts[is_biomolecule_key] = total > threshold
    return counts


def count_amino_acids_in_mol(mol: Chem.Mol | None) -> dict[str, Any]:
    """Count L-amino-acid SMARTS matches and set ``is_protein`` from threshold.

    Returns:
        Dict with ``num_*`` keys per residue type and boolean ``is_protein``.
    """
    return _count_smarts_in_mol(
        mol,
        json_filename=AMINO_ACIDS_JSON,
        is_biomolecule_key="is_protein",
        threshold=PROTEIN_RESIDUE_THRESHOLD,
    )


def count_nucleobases_in_mol(mol: Chem.Mol | None) -> dict[str, Any]:
    """Count nucleobase pattern matches and set ``is_nucleobase`` from threshold.

    Returns:
        Dict with ``num_*`` keys per base code and boolean ``is_nucleobase``.
    """
    return _count_smarts_in_mol(
        mol,
        json_filename=NUCLEOBASES_JSON,
        is_biomolecule_key="is_nucleobase",
        threshold=NUCLEOBASE_MOTIF_THRESHOLD,
    )


def count_all_bio_substructures_in_mol(mol: Chem.Mol | None) -> dict[str, Any]:
    """Merge amino-acid and nucleobase count dicts for one RDKit molecule.

    Returns:
        Dict with all ``num_*`` keys, ``is_protein``, and ``is_nucleobase``.
    """
    return {
        **count_amino_acids_in_mol(mol),
        **count_nucleobases_in_mol(mol),
    }
