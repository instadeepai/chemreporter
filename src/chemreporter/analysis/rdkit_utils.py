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
import functools
from typing import Any

import ase
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, SanitizeFlags, rdFingerprintGenerator, rdmolops

from chemreporter.analysis.bio_substructure_search import (
    count_all_bio_substructures_in_mol,
)
from chemreporter.analysis.obabel_utils import smiles_from_positions

# Disable RDKit warnings globally
RDLogger.DisableLog("rdApp.*")

RADIUS = 2
FP_SIZE = 1024


@functools.cache
def _fingerprint_keys(size: int) -> tuple[str, ...]:
    """Return the shared column keys for a fingerprint of the given size.

    The result is cached so the ``fingerprint_i`` key strings are built once and
    reused across every molecule instead of being reallocated per row.

    Args:
        size: Number of fingerprint bits.

    Returns:
        Tuple of ``fingerprint_0`` .. ``fingerprint_{size - 1}`` keys.
    """
    return tuple(f"fingerprint_{i}" for i in range(size))


@functools.cache
def get_fpgen(
    radius: int = RADIUS, fp_size: int = FP_SIZE
) -> rdFingerprintGenerator.FingerprintGenerator64:
    """Get the fingerprint generator.

    Args:
        radius: radius of the fingerprint
        fp_size: size of the fingerprint

    Returns:
        fingerprint generator
    """
    return rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=fp_size)


def mol_from_openbabel_smiles(
    smiles: str, positions: list[list[float]]
) -> Chem.Mol | None:
    """RDKit mol from Open Babel SMILES, with a fallback when default sanitize fails.

    OB-generated peptide/macrocycle SMILES sometimes trip RDKit's valence checks.
    We then parse with ``sanitize=False``, skip ``SANITIZE_PROPERTIES`` (valence),
    and run ``FastFindRings`` so ``GetSubstructMatches`` can run.

    Returns:
        Parsed molecule, or ``None`` if parsing or fallback sanitization fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        _update_mol_positions(mol, positions)
        return mol
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(
            mol,
            sanitizeOps=SanitizeFlags.SANITIZE_ALL ^ SanitizeFlags.SANITIZE_PROPERTIES,
        )
        rdmolops.FastFindRings(mol)
        _update_mol_positions(mol, positions)
    except (
        Chem.AtomValenceException,
        Chem.AtomKekulizeException,
        ValueError,
        RuntimeError,
    ):
        return None
    return mol


def _update_mol_positions(mol: Chem.Mol, positions: list[list[float]]):
    """Update the positions of the molecule.

    Args:
        mol: RDKit molecule
        positions: list of positions

    Returns:
        The molecule with updated positions.
    """
    if len(positions) != mol.GetNumAtoms():
        return None

    conf = Chem.Conformer(mol.GetNumAtoms())

    for idx, (x, y, z) in enumerate(positions):
        conf.SetAtomPosition(idx, (float(x), float(y), float(z)))

    mol.RemoveAllConformers()
    mol.AddConformer(conf)
    return mol


def calculate_graph_derived_properties(
    atoms: ase.Atoms,
    net_charge: int,
    fpgen: rdFingerprintGenerator.FingerprintGenerator64 | None,
) -> dict[str, Any]:
    """Graph descriptors, optional fingerprint, and bio motif counts.

    Args:
        atoms: ASE atoms object
        net_charge: net charge of the molecule
        fpgen: fingerprint generator

    Returns:
        dictionary with the properties and fingerprint
        if the molecule is not built, returns empty dictionary

        dictionary with the following keys:
            - molecular_weight: float
            - logp: float
            - tpsa: float
            - smiles: str
            - fingerprints as separate entries: f_0, f_1, ..., f_1023
            - amino-acid counts as separate entries: num_ala, num_arg, ..., num_val
            - Nucleotide base counts as separate entries: num_ade, num_ura, ...
            if the molecule is not built, with returns None
    """
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    positions = atoms.get_positions().tolist()
    smiles = smiles_from_positions(
        atomic_numbers=atomic_numbers, positions=positions, net_charge=net_charge
    )
    if len(smiles) == 0:  # empty string
        return {"smiles": None}
    rdkit_mol = mol_from_openbabel_smiles(smiles, positions)

    if rdkit_mol is None:
        return {"smiles": None}
    properties = get_graph_derived_properties(rdkit_mol)

    bio_substructure_counts = count_all_bio_substructures_in_mol(rdkit_mol)

    if fpgen is None:
        rdkit_mol = None
        return {"smiles": smiles, **properties, **bio_substructure_counts}

    fingerprint = get_fingerprint(rdkit_mol, fpgen)
    rdkit_mol = None

    return {
        "smiles": smiles,
        **properties,
        **fingerprint,
        **bio_substructure_counts,
    }


def get_fingerprint(
    rdkit_mol: Chem.Mol, fpgen: rdFingerprintGenerator.FingerprintGenerator64
) -> dict:
    """Get the fingerprint of the molecule.

    Args:
        rdkit_mol: RDKit molecule
        fpgen: fingerprint generator

    Returns:
        dictionary with the fingerprint as separate entries: fingerprint_0
        fingerprint_1, ..., fingerprint_1023
    """
    fingerprint = fpgen.GetFingerprint(rdkit_mol)
    fp = np.frombuffer(fingerprint.ToBitString().encode("ascii"), "S1").astype(int)

    values = fp.tolist()
    return dict(zip(_fingerprint_keys(len(values)), values))


def get_graph_derived_properties(rdkit_mol: Chem.Mol) -> dict:
    """Get the properties of the molecule.

    Args:
        rdkit_mol: RDKit molecule

    Returns:
        dictionary with the properties Use the schema to get the keys
        dictionary contains:
            - molecular_weight: float
            - logp: float
            - tpsa: float
            - smiles: str
    """
    result = {
        "logp": Descriptors.MolLogP(rdkit_mol),
        "tpsa": Descriptors.TPSA(rdkit_mol),
        "smiles": Chem.MolToSmiles(rdkit_mol),
    }

    return result
