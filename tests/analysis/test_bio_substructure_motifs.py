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
"""Amino-acid and nucleobase motif detection (SMARTS) and PDB fixture smoke tests."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
from rdkit import Chem

from chemreporter.analysis.bio_substructure_search import (
    NUCLEOBASE_MOTIF_THRESHOLD,
    PROTEIN_RESIDUE_THRESHOLD,
    count_amino_acids_in_mol,
    count_nucleobases_in_mol,
)
from chemreporter.analysis.obabel_utils import smiles_from_positions
from chemreporter.analysis.rdkit_utils import mol_from_openbabel_smiles
from chemreporter.query_database_tools.table_schemas import (
    AMINO_ACID_CODES,
    NUCLEOBASE_CODES,
)

_FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "pdb"
PDB_2LUF = _FIXTURES / "2LUF.pdb"
PDB_3Q51 = _FIXTURES / "3Q51.pdb"
PDB_3CGY = _FIXTURES / "3CGY.pdb"
PDB_8C3U = _FIXTURES / "8C3U.pdb"


# --- PDB fixtures ---

_EXPECTED_2LUF_SEQRES = {
    "Ser": 3,
    "Pro": 4,
    "Arg": 1,
    "Gly": 3,
    "Asp": 1,
    "Lys": 1,
    "Leu": 2,
    "Trp": 1,
    "Gln": 1,
    "Ile": 1,
    "Tyr": 1,
    "Asn": 1,
}

# Deposited SEQRES lists 12 A; protonated fixture has 11 adenosine residues (OB count).
_EXPECTED_3Q51_SEQRES = {
    "Ade": 11,
    "Cyt": 9,
    "Gua": 7,
    "Ura": 5,
}

# Full asymmetric unit: one count per (chain, resseq, insertion, resname) from
# ATOM lines on the fixture.
_EXPECTED_3CGY_COMPOSITION = {
    "Ala": 27,
    "Arg": 16,
    "Asn": 18,
    "Asp": 27,
    "Cys": 6,
    "Gln": 19,
    "Glu": 30,
    "Gly": 30,
    "His": 12,  # 13 histidine in PDB, 1 is missed (distorted geometry)
    "Ile": 30,
    "Leu": 35,
    "Lys": 10,
    "Met": 9,
    "Phe": 15,
    "Pro": 14,  # 16 in PDB, 2 prolines are missed
    "Ser": 23,
    "Thr": 7,
    "Tyr": 9,
    "Val": 38,
}

_EXPECTED_8C3U_COMPOSITION = {
    "Ala": 10,
    "Arg": 6,
    "Asn": 18,
    "Asp": 16,
    "Cys": 4,
    "Gln": 24,
    "Glu": 22,
    "Gly": 16,
    "His": 1,  # 2 histidine in PDB, 1 is missed
    "Ile": 10,
    "Leu": 30,
    "Lys": 30,
    "Met": 12,
    "Phe": 18,
    "Pro": 16,
    "Ser": 26,
    "Thr": 12,
    "Trp": 2,
    "Tyr": 8,
    "Val": 22,
}


def _mol_from_pdb_first_model(pdb_path: Path) -> Chem.Mol | None:
    atoms = read(pdb_path, index=0)
    smiles = smiles_from_positions(
        atomic_numbers=atoms.get_atomic_numbers().tolist(),
        positions=atoms.get_positions().tolist(),
        net_charge=int(atoms.info.get("charge", 1)),
    )
    return mol_from_openbabel_smiles(smiles, atoms.get_positions().tolist())


# --- Schema / unit-style tests (amino acids) ---


def test_amino_acid_field_names_match_json():
    """Schema field names align with amino-acid codes."""
    names = [f"num_{c.lower()}" for c in AMINO_ACID_CODES]
    assert "num_ala" in names
    assert "num_gly" in names
    assert "num_leu" in names
    assert len(names) == 20


def test_amino_acid_count_alanine_free():
    """Single alanine SMILES yields at least one alanine motif match."""
    mol = Chem.MolFromSmiles("N[C@@H](C)C(=O)O")
    assert mol is not None
    out = count_amino_acids_in_mol(mol)
    assert out["num_ala"] >= 1
    assert out["is_protein"] is False


def test_amino_acid_is_protein_when_more_than_three_residues():
    """Short peptide exceeds protein motif-count threshold."""
    smi = "N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O"
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    out = count_amino_acids_in_mol(mol)
    total_aa = sum(out[f"num_{c.lower()}"] for c in AMINO_ACID_CODES)
    assert total_aa > PROTEIN_RESIDUE_THRESHOLD
    assert out["is_protein"] is True


def test_amino_acid_none_mol_returns_zeros():
    """None mol yields zero counts and is_protein false."""
    out = count_amino_acids_in_mol(None)
    assert out["num_ala"] == 0
    assert out["is_protein"] is False


# --- Schema / unit-style tests (nucleobases) ---


def test_nucleobase_field_names():
    """Schema field names align with nucleobase codes."""
    names = [f"num_{c.lower()}" for c in NUCLEOBASE_CODES]
    assert "num_ade" in names
    assert "num_thy" in names
    assert "num_ura" in names
    assert len(names) == 5


def test_nucleobase_thymine_free_base_smiles():
    """Free thymine matches one Thy pattern."""
    mol = Chem.MolFromSmiles("Cc1c[nH]c(=O)[nH]c1=O")
    assert mol is not None
    out = count_nucleobases_in_mol(mol)
    assert out["num_thy"] == 1
    assert out["is_nucleobase"] is False


def test_nucleobase_is_nucleobase_when_many_motifs():
    """Several thymines exceed is_nucleobase threshold."""
    _t = "Cc1c[nH]c(=O)[nH]c1=O"
    smi = ".".join([_t] * 4)
    mol = Chem.MolFromSmiles(smi)
    assert mol is not None
    out = count_nucleobases_in_mol(mol)
    assert out["num_thy"] > NUCLEOBASE_MOTIF_THRESHOLD
    assert out["is_nucleobase"] is True


def test_nucleobase_none_mol():
    """None mol yields zero counts and is_nucleobase false."""
    out = count_nucleobases_in_mol(None)
    assert out["num_thy"] == 0
    assert out["is_nucleobase"] is False


def test_pdb_2luf_build_rdkit_mol_uses_relaxed_sanitize():
    """2LUF fixture builds an RDKit mol via relaxed sanitization path."""
    mol = _mol_from_pdb_first_model(PDB_2LUF)
    assert mol is not None


def test_pdb_2luf_substructure_counts_match_seqres():
    """2LUF amino-acid counts match SEQRES for chain A."""
    mol = _mol_from_pdb_first_model(PDB_2LUF)
    assert mol is not None
    out = count_amino_acids_in_mol(mol)
    for code, n_seq in _EXPECTED_2LUF_SEQRES.items():
        key = f"num_{code.lower()}"
        assert out.get(key, 0) == n_seq, (
            f"{key}: substructure count {out.get(key, 0)} does not match SEQRES {n_seq}"
        )
    matched = sum(out[k] for k in out if k.startswith("num_"))
    assert matched == sum(_EXPECTED_2LUF_SEQRES.values())


def test_pdb_3q51_rdkit_mol_nucleobase_counts_match_fixture():
    """3Q51 nucleobase counts match residue composition (OB base+C1' patterns)."""
    mol = _mol_from_pdb_first_model(PDB_3Q51)
    assert mol is not None
    out = count_nucleobases_in_mol(mol)
    for code, n_seq in _EXPECTED_3Q51_SEQRES.items():
        key = f"num_{code.lower()}"
        assert out.get(key, 0) == n_seq, (
            f"{key}: substructure count {out.get(key, 0)} does not match SEQRES {n_seq}"
        )


def test_pdb_3cgy_protonated_builds_mol():
    """Protonated 3CGY fixture yields a non-None RDKit mol."""
    mol = _mol_from_pdb_first_model(PDB_3CGY)
    assert mol is not None


def test_pdb_3cgy_substructure_counts_lte_full_composition():
    """3CGY amino-acid counts match tuned full-composition expectations."""
    mol = _mol_from_pdb_first_model(PDB_3CGY)
    assert mol is not None
    out = count_amino_acids_in_mol(mol)
    for code, n_seq in _EXPECTED_3CGY_COMPOSITION.items():
        key = f"num_{code.lower()}"
        assert out.get(key, 0) == n_seq, (
            f"{key}: substructure count {out.get(key, 0)} does not match SEQRES {n_seq}"
        )


def test_pdb_8c3u_protonated_builds_mol():
    """Protonated 8C3U fixture yields a non-None RDKit mol."""
    mol = _mol_from_pdb_first_model(PDB_8C3U)
    assert mol is not None


def test_pdb_8c3u_substructure_counts_lte_full_composition():
    """8C3U amino-acid counts match tuned full-composition expectations."""
    mol = _mol_from_pdb_first_model(PDB_8C3U)
    assert mol is not None
    out = count_amino_acids_in_mol(mol)
    for code, n_seq in _EXPECTED_8C3U_COMPOSITION.items():
        key = f"num_{code.lower()}"
        assert out.get(key, 0) == n_seq, (
            f"{key}: substructure count {out.get(key, 0)} does not match SEQRES {n_seq}"
        )
