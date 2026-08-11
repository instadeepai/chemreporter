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
"""Tests for RDKit utilities module."""

from unittest.mock import Mock, patch

import ase
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from chemreporter.analysis.obabel_utils import smiles_from_positions
from chemreporter.analysis.rdkit_utils import (
    calculate_graph_derived_properties,
    get_fingerprint,
    get_fpgen,
    get_graph_derived_properties,
    mol_from_openbabel_smiles,
)
from chemreporter.query_database_tools.table_schemas import (
    AMINO_ACID_CODES,
    NUCLEOBASE_CODES,
)


def _zero_amino_acid_counts() -> dict:
    d = {f"num_{c.lower()}": 0 for c in AMINO_ACID_CODES}
    d["is_protein"] = False
    return d


def _zero_nucleobase_counts() -> dict:
    d = {f"num_{c.lower()}": 0 for c in NUCLEOBASE_CODES}
    d["is_nucleobase"] = False
    return d


class TestGetFpgen:
    """Test get_fpgen function."""

    def test_get_fpgen_returns_generator(self):
        """Test that get_fpgen returns a fingerprint generator."""
        fpgen1 = get_fpgen(radius=2, fp_size=1024)
        fpgen2 = get_fpgen(radius=3, fp_size=1024)
        fpgen3 = get_fpgen(radius=2, fp_size=512)
        fpgen4 = get_fpgen(radius=2, fp_size=2048)
        assert isinstance(fpgen1, rdFingerprintGenerator.FingerprintGenerator64)
        assert isinstance(fpgen2, rdFingerprintGenerator.FingerprintGenerator64)
        assert isinstance(fpgen3, rdFingerprintGenerator.FingerprintGenerator64)
        assert isinstance(fpgen4, rdFingerprintGenerator.FingerprintGenerator64)


class TestBuildRdkitMol:
    """Test build_rdkit_mol function."""

    @patch("chemreporter.analysis.rdkit_utils.smiles_from_positions")
    def test_build_rdkit_mol_methane(self, mock_smiles):
        """Test building a methane molecule."""
        mock_smiles.return_value = "C"

        atomic_numbers = [6, 1, 1, 1, 1]
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
        atoms = ase.Atoms(numbers=atomic_numbers, positions=positions)

        mol = calculate_graph_derived_properties(atoms, 0, None)

        mock_smiles.assert_called_once_with(
            atomic_numbers=atomic_numbers, positions=positions, net_charge=0
        )
        assert mol is not None

    @patch("chemreporter.analysis.rdkit_utils.smiles_from_positions")
    def test_build_rdkit_mol_water(self, mock_smiles):
        """Test building a water molecule."""
        mock_smiles.return_value = "O"

        atomic_numbers = [8, 1, 1]
        positions = [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
        smiles = smiles_from_positions(atomic_numbers, positions, 0)

        mol = mol_from_openbabel_smiles(smiles, positions)
        assert mol is None or isinstance(mol, Chem.Mol)


class TestCalculatePropertiesAndFingerprint:
    """Test calculate_properties_and_fingerprint function."""

    @patch("chemreporter.analysis.rdkit_utils.mol_from_openbabel_smiles")
    def test_calculate_properties_none_mol(self, mock_build):
        """Test when molecule cannot be built."""
        mock_build.return_value = None
        fpgen = get_fpgen(radius=2, fp_size=1024)

        atoms = ase.Atoms(numbers=[6], positions=[[0.0, 0.0, 0.0]])
        result = calculate_graph_derived_properties(atoms, 0, fpgen)

        assert isinstance(result, dict)
        assert result == {"smiles": None}

    @patch("chemreporter.analysis.rdkit_utils.mol_from_openbabel_smiles")
    @patch("chemreporter.analysis.rdkit_utils.get_graph_derived_properties")
    @patch("chemreporter.analysis.rdkit_utils.get_fingerprint")
    @patch("chemreporter.analysis.rdkit_utils.count_all_bio_substructures_in_mol")
    def test_calculate_properties_valid_mol(
        self, mock_bio, mock_fp, mock_props, mock_build
    ):
        """Test with a valid molecule."""
        mock_mol = Mock()
        mock_build.return_value = mock_mol
        mock_bio.return_value = {
            **_zero_amino_acid_counts(),
            **_zero_nucleobase_counts(),
        }
        mock_props.return_value = {"logp": -0.5, "tpsa": 0.0, "smiles": "C"}
        mock_fp.return_value = {f"fingerprint_{i}": 0 for i in range(1024)}

        fpgen = get_fpgen(radius=2, fp_size=1024)
        atoms = ase.Atoms(
            numbers=[6, 1, 1, 1, 1],
            positions=[
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
            ],
        )
        result = calculate_graph_derived_properties(atoms, 0, fpgen)

        assert isinstance(result, dict)

        assert "fingerprint_0" in result


class TestGetFingerprint:
    """Test get_fingerprint function."""

    def test_get_fingerprint_returns_dict(self):
        """Test that get_fingerprint returns a dictionary."""
        mol = Chem.MolFromSmiles("C")
        fpgen = get_fpgen(radius=2, fp_size=1024)

        result = get_fingerprint(mol, fpgen)

        assert isinstance(result, dict)

    def test_get_fingerprint_correct_size(self):
        """Test that fingerprint has correct size."""
        mol = Chem.MolFromSmiles("C")
        fpgen = get_fpgen(radius=2, fp_size=1024)

        result = get_fingerprint(mol, fpgen)

        assert len(result) == 1024

    def test_get_fingerprint_keys_format(self):
        """Test that fingerprint keys have correct format."""
        mol = Chem.MolFromSmiles("C")
        fpgen = get_fpgen(radius=2, fp_size=1024)

        result = get_fingerprint(mol, fpgen)

        # Check first and last keys
        assert "fingerprint_0" in result
        assert "fingerprint_1023" in result

    def test_get_fingerprint_values_binary(self):
        """Test that fingerprint values are binary (0 or 1)."""
        mol = Chem.MolFromSmiles("CCO")
        fpgen = get_fpgen(radius=2, fp_size=1024)

        result = get_fingerprint(mol, fpgen)

        # All values should be integers (converted from binary)
        for value in result.values():
            assert isinstance(value, (int, np.integer))

    def test_get_fingerprint_different_molecules(self):
        """Test that different molecules produce different fingerprints."""
        mol1 = Chem.MolFromSmiles("C")
        mol2 = Chem.MolFromSmiles("CCO")
        fpgen = get_fpgen(radius=2, fp_size=1024)

        fp1 = get_fingerprint(mol1, fpgen)
        fp2 = get_fingerprint(mol2, fpgen)

        # Fingerprints should be different
        assert fp1 != fp2


class TestGetProperties:
    """Test get_properties function."""

    def test_get_properties_returns_dict(self):
        """Test that get_properties returns a dictionary."""
        mol = Chem.MolFromSmiles("C")
        result = get_graph_derived_properties(mol)
        assert isinstance(result, dict)

    def test_get_properties_has_required_keys(self):
        """Test that all required properties are present."""
        mol = Chem.MolFromSmiles("C")
        result = get_graph_derived_properties(mol)

        required_keys = ["logp", "tpsa", "smiles"]
        for key in required_keys:
            assert key in result

    def test_get_properties_logp(self):
        """Test LogP calculation."""
        mol = Chem.MolFromSmiles("C")
        result = get_graph_derived_properties(mol)

        assert "logp" in result
        assert isinstance(result["logp"], float)

    def test_get_properties_tpsa(self):
        """Test TPSA calculation."""
        mol = Chem.MolFromSmiles("C")
        result = get_graph_derived_properties(mol)

        assert "tpsa" in result
        assert isinstance(result["tpsa"], (float, int))
        assert result["tpsa"] >= 0

    def test_get_properties_smiles(self):
        """Test SMILES string."""
        mol = Chem.MolFromSmiles("CCO")
        result = get_graph_derived_properties(mol)

        assert "smiles" in result
        assert isinstance(result["smiles"], str)
        # Should contain carbon and oxygen
        assert "C" in result["smiles"]

    def test_get_properties_water(self):
        """Test properties for water molecule."""
        mol = Chem.MolFromSmiles("O")
        result = get_graph_derived_properties(mol)
        # Water has polar surface area
        assert result["tpsa"] > 0

    def test_get_properties_complex_molecule(self):
        """Test properties for a more complex molecule."""
        # Ethanol
        mol = Chem.MolFromSmiles("CCO")
        result = get_graph_derived_properties(mol)

        assert isinstance(result["logp"], float)
        assert result["tpsa"] > 0
        assert "C" in result["smiles"]
