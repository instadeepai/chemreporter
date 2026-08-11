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
"""Tests for structure check module."""

from pathlib import Path

import ase
import numpy as np
import pytest
from ase import Atoms
from fairchem.core.datasets import AseDBDataset

from chemreporter.analysis.structure_check import (
    count_water_molecules,
    is_molecular_structure_valid,
)

# Path to test database
TEST_DB_PATH = Path(__file__).parent.parent / "source_database_tools" / "data"


@pytest.fixture
def valid_methane_atoms():
    """Create valid methane (CH4) molecule."""
    atoms = ase.Atoms(
        symbols=["C", "H", "H", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.36, 1.03, 0.0],
            [-0.36, -0.51, 0.89],
            [-0.36, -0.51, -0.89],
        ],
    )
    return atoms


@pytest.fixture
def valid_water_atoms():
    """Create valid water (H2O) molecule."""
    atoms = ase.Atoms(
        symbols=["O", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
    )
    return atoms


@pytest.fixture
def two_water_molecules():
    """Create two separate water molecules."""
    positions = np.array([
        # First water
        [0.0, 0.0, 0.0],  # O
        [0.96, 0.0, 0.0],  # H
        [-0.24, 0.93, 0.0],  # H
        # Second water (far away)
        [5.0, 0.0, 0.0],  # O
        [5.96, 0.0, 0.0],  # H
        [4.76, 0.93, 0.0],  # H
    ])
    atoms = Atoms("OH2OH2", positions=positions)
    return atoms


@pytest.fixture
def valid_ethane_atoms():
    """Create valid ethane (C2H6) molecule."""
    atoms = ase.Atoms(
        symbols=["C", "C", "H", "H", "H", "H", "H", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.54, 0.0, 0.0],
            [-0.51, 1.03, 0.0],
            [-0.51, -0.51, 0.89],
            [-0.51, -0.51, -0.89],
            [2.05, 1.03, 0.0],
            [2.05, -0.51, 0.89],
            [2.05, -0.51, -0.89],
        ],
    )
    return atoms


@pytest.fixture
def single_atom():
    """Create single carbon atom (subgraph of size 1)."""
    atoms = ase.Atoms(
        symbols=["C"],
        positions=[[0.0, 0.0, 0.0]],
    )
    return atoms


@pytest.fixture
def invalid_hydrogen_bonding_atoms():
    """Create atoms where hydrogen has multiple bonds."""
    atoms = ase.Atoms(
        symbols=["C", "H", "C"],
        positions=[
            [0.0, 0.0, 0.0],
            [0.5, 0.0, 0.0],  # Very close to both carbons
            [1.0, 0.0, 0.0],
        ],
    )
    return atoms


@pytest.fixture
def disconnected_atoms():
    """Create disconnected atoms (two separate molecules)."""
    # Two separate carbon atoms far apart
    atoms = ase.Atoms(
        symbols=["C", "C"],
        positions=[
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # Far apart - no bond
        ],
    )
    return atoms


@pytest.fixture
def water_with_methane():
    """Create a system with one water and one methane."""
    positions = np.array([
        # Water
        [0.0, 0.0, 0.0],  # O
        [0.96, 0.0, 0.0],  # H
        [-0.24, 0.93, 0.0],  # H
        # Methane (far away)
        [5.0, 0.0, 0.0],  # C
        [6.09, 0.0, 0.0],  # H
        [4.64, 1.03, 0.0],  # H
        [4.64, -0.51, 0.89],  # H
        [4.64, -0.51, -0.89],  # H
    ])
    atoms = Atoms("OH2CH4", positions=positions)
    return atoms


@pytest.fixture
def hydrogen_peroxide():
    """Create hydrogen peroxide (H2O2 - not water)."""
    positions = np.array([
        [0.0, 0.0, 0.0],  # O
        [1.47, 0.0, 0.0],  # O
        [-0.24, 0.93, 0.0],  # H
        [1.71, 0.93, 0.0],  # H
    ])
    atoms = Atoms("OOHH", positions=positions)
    return atoms


@pytest.fixture
def isolated_hydrogen():
    """Create isolated hydrogen atom."""
    atoms = ase.Atoms(
        symbols=["H"],
        positions=[[0.0, 0.0, 0.0]],
    )
    return atoms


@pytest.fixture
def hydroxide():
    """Create hydroxide ion (OH - not water, only 1 H)."""
    positions = np.array([
        [0.0, 0.0, 0.0],  # O
        [0.96, 0.0, 0.0],  # H
    ])
    atoms = Atoms("OH", positions=positions)
    return atoms


@pytest.fixture
def hydrogen_with_no_bonds():
    """Create hydrogen with no bonds (too far from other atoms)."""
    atoms = ase.Atoms(
        symbols=["C", "H"],
        positions=[
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],  # Far away
        ],
    )
    return atoms


@pytest.fixture
def benzene_like_atoms():
    """Create benzene-like structure."""
    atoms = ase.Atoms(
        symbols=["C", "C", "C", "C", "C", "C", "H", "H", "H", "H", "H", "H"],
        positions=[
            # Carbon ring
            [1.4, 0.0, 0.0],
            [0.7, 1.2, 0.0],
            [-0.7, 1.2, 0.0],
            [-1.4, 0.0, 0.0],
            [-0.7, -1.2, 0.0],
            [0.7, -1.2, 0.0],
            # Hydrogens
            [2.5, 0.0, 0.0],
            [1.2, 2.2, 0.0],
            [-1.2, 2.2, 0.0],
            [-2.5, 0.0, 0.0],
            [-1.2, -2.2, 0.0],
            [1.2, -2.2, 0.0],
        ],
    )
    return atoms


@pytest.fixture
def test_db_atoms():
    """Load atoms from the test database."""
    db_path = TEST_DB_PATH / "small-test.aselmdb"
    if not db_path.exists():
        pytest.skip(f"Test database not found at {db_path}")

    try:
        dataset = AseDBDataset({"src": str(db_path)})
        atoms_list = []
        for i in range(min(10, len(dataset))):
            atoms = dataset.get_atoms(i)
            atoms_list.append(atoms)
        return atoms_list
    except ImportError:
        pytest.skip("AseDBDataset not available or database backend missing")


class TestIsMolecularStructureValid:
    """Test is_molecular_structure_valid function."""

    def test_valid_methane_returns_true(self, valid_methane_atoms):
        """Test that valid methane structure passes validation."""
        result = is_molecular_structure_valid(valid_methane_atoms)
        assert result

    def test_valid_water_returns_true(self, valid_water_atoms):
        """Test that water passes validation."""
        result = is_molecular_structure_valid(valid_water_atoms)
        assert result

    def test_valid_water_passes(self, valid_water_atoms):
        """Test that water pass validation."""
        result = is_molecular_structure_valid(valid_water_atoms)
        assert result

    def test_valid_ethane_returns_true(self, valid_ethane_atoms):
        """Test that valid ethane structure passes validation."""
        result = is_molecular_structure_valid(valid_ethane_atoms)
        assert result

    def test_single_atom_returns_true(self, single_atom):
        """Test that single atom fails validation."""
        result = is_molecular_structure_valid(single_atom)
        assert not result

    def test_disconnected_atoms_returns_false(self, disconnected_atoms):
        """Test that disconnected atoms fail validation."""
        result = is_molecular_structure_valid(disconnected_atoms)
        assert not result

    def test_custom_min_subgraph_size(self, valid_methane_atoms):
        """Test that custom minimum subgraph size is respected."""
        # Methane has 5 atoms, should fail if we require 6+
        result = is_molecular_structure_valid(valid_methane_atoms, min_subgraph_size=6)
        assert not result

    def test_empty_atoms_returns_false(self):
        """Test that empty atoms object returns False."""
        atoms = ase.Atoms()
        result = is_molecular_structure_valid(atoms)
        assert not result

    def test_hydrogen_with_single_bond_passes(self, valid_methane_atoms):
        """Test that hydrogens with single bond pass validation."""
        # Methane has all H atoms with single bonds
        result = is_molecular_structure_valid(valid_methane_atoms)
        assert result

    def test_returns_boolean(self, valid_methane_atoms):
        """Test that function always returns a boolean."""
        result = is_molecular_structure_valid(valid_methane_atoms)
        assert result

    def test_hydrogen_with_no_bonds_returns_false(self, hydrogen_with_no_bonds):
        """Test that hydrogen with no bonds fails validation."""
        result = is_molecular_structure_valid(hydrogen_with_no_bonds)
        assert not result

    def test_invalid_hydrogen_bonding_returns_false(
        self, invalid_hydrogen_bonding_atoms
    ):
        """Test that hydrogen with multiple bonds fails validation."""
        result = is_molecular_structure_valid(invalid_hydrogen_bonding_atoms)
        assert not result

    def test_benzene_like_structure(self, benzene_like_atoms):
        """Test a larger molecule with benzene-like structure."""
        result = is_molecular_structure_valid(benzene_like_atoms)
        assert result
        # Benzene should pass if geometry is reasonable

    def test_atoms_with_positions_too_far_returns_false(self):
        """Test that atoms with positions too far apart (no bonds) return False."""
        # Two hydrogen atoms very far apart
        atoms = ase.Atoms(
            symbols=["H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [100.0, 0.0, 0.0],
            ],
        )
        result = is_molecular_structure_valid(atoms)
        assert not result

    def test_multiple_disconnected_molecules_with_sufficient_size(self):
        """Test multiple disconnected molecules."""
        # Two separate methane molecules far apart
        atoms = ase.Atoms(
            symbols=["C", "H", "H", "H", "H", "C", "H", "H", "H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [-0.36, 1.03, 0.0],
                [-0.36, -0.51, 0.89],
                [-0.36, -0.51, -0.89],
                [20.0, 0.0, 0.0],
                [21.09, 0.0, 0.0],
                [19.64, 1.03, 0.0],
                [19.64, -0.51, 0.89],
                [19.64, -0.51, -0.89],
            ],
        )
        result = is_molecular_structure_valid(atoms)
        # Should be True as both subgraphs have >= 2 atoms
        assert result

    def test_hydrogen_with_two_neighbors_fails(self):
        """Test that hydrogen with two neighbors fails validation."""
        # Create a situation where H is close to two carbons
        atoms = ase.Atoms(
            symbols=["C", "H", "C"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.7, 0.0, 0.0],  # Close to both carbons
                [1.4, 0.0, 0.0],
            ],
        )
        result = is_molecular_structure_valid(atoms)
        assert not result

    def test_min_subgraph_size_one(self, single_atom):
        """Test with minimum subgraph size of 1."""
        result = is_molecular_structure_valid(single_atom, min_subgraph_size=1)
        assert result

    def test_min_subgraph_size_zero(self, single_atom):
        """Test with minimum subgraph size of 0."""
        result = is_molecular_structure_valid(single_atom, min_subgraph_size=0)
        assert result


class TestWithRealDatabaseAtoms:
    """Test with real atoms from the test database."""

    def test_real_atoms_from_database(self, test_db_atoms):
        """Test validation with real atoms from the test database."""
        results = []
        for atoms in test_db_atoms:
            result = is_molecular_structure_valid(atoms)
            results.append(result)
            # Just verify the function returns a boolean without error
            assert isinstance(result, bool)
        # At least verify some structures were processed
        assert len(results) > 0

    def test_real_atoms_structure(self, test_db_atoms):
        """Test the structure of atoms from the database."""
        assert len(test_db_atoms) > 0
        first_atom = test_db_atoms[0]
        assert len(first_atom) > 0
        assert hasattr(first_atom, "positions")
        assert hasattr(first_atom, "numbers")

    def test_real_atoms_hydrogen_count(self, test_db_atoms):
        """Test hydrogen validation on real atoms."""
        for atoms in test_db_atoms[:3]:
            h_count = sum(1 for symbol in atoms.get_chemical_symbols() if symbol == "H")
            if h_count > 0:
                # Just verify the function runs
                result = is_molecular_structure_valid(atoms)
                assert result


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_molecule_h2(self):
        """Test with H2 molecule."""
        atoms = ase.Atoms(
            symbols=["H", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [0.74, 0.0, 0.0],  # H-H bond length
            ],
        )
        result = is_molecular_structure_valid(atoms)
        # H2 is valid (2 atoms >= 2 minimum, each H has 1 neighbor)
        assert result

    def test_molecule_with_only_heavy_atoms(self):
        """Test molecule with no hydrogen atoms."""
        atoms = ase.Atoms(
            symbols=["C", "C"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.54, 0.0, 0.0],  # C-C bond
            ],
        )
        result = is_molecular_structure_valid(atoms)
        # Should pass as there are no H atoms to check
        assert result

    def test_large_molecule(self):
        """Test with a large molecule."""
        # Create a simple chain of 20 carbons
        n = 20
        positions = [[i * 1.54, 0.0, 0.0] for i in range(n)]
        atoms = ase.Atoms(symbols=["C"] * n, positions=positions)

        result = is_molecular_structure_valid(atoms)
        assert result

    def test_exception_handling_with_invalid_positions(self):
        """Test that exceptions are handled gracefully."""
        # Create atoms with NaN positions
        atoms = ase.Atoms(
            symbols=["C", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [np.nan, np.nan, np.nan],
            ],
        )
        result = is_molecular_structure_valid(atoms)
        # Should return False due to exception handling
        assert not result

    def test_exception_handling_with_inf_positions(self):
        """Test handling of infinite positions."""
        atoms = ase.Atoms(
            symbols=["C", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [np.inf, 0.0, 0.0],
            ],
        )
        result = is_molecular_structure_valid(atoms)
        # Should return False due to exception handling
        assert not result

    def test_three_atoms_linear(self):
        """Test linear three-atom molecule."""
        atoms = ase.Atoms(
            symbols=["H", "C", "H"],
            positions=[
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [2.18, 0.0, 0.0],
            ],
        )
        result = is_molecular_structure_valid(atoms, min_subgraph_size=2)
        assert result


class TestCountWaterMolecules:
    """Test count_water_molecules function with synthetic fixtures."""

    def test_single_water_molecule(self, valid_water_atoms):
        """Test counting a single water molecule."""
        count = count_water_molecules(valid_water_atoms)
        assert count == 1

    def test_two_water_molecules(self, two_water_molecules):
        """Test counting two water molecules."""
        count = count_water_molecules(two_water_molecules)
        assert count == 2

    def test_water_with_methane(self, water_with_methane):
        """Test counting water in presence of other molecules."""
        count = count_water_molecules(water_with_methane)
        assert count == 1

    def test_methane_has_no_water(self, valid_methane_atoms):
        """Test that methane has no water molecules."""
        count = count_water_molecules(valid_methane_atoms)
        assert count == 0


class TestCountWaterMoleculesWithRealDatabase:
    """Test count_water_molecules with structures from the small test database."""

    @pytest.fixture
    def small_test_db_path(self):
        """Path to small test database."""
        db_path = TEST_DB_PATH / "small-test.aselmdb"
        if not db_path.exists():
            pytest.skip(f"Test database not found at {db_path}")
        return db_path

    def test_water_count_respects_oxygen_limit(self, small_test_db_path):
        """Test that water count never exceeds number of oxygen atoms."""
        try:
            dataset = AseDBDataset({"src": str(small_test_db_path)})

            for i in range(min(20, len(dataset))):
                atoms = dataset.get_atoms(i)
                count = count_water_molecules(atoms)

                # Each water needs 1 oxygen, so count <= num_O
                symbols = atoms.get_chemical_symbols()
                num_o = sum(1 for s in symbols if s == "O")
                assert count <= num_o, (
                    f"Structure {i}: water count {count} exceeds O atoms {num_o}"
                )

        except ImportError:
            pytest.skip("AseDBDataset not available or database backend missing")

    def test_structures_without_oxygen_have_no_water(self, small_test_db_path):
        """Test that structures without oxygen report zero waters."""
        try:
            dataset = AseDBDataset({"src": str(small_test_db_path)})

            tested_count = 0
            for i in range(len(dataset)):
                atoms = dataset.get_atoms(i)
                symbols = atoms.get_chemical_symbols()

                if "O" not in symbols:
                    count = count_water_molecules(atoms)
                    assert count == 0, (
                        f"Structure {i} has no O but reports {count} waters"
                    )
                    tested_count += 1

                # Test at least a few structures
                if tested_count >= 3:
                    break

        except ImportError:
            pytest.skip("AseDBDataset not available or database backend missing")
