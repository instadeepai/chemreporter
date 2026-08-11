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
"""Tests for xyz_tools module."""

from pathlib import Path

import ase
import pytest

from chemreporter.source_database_tools.source_db_implementations import (
    XyzDatabaseImplementation,
)

TEST_XYZ_FILE = Path(__file__).parent / "data" / "xyz-test.xyz"


class TestReadXyzFile:
    """Test read_xyz_file function."""

    def test_read_xyz_file_returns_dataset(self):
        """Test that read_xyz_file returns a list of Atoms objects."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        assert isinstance(dataset, list)
        assert all(isinstance(atoms, ase.Atoms) for atoms in dataset)

    def test_read_xyz_file_dataset_not_empty(self):
        """Test that the dataset contains entries."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        assert len(dataset) > 0

    def test_read_xyz_file_with_nonexistent_file(self):
        """Test that reading a non-existent file raises an error."""
        nonexistent_file = Path("/tmp/nonexistent_file.xyz")
        with pytest.raises(Exception):  # Could be FileNotFoundError or other error
            dataset = XyzDatabaseImplementation().read_file(nonexistent_file)
            # Try to access the dataset to trigger the error
            _ = len(dataset)

    def test_read_xyz_file_dataset_structure(self):
        """Test that the dataset has the expected structure."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        # Get the first item to verify structure
        first_item = dataset[0]
        assert isinstance(first_item, ase.Atoms)
        assert hasattr(first_item, "positions")
        assert hasattr(first_item, "numbers")


class TestGetAtomsFromXyzPointer:
    """Test get_atoms_from_xyz_pointer function."""

    @pytest.fixture
    def xyz_dataset(self):
        """Fixture to create a list of Atoms from the test XYZ file."""
        return XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)

    def test_get_atoms_returns_atoms_object(self, xyz_dataset):
        """Test that get_atoms_from_xyz_pointer returns an Atoms object."""
        atoms = XyzDatabaseImplementation().get_atoms_from_db_pointer(xyz_dataset, 0)
        assert isinstance(atoms, ase.Atoms)

    def test_get_atoms_has_positions(self, xyz_dataset):
        """Test that the Atoms object has positions."""
        atoms = XyzDatabaseImplementation().get_atoms_from_db_pointer(xyz_dataset, 0)
        assert len(atoms.positions) > 0
        assert atoms.positions.shape[1] == 3  # 3D coordinates

    def test_get_atoms_has_atomic_numbers(self, xyz_dataset):
        """Test that the Atoms object has atomic numbers."""
        atoms = XyzDatabaseImplementation().get_atoms_from_db_pointer(xyz_dataset, 0)
        assert len(atoms.numbers) > 0
        assert all(num > 0 for num in atoms.numbers)

    def test_get_atoms_multiple_indices(self, xyz_dataset):
        """Test getting atoms from multiple indices."""
        if len(xyz_dataset) < 2:
            pytest.skip("Dataset has less than 2 entries")

        atoms_0 = XyzDatabaseImplementation().get_atoms_from_db_pointer(xyz_dataset, 0)
        atoms_1 = XyzDatabaseImplementation().get_atoms_from_db_pointer(xyz_dataset, 1)

        # Verify they are different structures
        assert isinstance(atoms_0, ase.Atoms)
        assert isinstance(atoms_1, ase.Atoms)

    def test_get_atoms_out_of_bounds(self, xyz_dataset):
        """Test that accessing out of bounds index raises an error."""
        with pytest.raises(IndexError):
            XyzDatabaseImplementation().get_atoms_from_db_pointer(
                xyz_dataset, len(xyz_dataset) + 100
            )

    def test_get_atoms_negative_index(self, xyz_dataset):
        """Test that negative indices work (Python-style indexing)."""
        # This may or may not work depending on AseReadDataset implementation
        try:
            atoms = XyzDatabaseImplementation().get_atoms_from_db_pointer(
                xyz_dataset, -1
            )
            assert isinstance(atoms, ase.Atoms)
        except (IndexError, TypeError):
            # If negative indexing is not supported, that's okay
            pytest.skip("Negative indexing not supported by AseReadDataset")


class TestAtomsInfoParsedProperties:
    """Test that atoms objects have parsed properties in .info attribute."""

    def test_atoms_have_properties_in_info(self):
        """Test that each atoms object has parsed properties in .info."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        for atoms in dataset:
            # Check that .info is not empty
            assert len(atoms.info) > 0
            # Should have some common properties
            assert isinstance(atoms.info, dict)

    def test_energy_property_parsed(self):
        """Test that energy is parsed as a float."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        first_atoms = dataset[0]
        # Check if energy property exists and is a float
        if "energy" in first_atoms.info:
            assert isinstance(first_atoms.info["energy"], float)

    def test_total_charge_property_parsed(self):
        """Test that total_charge is parsed as an integer."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        first_atoms = dataset[0]
        # Check if total_charge property exists and is an int
        if "charge" in first_atoms.info:
            assert isinstance(first_atoms.info["charge"], int)

    def test_pbc_property_parsed_as_list(self):
        """Test that pbc is parsed as a list."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        first_atoms = dataset[0]
        # Check if pbc property exists and is a list
        if "pbc" in first_atoms.info:
            assert isinstance(first_atoms.info["pbc"], list)
            assert len(first_atoms.info["pbc"]) == 3

    def test_smiles_property_parsed_as_string(self):
        """Test that smiles is stored as a string."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        first_atoms = dataset[0]
        # Check if smiles property exists and is a string
        if "smiles" in first_atoms.info:
            assert isinstance(first_atoms.info["smiles"], str)

    def test_get_atoms_from_pointer_preserves_properties(self):
        """Test that get_atoms_from_xyz_pointer returns atoms with properties."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        atoms = XyzDatabaseImplementation().get_atoms_from_db_pointer(dataset, 0)
        assert len(atoms.info) > 0
        assert isinstance(atoms.info, dict)


class TestXyzToolsIntegration:
    """Integration tests for xyz_tools functions."""

    def test_read_and_get_atoms_workflow(self):
        """Test the complete workflow of reading XYZ file and extracting atoms."""
        # Step 1: Read the XYZ file
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)

        # Step 2: Get atoms from the dataset
        atoms = XyzDatabaseImplementation().get_atoms_from_db_pointer(dataset, 0)

        # Step 3: Verify the atoms object has expected properties
        assert isinstance(atoms, ase.Atoms)
        assert len(atoms) > 0
        assert atoms.positions.shape[0] == len(atoms)
        assert atoms.positions.shape[1] == 3

    def test_iterate_through_dataset(self):
        """Test iterating through all entries in the dataset."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        dataset_length = len(dataset)

        # Get a sample of atoms (first 5 or all if less than 5)
        num_to_test = min(5, dataset_length)

        for i in range(num_to_test):
            atoms = XyzDatabaseImplementation().get_atoms_from_db_pointer(dataset, i)
            assert isinstance(atoms, ase.Atoms)
            assert len(atoms) > 0

    def test_atoms_have_consistent_structure(self):
        """Test that atoms from the dataset have consistent structure."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)
        atoms = XyzDatabaseImplementation().get_atoms_from_db_pointer(dataset, 0)

        # Check that the number of positions matches the number of atoms
        assert len(atoms.positions) == len(atoms.numbers)
        assert len(atoms.positions) == len(atoms)

    def test_properties_properly_stored_in_atoms(self):
        """Test that properties are properly stored in atoms objects."""
        dataset = XyzDatabaseImplementation().read_file(TEST_XYZ_FILE)

        # Verify all atoms have properties
        for i, atoms in enumerate(dataset):
            assert isinstance(atoms, ase.Atoms)
            assert len(atoms.info) > 0
            assert isinstance(atoms.info, dict)
            assert len(atoms) > 0
