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
"""Tests for fetching from different database types using config files."""

from pathlib import Path

from chemreporter.source_database_tools.fetch import fetch_source_database_readers

# Paths to test config files
TEST_CONFIG_PATH = Path(__file__).parent / "data"


class TestCompareDatabaseTypes:
    """Test comparing different database types."""

    def test_configs_path_exists(self):
        """Test that config directory exists."""
        assert TEST_CONFIG_PATH.exists()
        assert len(list(TEST_CONFIG_PATH.glob("*.yaml"))) > 1

    def test_fetch_source_database_readers(self):
        """Test that different formats are handled correctly."""
        readers = fetch_source_database_readers(dir_path=TEST_CONFIG_PATH)

        readers_file_extensions = [reader.file_extension for reader in readers]
        assert len(readers) == 2
        assert "xyz" in readers_file_extensions
        assert "aselmdb" in readers_file_extensions


class TestIndividualDatabaseTypes:
    """Test fetching from XYZ database."""

    def test_fetch_xyz_database(self):
        """Test fetching database readers ."""
        readers = fetch_source_database_readers(TEST_CONFIG_PATH)

        assert readers is not None
        assert isinstance(readers, list)
        assert len(readers) > 0
        readers_file_extensions = [reader.file_extension for reader in readers]
        # Check first reader
        reader = readers[readers_file_extensions.index("xyz")]
        assert reader.database_name == "small"
        assert reader.split_name == "train"
        assert reader.file_extension == "xyz"

        key = "small_train_xyz-test_0"
        atoms = reader.fetch_atoms_from_key_index(key)

        assert atoms is not None
        assert len(atoms) > 0
        assert hasattr(atoms, "positions")
        assert hasattr(atoms, "numbers")

    def test_fetch_aselmdb_database_with_charges(self):
        """Test fetching database readers from ASE database with charges."""
        readers = fetch_source_database_readers(TEST_CONFIG_PATH)

        assert readers is not None
        assert isinstance(readers, list)
        assert len(readers) > 0
        readers_file_extensions = [reader.file_extension for reader in readers]
        reader = readers[readers_file_extensions.index("aselmdb")]
        assert reader.database_name == "small-charges"
        assert reader.split_name == "train"
        assert reader.file_extension == "aselmdb"

        # Fetch first atom
        key = "small-charges_train_small-test_0"
        atoms = reader.fetch_atoms_from_key_index(key)

        assert atoms is not None
        assert len(atoms) > 0

        # Check for charge metadata in atoms.info
        assert "charge" in atoms.info
        assert "spin" in atoms.info
        assert "mulliken_charges" in atoms.info
        assert "lowdin_charges" in atoms.info

        # Verify charge arrays have correct length
        assert len(atoms.info["mulliken_charges"]) == len(atoms)
        assert len(atoms.info["lowdin_charges"]) == len(atoms)

        # Check for expected metadata fields
        expected_fields = [
            "num_atoms",
            "num_electrons",
            "composition",
            "data_id",
        ]

        for field in expected_fields:
            assert field in atoms.info, f"{field} should be in atoms.info"
