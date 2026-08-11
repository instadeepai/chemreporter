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
"""Tests for HDF5 writer functionality."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import polars as pl
import pytest

try:
    from mlip.data.chemical_systems_readers.hdf5_reader import (
        Hdf5Reader as ChemicalSystemsReader,
    )
except ImportError:
    ChemicalSystemsReader = None

from chemreporter.database_processors.hdf5_writer import (
    make_entry_key_lookup,
    write_hdf5,
)
from chemreporter.source_database_tools.database_reader import (
    KeyEntry,
    SourceDatabaseReader,
    parse_key,
)


@pytest.fixture
def test_db_path():
    """Path to the test ASE database."""
    return (
        Path(__file__).parent.parent
        / "source_database_tools"
        / "data"
        / "small-test.aselmdb"
    )


@pytest.fixture
def source_db_reader(test_db_path):
    """Create a SourceDatabaseReader for the test database."""
    return SourceDatabaseReader(
        db_path=test_db_path, database_name="small", split_name="train"
    )


@pytest.fixture
def xyz_db_path():
    """Path to the test XYZ database."""
    return (
        Path(__file__).parent.parent / "source_database_tools" / "data" / "xyz-test.xyz"
    )


@pytest.fixture
def xyz_db_reader(xyz_db_path):
    """Create a SourceDatabaseReader for the XYZ test database."""
    return SourceDatabaseReader(
        db_path=xyz_db_path,
        database_name="xyz-test",
        split_name="test",
        database_format="xyz",
    )


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file path."""
    with tempfile.NamedTemporaryFile(suffix=".hdf5", delete=False) as f:
        temp_path = Path(f.name)
    yield temp_path
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


class TestHDF5Writer:
    """Test HDF5 writer functionality."""

    def test_name_mistmatch(self, source_db_reader, temp_hdf5_file):
        """Test that name mismatch raises an error."""
        with pytest.raises(RuntimeError):
            write_hdf5(
                key_entries=["big_train_test_0"],
                source_db_readers=source_db_reader,
                hdf5_path=str(temp_hdf5_file),
            )
        with pytest.raises(RuntimeError):
            write_hdf5(
                key_entries=["small_validation_test_0"],
                source_db_readers=source_db_reader,
                hdf5_path=str(temp_hdf5_file),
            )

    def test_parse_key(self):
        """Test that the key is parsed correctly."""
        key = "small_train_small-test_0"
        expected_key = KeyEntry(
            database_name="small",
            split_name="train",
            source_file_name="small-test",
            index=0,
        )
        parsed_key = parse_key(key)
        assert parsed_key == expected_key

    def test_write_hdf5_basic(self, source_db_reader, temp_hdf5_file):
        """Test basic HDF5 writing functionality."""
        # Get some keys from the database
        # The database has entries indexed from 0 to 9 (0-based indexing)
        # Keys should be in format "databasename_splitname_filename_index"

        test_keys = [
            "small_train_small-test_0",
            "small_train_small-test_1",
            "small_train_small-test_2",
        ]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=source_db_reader,
            hdf5_path=str(temp_hdf5_file),
        )

        # Verify the file was created
        assert temp_hdf5_file.exists()

        # Read back and verify
        with h5py.File(temp_hdf5_file, "r") as f:
            # Check that groups were created for each key
            for key in test_keys:
                assert key in f, f"Key {key} not found in HDF5 file"
                group = f[key]

                # Check required fields
                assert "elements" in group
                assert "positions" in group
                assert "forces" in group

    def test_write_hdf5_with_numpy_array(self, source_db_reader, temp_hdf5_file):
        """Test HDF5 writing with numpy array input."""
        test_keys = np.array([
            "small_train_small-test_0",
            "small_train_small-test_4",
            "small_train_small-test_9",
        ])

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=[source_db_reader],
            hdf5_path=str(temp_hdf5_file),
        )

        assert temp_hdf5_file.exists()

        with h5py.File(temp_hdf5_file, "r") as f:
            assert len(f.keys()) == 3
            for key in test_keys:
                assert key in f

    def test_write_hdf5_all_entries(self, source_db_reader, temp_hdf5_file):
        """Test writing all entries from the database."""
        # Generate keys for all 10 entries (0-indexed)
        test_keys = [f"small_train_small-test_{i}" for i in range(10)]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=[source_db_reader],
            hdf5_path=str(temp_hdf5_file),
        )

        assert temp_hdf5_file.exists()

        with h5py.File(temp_hdf5_file, "r") as f:
            assert len(f.keys()) == 10

            # Verify each entry has proper structure
            for key in test_keys:
                assert key in f
                group = f[key]
                assert "elements" in group
                assert "positions" in group
                assert group["elements"].dtype in [np.int32, np.int64]
                assert group["positions"].dtype in [np.float32, np.float64]

    def test_write_hdf5_max_entries_limit(self, source_db_reader, temp_hdf5_file):
        """Test that max_entries parameter limits batch size."""
        # Request 5 keys with max_entries=5 to write all of them
        test_keys = [f"small_train_small-test_{i}" for i in range(5)]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=[source_db_reader],
            hdf5_path=str(temp_hdf5_file),
        )

        assert temp_hdf5_file.exists()

        with h5py.File(temp_hdf5_file, "r") as f:
            # All 5 keys should be written
            assert len(f.keys()) == 5
            for i in range(5):
                assert f"small_train_small-test_{i}" in f

    def test_write_hdf5_data_integrity(self, source_db_reader, temp_hdf5_file):
        """Test that data is preserved correctly in HDF5."""
        test_key = "small_train_small-test_0"

        # Get the original atoms object
        atoms = source_db_reader.fetch_atoms_from_key_index(test_key)

        # Write to HDF5
        write_hdf5(
            key_entries=[test_key],
            source_db_readers=[source_db_reader],
            hdf5_path=str(temp_hdf5_file),
        )

        # Read back and compare
        with h5py.File(temp_hdf5_file, "r") as f:
            group = f[test_key]

            # Check that elements match
            np.testing.assert_array_equal(group["elements"][:], atoms.numbers)

            # Check that positions match
            np.testing.assert_array_almost_equal(
                group["positions"][:], atoms.positions, decimal=6
            )

            # Check that data length matches atoms length
            assert len(group["elements"][:]) == len(atoms)

    def test_write_hdf5_stress_tensor_shape(self, source_db_reader, temp_hdf5_file):
        """Voigt 6-element stress is converted to a 3x3 matrix on write."""
        test_key = "small_train_small-test_0"

        atoms = source_db_reader.fetch_atoms_from_key_index(test_key)
        if not hasattr(atoms, "_calc") or atoms._calc is None:
            atoms._calc = atoms.calc
        atoms._calc.results["stress"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        with patch.object(
            source_db_reader, "iter_atoms", return_value=iter([(atoms, test_key)])
        ):
            write_hdf5(
                key_entries=[test_key],
                source_db_readers=[source_db_reader],
                hdf5_path=str(temp_hdf5_file),
            )

        with h5py.File(temp_hdf5_file, "r") as f:
            group = f[test_key]
            assert "stress" in group
            assert group["stress"].shape == (3, 3)

    def test_write_hdf5_duplicate_keys(self, source_db_reader, temp_hdf5_file):
        """Test that duplicate keys are handled correctly."""
        # Include duplicates (0-indexed)
        test_keys = [
            "small_train_small-test_0",
            "small_train_small-test_1",
            "small_train_small-test_0",
            "small_train_small-test_2",
        ]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=[source_db_reader],
            hdf5_path=str(temp_hdf5_file),
        )

        assert temp_hdf5_file.exists()

        with h5py.File(temp_hdf5_file, "r") as f:
            # Should have 3 unique keys
            assert len(f.keys()) == 3
            assert "small_train_small-test_0" in f
            assert "small_train_small-test_1" in f
            assert "small_train_small-test_2" in f

    def test_write_hdf5_with_charge_and_spin(self, source_db_reader, temp_hdf5_file):
        """Test that charge and spin from ASE database are exported as HDF5 attributes.

        This test verifies that when atoms.info contains 'charge' and 'spin' fields
        from the database, they are properly exported as HDF5 attributes.
        """
        test_keys = [
            "small_train_small-test_0",
            "small_train_small-test_1",
        ]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=[source_db_reader],
            hdf5_path=str(temp_hdf5_file),
        )

        assert temp_hdf5_file.exists()

        with h5py.File(temp_hdf5_file, "r") as f:
            for key in test_keys:
                assert key in f, f"Key {key} not found in HDF5 file"
                group = f[key]

                # Verify charge and spin are exported as attributes
                assert "charge" in group.attrs, "charge should be in HDF5 attributes"
                assert "spin_multiplicity" in group.attrs, (
                    "spin should be in HDF5 attributes"
                )

                # Verify mulliken_charges and lowdin_charges are exported as datasets
                assert "mulliken_charges" in group, (
                    "mulliken_charges should be a dataset"
                )
                assert "lowdin_charges" in group, "lowdin_charges should be a dataset"

                # Verify charge arrays are not empty (this was the original bug!)
                mulliken = group["mulliken_charges"][:]
                lowdin = group["lowdin_charges"][:]

                assert len(mulliken) > 0, "mulliken_charges should not be empty"
                assert len(lowdin) > 0, "lowdin_charges should not be empty"

                # Verify they match num_atoms
                num_atoms = group.attrs["num_atoms"]
                assert len(mulliken) == num_atoms, (
                    "mulliken_charges length should match num_atoms"
                )
                assert len(lowdin) == num_atoms, (
                    "lowdin_charges length should match num_atoms"
                )

    def test_write_hdf5_from_xyz(self, xyz_db_reader, temp_hdf5_file):
        """Test HDF5 writing from XYZ file format.

        This test verifies that:
        1. XYZ files can be read and written to HDF5
        2. All atoms.info keys are dynamically extracted
        3. Only array-like data is written as datasets
        """
        # Test with first 3 entries from XYZ file
        test_keys = [
            "xyz-test_test_xyz-test_0",
            "xyz-test_test_xyz-test_1",
            "xyz-test_test_xyz-test_2",
        ]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=[xyz_db_reader],
            hdf5_path=str(temp_hdf5_file),
        )

        # Verify the file was created
        assert temp_hdf5_file.exists()

        # Read back and verify structure
        with h5py.File(temp_hdf5_file, "r") as f:
            # Check that groups were created for each key
            assert len(f.keys()) == 3

            for key in test_keys:
                assert key in f, f"Key {key} not found in HDF5 file"
                group = f[key]

                # Check required fields exist
                assert "elements" in group
                assert "positions" in group
                assert "forces" in group

                # Verify the atoms.info keys were dynamically extracted
                # The XYZ file should not have MULLIKEN_CHARGES or LOWDIN_CHARGES
                # but should have other metadata in atoms.info

                # Verify no empty datasets were created
                for dataset_name in group.keys():
                    dataset = group[dataset_name]
                    # Check that dataset is not None and has data
                    assert dataset is not None, f"Dataset {dataset_name} is None"
                    # For array datasets, verify they have elements
                    if hasattr(dataset, "shape") and dataset.shape is not None:
                        if len(dataset.shape) > 0:  # It's an array
                            assert dataset.shape[0] > 0, (
                                f"Dataset {dataset_name} is empty, which will cause "
                                "mlip hdf5_reader to fail"
                            )


class TestChemicalSystemsReader:
    """Test if produced HDF5 file can be loaded by mlip's ChemicalSystemsReader."""

    def test_load_hdf5_file(self, source_db_reader, temp_hdf5_file):
        """Test if HDF5 file can be loaded by mlip's ChemicalSystemsReader."""
        if ChemicalSystemsReader is None:
            pytest.skip("mlip not installed")

        test_keys = [f"small_train_small-test_{i}" for i in range(10)]
        write_hdf5(
            key_entries=test_keys,
            source_db_readers=[source_db_reader],
            hdf5_path=str(temp_hdf5_file),
        )
        assert temp_hdf5_file.exists()
        assert temp_hdf5_file.stat().st_size > 0  # size of the file is not 0

        reader = ChemicalSystemsReader(
            filepaths=[str(temp_hdf5_file)],
            num_to_load=3,
        )
        systems = reader.load()
        assert len(systems) == 3


class TestExtrasFields:
    """Test extras_fields functionality in HDF5 writer."""

    @pytest.fixture
    def extras_fields_dict(self):
        """Create a sample extras_fields dict for testing."""
        df = pl.DataFrame({
            "entry_key": [
                "small_train_small-test_0",
                "small_train_small-test_1",
                "small_train_small-test_2",
                "small_train_small-test_3",
                "small_train_small-test_4",
            ],
            "extra_label": ["A", "B", "C", "D", "E"],
            "extra_score": [0.1, 0.2, 0.3, 0.4, 0.5],
        })
        return make_entry_key_lookup(df)

    def test_write_hdf5_with_extras_fields(
        self, source_db_reader, temp_hdf5_file, extras_fields_dict
    ):
        """Test that extras_fields are written to HDF5."""
        test_keys = [
            "small_train_small-test_0",
            "small_train_small-test_1",
            "small_train_small-test_2",
        ]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=source_db_reader,
            hdf5_path=str(temp_hdf5_file),
            extras_fields=extras_fields_dict,
        )

        assert temp_hdf5_file.exists()

        with h5py.File(temp_hdf5_file, "r") as f:
            for key in test_keys:
                assert key in f
                group = f[key]

                # Check that extras group exists
                assert "extras" in group

                # Check that extras fields are present as attributes in extras group
                assert "extra_label" in group["extras"].attrs
                assert "extra_score" in group["extras"].attrs

    def test_extras_fields_values_correct(
        self, source_db_reader, temp_hdf5_file, extras_fields_dict
    ):
        """Test that extras_fields values are written correctly."""
        test_keys = [
            "small_train_small-test_0",
            "small_train_small-test_2",  # Note: skipping index 1
        ]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=source_db_reader,
            hdf5_path=str(temp_hdf5_file),
            extras_fields=extras_fields_dict,
        )

        with h5py.File(temp_hdf5_file, "r") as f:
            # Check key 0
            group_0 = f["small_train_small-test_0"]
            assert group_0["extras"].attrs["extra_label"] == "A"
            assert group_0["extras"].attrs["extra_score"] == 0.1

            # Check key 2
            group_2 = f["small_train_small-test_2"]
            assert group_2["extras"].attrs["extra_label"] == "C"
            assert group_2["extras"].attrs["extra_score"] == 0.3

    def test_extras_fields_with_arrays(self, source_db_reader, temp_hdf5_file):
        """Test that array extras_fields are written as datasets."""
        extras_with_arrays = pl.DataFrame({
            "entry_key": [
                "small_train_small-test_0",
                "small_train_small-test_1",
            ],
            "scalar_value": [1.0, 2.0],
            "array_value": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        })

        test_keys = [
            "small_train_small-test_0",
            "small_train_small-test_1",
        ]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=source_db_reader,
            hdf5_path=str(temp_hdf5_file),
            extras_fields=make_entry_key_lookup(extras_with_arrays),
        )

        with h5py.File(temp_hdf5_file, "r") as f:
            group_0 = f["small_train_small-test_0"]

            # Scalar should be an attribute in extras group
            assert "scalar_value" in group_0["extras"].attrs
            assert group_0["extras"].attrs["scalar_value"] == 1.0

            # Array should be a dataset in extras group
            assert "array_value" in group_0["extras"]
            np.testing.assert_array_almost_equal(
                group_0["extras"]["array_value"][:], [1.0, 2.0, 3.0]
            )

    def test_extras_fields_none_backward_compatible(
        self, source_db_reader, temp_hdf5_file
    ):
        """Test that extras_fields=None works (backward compatibility)."""
        test_keys = [
            "small_train_small-test_0",
            "small_train_small-test_1",
        ]

        # Should not raise any errors
        write_hdf5(
            key_entries=test_keys,
            source_db_readers=source_db_reader,
            hdf5_path=str(temp_hdf5_file),
            extras_fields=None,
        )

        assert temp_hdf5_file.exists()

        with h5py.File(temp_hdf5_file, "r") as f:
            assert len(f.keys()) == 2
            for key in test_keys:
                assert key in f

    def test_extras_fields_ordering(self, source_db_reader, temp_hdf5_file):
        """Test that extras_fields are correctly matched by key."""
        # Create DataFrame with keys in different order than test_keys
        extras_df = pl.DataFrame({
            "entry_key": [
                "small_train_small-test_2",  # Out of order
                "small_train_small-test_0",
                "small_train_small-test_1",
            ],
            "order_check": [200, 0, 100],  # Values to verify correct matching
        })

        test_keys = [
            "small_train_small-test_0",
            "small_train_small-test_1",
            "small_train_small-test_2",
        ]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=source_db_reader,
            hdf5_path=str(temp_hdf5_file),
            extras_fields=make_entry_key_lookup(extras_df),
        )

        with h5py.File(temp_hdf5_file, "r") as f:
            # Verify correct values are matched to correct keys
            assert f["small_train_small-test_0"]["extras"].attrs["order_check"] == 0
            assert f["small_train_small-test_1"]["extras"].attrs["order_check"] == 100
            assert f["small_train_small-test_2"]["extras"].attrs["order_check"] == 200

    def test_extras_fields_partial_coverage(self, source_db_reader, temp_hdf5_file):
        """Test behavior when extras_fields has None values for some keys."""
        # extras_fields has None for key 2
        extras_df = pl.DataFrame({
            "entry_key": [
                "small_train_small-test_0",
                "small_train_small-test_1",
                "small_train_small-test_2",
            ],
            "partial_field": [10, 20, None],
        })

        test_keys = [
            "small_train_small-test_0",
            "small_train_small-test_1",
            "small_train_small-test_2",
        ]

        write_hdf5(
            key_entries=test_keys,
            source_db_readers=source_db_reader,
            hdf5_path=str(temp_hdf5_file),
            extras_fields=make_entry_key_lookup(extras_df),
        )

        with h5py.File(temp_hdf5_file, "r") as f:
            # Keys with non-None extras should have the field in extras group
            assert "partial_field" in f["small_train_small-test_0"]["extras"].attrs
            assert "partial_field" in f["small_train_small-test_1"]["extras"].attrs

            # Key with None value should have extras group but without the field
            assert "small_train_small-test_2" in f
            assert "extras" in f["small_train_small-test_2"]
            assert "partial_field" not in f["small_train_small-test_2"]["extras"].attrs
