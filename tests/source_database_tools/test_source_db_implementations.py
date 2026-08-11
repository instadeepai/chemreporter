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
"""Unit tests for source_db_implementations.py."""

from unittest.mock import MagicMock, patch

import ase
import ase.io
import numpy as np
import pytest
from ase.calculators.singlepoint import SinglePointCalculator

from chemreporter.source_database_tools.database_item import DatasetItem
from chemreporter.source_database_tools.exceptions import SourceDatabaseReaderUsageError
from chemreporter.source_database_tools.source_db_implementations import (
    AseDBDatasetImplementation,
    AselmdbDatabaseImplementationOc20,
    AselmdbDatabaseImplementationOdac23,
    AselmdbDatabaseImplementationOmat,
    AselmdbDatabaseImplementationOmc25,
    AselmdbDatabaseImplementationOmol25,
    XyzDatabaseImplementation,
    correct_database_name,
)


def _water_dataset_item(**info) -> DatasetItem:
    atoms = ase.Atoms(
        "H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]], info=dict(info)
    )
    return DatasetItem(
        database_name="omol25-1",
        split_name="train",
        key="omol25_train_shard1_0",
        atoms=atoms,
        name_mapping={},
        additional_fields=[],
    )


def test_correct_database_name_splits_subset():
    """A database name with a non-alphanumeric separator yields a subset."""
    item = _water_dataset_item()
    item.database_name = "omat24-rattled-1000"

    result = correct_database_name(item)

    assert result == {"database_name": "omat24", "subset": "rattled-1000"}


def test_correct_database_name_raises_without_separator():
    """A database name with no separator cannot be split into a subset."""
    item = _water_dataset_item()
    item.database_name = "omat24"

    with pytest.raises(SourceDatabaseReaderUsageError, match="non-alphanumeric"):
        correct_database_name(item)


@pytest.mark.parametrize(
    "data_id, source, reference_source",
    [
        ("rgd", "rgd_uks/MR_693393_1_13_0_1/orca.tar.zst", None),
        ("reactivity", "pmechdb/r_12345_step3/orca.tar.zst", None),
        ("ani1", "ani1x/mol_5_2/orca.tar.zst", None),
        ("trans1x", "trans1x/mol_5_3_2/orca.tar.zst", None),
        ("metal_complexes", "tm_react/step2_0/orca.tar.zst", None),
        (
            "metal_complexes",
            "ground/mol_1/orca.tar.zst",
            "ref/mol_1_step2/orca.tar.zst",
        ),
        ("something_else", "some/source", None),
        (None, None, None),
    ],
)
def test_omol25_additional_ase_info_covers_all_reactivity_branches(
    data_id, source, reference_source
):
    """Every match-case branch (including the default) returns a plain dict."""
    impl = AselmdbDatabaseImplementationOmol25()
    item = _water_dataset_item(
        data_id=data_id, source=source, reference_source=reference_source
    )

    result = impl.get_additional_ase_info_omol25(item)

    assert isinstance(result, dict)


def test_omc25_implementation_derived_fields():
    """OMC25's derived fields include composition, num_atoms, charge, and spin."""
    impl = AselmdbDatabaseImplementationOmc25()
    assert impl.database_format == "aselmdb_omc"
    item = _water_dataset_item()

    result = impl.omc25_derived_fields(item)

    assert result["composition"] == "H2O"
    assert result["num_atoms"] == 3
    assert result["net_charge"] == 0
    assert result["spin_multiplicity"] == 1


def test_omat_implementation_additional_fields():
    """OMAT's additional fields include num_atoms plus charge/spin."""
    impl = AselmdbDatabaseImplementationOmat()
    assert impl.database_format == "aselmdb_omat"
    assert correct_database_name in impl.get_additional_fields
    item = _water_dataset_item()

    result = impl.get_additional_ase_info_omat(item)

    assert result["num_atoms"] == 3
    assert result["net_charge"] == 0


def test_odac23_implementation_additional_fields():
    """ODAC's additional fields include num_atoms, charge/spin, and composition."""
    impl = AselmdbDatabaseImplementationOdac23()
    assert impl.database_format == "aselmdb_odac"
    item = _water_dataset_item()

    result = impl.get_additional_ase_info_odac(item)

    assert result["num_atoms"] == 3
    assert result["net_charge"] == 0
    assert result["spin_multiplicity"] == 1
    assert result["composition"] == "H2O"


def test_oc20_aselmdb_init_sets_expected_fields():
    """The aselmdb OC20 implementation wires up its extra fields and file name."""
    impl = AselmdbDatabaseImplementationOc20()

    assert impl.database_format == "aselmdb_oc"
    assert impl.file_extension == "aselmdb"
    assert impl._current_oc20_shard_filename is None


@patch("chemreporter.source_database_tools.source_db_implementations.AseDBDataset")
def test_oc20_aselmdb_read_file_tracks_shard_filename(mock_ase_db_dataset):
    """read_file remembers the shard filename before delegating to the base class."""
    impl = AselmdbDatabaseImplementationOc20()
    mock_dataset = MagicMock()
    mock_ase_db_dataset.return_value = mock_dataset

    result = impl.read_file("/some/dir/shard_007.aselmdb")

    assert impl._current_oc20_shard_filename == "shard_007.aselmdb"
    mock_ase_db_dataset.assert_called_once_with({"src": "/some/dir/shard_007.aselmdb"})
    assert result is mock_dataset


def test_oc20_aselmdb_get_atoms_from_db_pointer_sets_num_atoms():
    """num_atoms is injected into atoms.info when missing from the ASE atoms."""
    impl = AselmdbDatabaseImplementationOc20()
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    mock_db_pointer = MagicMock()
    mock_db_pointer.get_atoms.return_value = atoms

    result = impl.get_atoms_from_db_pointer(mock_db_pointer, 0)

    assert result.info["num_atoms"] == 3


def test_ase_db_dataset_implementation_rehydrates_missing_calculator():
    """A calculator is rebuilt from atoms.info/arrays when none is attached."""
    impl = AseDBDatasetImplementation("aselmdb_omol")
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    atoms.info["energy"] = -76.4
    atoms.new_array("forces", np.zeros((3, 3)))

    mock_db_pointer = MagicMock()
    mock_db_pointer.get_atoms.return_value = atoms

    result = impl.get_atoms_from_db_pointer(mock_db_pointer, 0)

    assert isinstance(result.calc, SinglePointCalculator)
    assert result.calc.results["energy"] == -76.4
    np.testing.assert_array_equal(result.calc.results["forces"], np.zeros((3, 3)))


def test_ase_db_dataset_implementation_keeps_existing_calculator():
    """An already-attached calculator is left untouched."""
    impl = AseDBDatasetImplementation("aselmdb_omol")
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-1.0)

    mock_db_pointer = MagicMock()
    mock_db_pointer.get_atoms.return_value = atoms

    result = impl.get_atoms_from_db_pointer(mock_db_pointer, 0)

    assert result.calc.results["energy"] == -1.0


def test_ase_db_dataset_implementation_close_closes_underlying_envs():
    """close() closes each db's env (if present) and the db itself."""
    impl = AseDBDatasetImplementation("aselmdb_omol")

    db_with_env = MagicMock()
    db_without_close = MagicMock(spec=["env"])
    db_without_close.env.close = MagicMock()

    mock_db_pointer = MagicMock()
    mock_db_pointer.dbs = [db_with_env, db_without_close]

    impl.close(mock_db_pointer)

    db_with_env.env.close.assert_called_once()
    db_with_env.close.assert_called_once()
    db_without_close.env.close.assert_called_once()


def test_source_database_implementation_default_hooks_are_no_ops():
    """The base class's default close/read_supplementary_info hooks do nothing."""
    impl = XyzDatabaseImplementation()

    # Neither call should raise, and both are no-op "pass" implementations.
    assert impl.close(MagicMock()) is None
    assert (
        impl.read_supplementary_info(files_dir="/tmp", files_index={}) is None  # noqa: S108
    )


def test_xyz_database_implementation_read_file_attaches_calculator(tmp_path):
    """read_file attaches a SinglePointCalculator when energy/forces are present."""
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    atoms.info["energy"] = -76.4
    atoms.new_array("forces", np.ones((3, 3)))
    # ase.io.read auto-attaches a calculator for standard extxyz energy/forces
    # columns, which would bypass the manual fallback below it is meant to
    # cover; patch ase.io.read to return the raw atoms instead, as if the file
    # only stored these as plain info/array entries.
    xyz_path = tmp_path / "with_calc.extxyz"
    ase.io.write(str(xyz_path), atoms, format="extxyz")

    impl = XyzDatabaseImplementation()
    with patch(
        "chemreporter.source_database_tools.source_db_implementations.ase.io.read",
        return_value=[atoms],
    ):
        atoms_list = impl.read_file(xyz_path)

    assert len(atoms_list) == 1
    assert atoms_list[0].calc is not None
    assert atoms_list[0].calc.results["energy"] == pytest.approx(-76.4)
    np.testing.assert_allclose(atoms_list[0].calc.results["forces"], np.ones((3, 3)))


def test_xyz_database_implementation_read_file_without_calculator(tmp_path):
    """read_file leaves atoms without a calculator when no energy/forces are stored."""
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]])

    xyz_path = tmp_path / "without_calc.extxyz"
    ase.io.write(str(xyz_path), atoms, format="extxyz")

    impl = XyzDatabaseImplementation()
    atoms_list = impl.read_file(xyz_path)

    assert len(atoms_list) == 1
    assert atoms_list[0].calc is None

    result = impl.get_atoms_from_db_pointer(atoms_list, 0)
    assert result is atoms_list[0]
