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
import lzma
import pickle

import ase.io
import numpy as np
import pytest

from chemreporter.source_database_tools.open_catalyst_20.catalyst_data_parser import (
    get_catalyst_data,
)
from chemreporter.source_database_tools.open_catalyst_20.supplementary_info_impl import (  # noqa: E501
    sidecar_path,
)
from chemreporter.source_database_tools.source_db_implementations import (
    XyzDatabaseImplementationOc20,
)


def create_mock_oc20_data(tmp_path):
    """Create mock OC20 files for testing."""
    # Create mock extxyz data with energy and forces
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    atoms.info["energy"] = -181.54722937
    atoms.arrays["forces"] = np.zeros((3, 3))

    xyz_path = tmp_path / "test.extxyz.xz"
    with lzma.open(xyz_path, "wt") as f:
        ase.io.write(f, atoms, format="extxyz")

    # Create mock supplementary info
    txt_path = tmp_path / "test.txt.xz"
    with lzma.open(txt_path, "wt") as f:
        f.write("sid_test_1,frame258,-181.54722937\n")
    return xyz_path, txt_path


def test_oc20_implementation(tmp_path):
    """Test XyzDatabaseImplementationOc20 with compressed files."""
    oc20_impl = XyzDatabaseImplementationOc20()

    xyz_path, txt_path = create_mock_oc20_data(tmp_path)

    # Mock supplementary_info as it's normally populated by read_supplementary_info
    oc20_impl.supplementary_info = {
        xyz_path.name: {
            0: {
                "sid": "sid_test_1",
                "frame_number": "frame258",
                "ref_energy": -181.54722937,
            }
        }
    }

    assert sidecar_path(xyz_path) == txt_path

    # Test read_file
    atoms_list = oc20_impl.read_file(xyz_path)
    assert len(atoms_list) == 1
    assert atoms_list[0].info["sid"] == "sid_test_1"
    assert atoms_list[0].info["frame_number"] == "frame258"

    # Verify calculator is attached and results are accessible
    assert atoms_list[0].calc is not None
    assert "energy" in atoms_list[0].calc.results
    assert "forces" in atoms_list[0].calc.results


def create_mock_oc20_uncompressed_data(tmp_path):
    """Create mock uncompressed OC20 files for testing."""
    # Create mock extxyz data
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    xyz_path = tmp_path / "test.extxyz"
    ase.io.write(str(xyz_path), atoms, format="extxyz")

    # Create mock supplementary info
    txt_path = tmp_path / "test.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("sid_test_2,frame34,-125.77005553\n")
    return xyz_path, txt_path


def test_oc20_uncompressed(tmp_path):
    """Test XyzDatabaseImplementationOc20 with uncompressed files."""
    oc20_impl = XyzDatabaseImplementationOc20()

    xyz_path, txt_path = create_mock_oc20_uncompressed_data(tmp_path)

    # Mock supplementary_info
    oc20_impl.supplementary_info = {
        xyz_path.name: {
            0: {
                "sid": "sid_test_2",
                "frame_number": "frame34",
                "ref_energy": -125.77005553,
            }
        }
    }

    assert sidecar_path(xyz_path) == txt_path

    # Test read_file
    atoms_list = oc20_impl.read_file(xyz_path)
    assert len(atoms_list) == 1
    assert atoms_list[0].info["sid"] == "sid_test_2"
    assert atoms_list[0].info["frame_number"] == "frame34"


def test_oc20_missing_file_in_supplementary_info(tmp_path):
    """Test error when file name is not in supplementary_info."""
    oc20_impl = XyzDatabaseImplementationOc20()

    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    xyz_path = tmp_path / "test_missing.extxyz.xz"
    with lzma.open(xyz_path, "wt") as f:
        ase.io.write(f, atoms, format="extxyz")

    # Set up supplementary_info with wrong file name
    oc20_impl.supplementary_info = {
        "wrong_filename.extxyz.xz": {
            0: {"frame_number": "frame1", "ref_energy": -100.0}
        }
    }

    with pytest.raises(ValueError, match=r"not found in supplementary info"):
        oc20_impl.read_file(xyz_path)


def test_oc20_frame_number_in_catalyst_data(tmp_path):
    """Test that frame_number is properly used in get_catalyst_data."""
    oc20_impl = XyzDatabaseImplementationOc20()

    # Create atoms with OC20-specific tags (0=bulk, 1=surface, 2=adsorbate)
    atoms = ase.Atoms(
        "H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]], tags=[2, 2, 1]
    )
    atoms.info["frame_number"] = "frame123"
    atoms.info["bulk_id"] = 8310
    atoms.info["ads_id"] = 49

    xyz_path = tmp_path / "test_catalyst.extxyz.xz"
    with lzma.open(xyz_path, "wt") as f:
        ase.io.write(f, atoms, format="extxyz")

    oc20_impl.supplementary_info = {
        xyz_path.name: {
            0: {
                "frame_number": "frame123",
                "ref_energy": -150.0,
                "bulk_id": 8310,
                "ads_id": 49,  #
            }
        }
    }

    atoms_list = oc20_impl.read_file(xyz_path)
    assert len(atoms_list) == 1
    assert atoms_list[0].info["frame_number"] == "frame123"

    # Test that get_catalyst_data uses frame_number correctly
    catalyst_data = get_catalyst_data(atoms_list[0])
    assert catalyst_data["catalyst_relaxation_frame_idx"] == 123  # int, "frame" removed


def test_oc20_multiple_frames_with_frame_numbers(tmp_path):
    """Test reading multiple frames, each with frame_number."""
    oc20_impl = XyzDatabaseImplementationOc20()

    atoms1 = ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 1]])
    atoms2 = ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.1]])
    atoms3 = ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 1.2]])

    xyz_path = tmp_path / "test_multi.extxyz.xz"
    with lzma.open(xyz_path, "wt") as f:
        ase.io.write(f, [atoms1, atoms2, atoms3], format="extxyz")

    oc20_impl.supplementary_info = {
        xyz_path.name: {
            0: {"frame_number": "frame100", "ref_energy": -100.0},
            1: {"frame_number": "frame200", "ref_energy": -101.0},
            2: {"frame_number": "frame300", "ref_energy": -102.0},
        }
    }

    atoms_list = oc20_impl.read_file(xyz_path)
    assert len(atoms_list) == 3
    assert atoms_list[0].info["frame_number"] == "frame100"
    assert atoms_list[1].info["frame_number"] == "frame200"
    assert atoms_list[2].info["frame_number"] == "frame300"
    assert atoms_list[0].info["ref_energy"] == -100.0
    assert atoms_list[1].info["ref_energy"] == -101.0
    assert atoms_list[2].info["ref_energy"] == -102.0


def test_oc20_missing_sid_in_pickle(tmp_path):
    """Test error when sid in txt file is not found in pickle."""
    oc20_impl = XyzDatabaseImplementationOc20()

    # Create pickle with one sid
    pickle_data = {
        "existing_sid": {
            "bulk_id": "bulk_test",
            "ads_id": "ads_test",
            "class": 0,
            "anomaly": 0,
        }
    }

    pickle_path = tmp_path / "oc20_data_mapping.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(pickle_data, f)

    # Create xyz file
    atoms = ase.Atoms("H2O", positions=[[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    xyz_path = tmp_path / "test_missing_sid.extxyz.xz"
    with lzma.open(xyz_path, "wt") as f:
        ase.io.write(f, atoms, format="extxyz")

    # Create txt file with a DIFFERENT sid that's NOT in pickle
    txt_path = tmp_path / "test_missing_sid.txt.xz"
    txt_content = "missing_sid_12345,frame100,-150.0\n"
    with lzma.open(txt_path, "wt") as f:
        f.write(txt_content)

    files_index = {0: xyz_path}

    # Should raise error with helpful message showing available keys
    with pytest.raises(
        ValueError, match=r"Supplementary info not found for sid='missing_sid_12345'"
    ):
        oc20_impl.read_supplementary_info(
            files_dir=tmp_path, files_index=files_index, download_function=None
        )


def test_oc20_catalyst_data_with_complete_fields(tmp_path):
    """Test that get_catalyst_data works with all fields present and not None."""
    oc20_impl = XyzDatabaseImplementationOc20()

    # Create complete pickle data with integer IDs
    pickle_data = {
        "complete_sid": {
            "bulk_id": 1234,  # Use integer instead of string
            "ads_id": 5678,  # Use integer instead of string
            "bulk_mpid": "mp-999",
            "bulk_symbols": "Au16",
            "ads_symbols": "*CH3",
            "miller_index": (1, 1, 0),
            "shift": 0.75,
            "top": True,
            "adsorption_site": "bridge",
            "class": 2,  # non-metals
            "anomaly": 3,  # surface_reconstruction
            "split": "test",
        }
    }

    pickle_path = tmp_path / "oc20_data_mapping.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump(pickle_data, f)

    # Create atoms with OC20 tags
    atoms = ase.Atoms(
        "CHHHOO",
        positions=[
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
        ],
        tags=[2, 2, 2, 2, 1, 1],  # 0=bulk, 1=surface, 2=adsorbate
    )
    atoms.info["energy"] = -200.0

    xyz_path = tmp_path / "complete.extxyz.xz"
    with lzma.open(xyz_path, "wt") as f:
        ase.io.write(f, atoms, format="extxyz")

    # Create txt file
    txt_path = tmp_path / "complete.txt.xz"
    txt_content = "complete_sid,frame500,-200.0\n"
    with lzma.open(txt_path, "wt") as f:
        f.write(txt_content)

    # Load supplementary info
    files_index = {0: xyz_path}
    oc20_impl.read_supplementary_info(
        files_dir=tmp_path, files_index=files_index, download_function=None
    )

    # Read atoms with supplementary info
    atoms_list = oc20_impl.read_file(xyz_path)
    assert len(atoms_list) == 1

    loaded_atoms = atoms_list[0]

    # Verify all fields are present and not None in atoms.info
    assert loaded_atoms.info["frame_number"] == "frame500"
    assert loaded_atoms.info["frame_number"] is not None
    assert loaded_atoms.info["bulk_id"] == 1234
    assert loaded_atoms.info["bulk_id"] is not None
    assert loaded_atoms.info["ads_id"] == 5678
    assert loaded_atoms.info["ads_id"] is not None
    assert loaded_atoms.info["bulk_symbols"] == "Au16"
    assert loaded_atoms.info["bulk_symbols"] is not None
    assert loaded_atoms.info["ads_symbols"] == "*CH3"
    assert loaded_atoms.info["ads_symbols"] is not None

    # Test get_catalyst_data doesn't fail with AttributeError
    catalyst_data = get_catalyst_data(loaded_atoms)

    # Verify all catalyst data fields are not None
    assert (
        catalyst_data["catalyst_relaxation_frame_idx"] == 500
    )  # int, "frame" stripped
    assert catalyst_data["catalyst_relaxation_frame_idx"] is not None
    assert catalyst_data["catalyst_bulk_id"] == 1234  # Integer value
    assert catalyst_data["catalyst_bulk_id"] is not None
    assert catalyst_data["catalyst_adsorbate_id"] == 5678  # Integer value
    assert catalyst_data["catalyst_adsorbate_id"] is not None
    assert catalyst_data["catalyst_bulk_symbols"] == "Au16"
    assert catalyst_data["catalyst_bulk_symbols"] is not None
    assert catalyst_data["catalyst_adsorbate_symbols"] == "*CH3"
    assert catalyst_data["catalyst_adsorbate_symbols"] is not None
    assert catalyst_data["catalyst_miller_index"] == (1, 1, 0)
    assert catalyst_data["catalyst_miller_index"] is not None
    assert catalyst_data["catalyst_class"] == "non-metals"  # mapped from 2
    assert catalyst_data["catalyst_class"] is not None
    assert (
        catalyst_data["catalyst_anomaly"] == "surface_reconstruction"
    )  # mapped from 3
    assert catalyst_data["catalyst_anomaly"] is not None
    assert catalyst_data["catalyst_reference_energy"] == -200.0
    assert catalyst_data["catalyst_reference_energy"] is not None
    assert catalyst_data["catalyst_num_adsorbate_atoms"] == 4  # C, H, H, H with tag 2
    assert catalyst_data["catalyst_num_bulk_atoms"] == 2  # O, O with tags 0+1
    assert isinstance(catalyst_data["catalyst_substrate_height"], float)
    assert (
        catalyst_data["catalyst_adsorbate_smiles"] is not None
    )  # SMILES should be generated
