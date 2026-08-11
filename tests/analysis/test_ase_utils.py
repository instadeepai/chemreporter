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
"""Tests for ASE utilities module,."""

from unittest.mock import Mock

import ase
import numpy as np
import pytest

from chemreporter.analysis.ase_utils import (
    calculate_dipole_moment,
    get_atoms_with_tags,
    get_forces,
    get_max_forces_norm,
    get_molecular_weight,
    get_net_forces_norm,
    get_unique_chemical_symbols,
    process_ase_info,
)
from chemreporter.source_database_tools.source_db_implementations import FIELDS


@pytest.fixture
def mock_atoms_with_forces():
    """Create a mock ASE Atoms object with forces."""
    atoms = Mock(spec=ase.Atoms)
    atoms.arrays = {}
    atoms._calc = Mock()
    atoms._calc.results = {
        "forces": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.2, -0.3, -0.4]])
    }
    return atoms


@pytest.fixture
def mock_atoms_with_charges_and_masses():
    """Create a mock ASE Atoms object with charges and positions."""
    atoms = Mock(spec=ase.Atoms)
    atoms.info = {
        "mulliken_charges": [0.5, -0.5, 0.3, -0.3],
        "data_id": "test_set",
        "charge": 0,
        "num_atoms": 4,
        "spin": 0.0,
        "source": "test_source",
        "reference_source": "test_reference_source",
        "composition": "C1H3",
    }
    atoms.arrays = {}
    atoms.get_positions.return_value = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    atoms.get_center_of_mass.return_value = np.array([0.25, 0.25, 0.25])
    atoms.get_atomic_numbers.return_value = np.array([6, 1, 1, 1])
    atoms.get_chemical_symbols.return_value = ["C", "H", "H", "H"]
    atoms.positions = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    atoms._calc = Mock()
    atoms._calc.results = {
        "forces": np.array([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [-0.2, -0.3, -0.4],
            [0.0, 0.0, 0.0],
        ])
    }
    atoms.get_masses.return_value = np.array([12.011, 1.008, 1.008, 1.008])
    return atoms


class TestGetForces:
    """Test get_forces function."""

    def test_get_forces_returns_array(self, mock_atoms_with_forces):
        """Test that get_forces returns a numpy array."""
        forces = get_forces(mock_atoms_with_forces)
        assert isinstance(forces, np.ndarray)

    def test_get_forces_correct_shape(self, mock_atoms_with_forces):
        """Test that forces have the correct shape."""
        forces = get_forces(mock_atoms_with_forces)
        assert forces.shape == (3, 3)

    def test_get_forces_correct_values(self, mock_atoms_with_forces):
        """Test that forces have correct values."""
        forces = get_forces(mock_atoms_with_forces)
        expected = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.2, -0.3, -0.4]])
        np.testing.assert_array_equal(forces, expected)


class TestGetNetForcesNorm:
    """Test get_sum_forces_norm function."""

    def test_net_forces_norm_returns_scalar(self, mock_atoms_with_forces):
        """Test that sum forces norm returns a scalar."""
        result = get_net_forces_norm(mock_atoms_with_forces)
        assert isinstance(result, (float, np.floating))

    def test_net_forces_norm_positive(self, mock_atoms_with_forces):
        """Test that sum forces norm is positive."""
        result = get_net_forces_norm(mock_atoms_with_forces)
        assert result >= 0

    def test_net_forces_norm_calculation(self, mock_atoms_with_forces):
        """Test the calculation is correct."""
        forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.2, -0.3, -0.4]])
        expected = np.linalg.norm(forces.sum(axis=0))
        result = get_net_forces_norm(mock_atoms_with_forces)
        np.testing.assert_almost_equal(result, expected)


class TestGetUniqueChemicalSymbols:
    """Test get_unique_chemical_symbols function."""

    def test_get_unique_chemical_symbols_returns_string(
        self, mock_atoms_with_charges_and_masses
    ):
        """Test that get_unique_chemical_symbols returns a string."""
        result = get_unique_chemical_symbols(mock_atoms_with_charges_and_masses)
        assert isinstance(result, str)

    def test_get_unique_chemical_symbols_correct_values(
        self, mock_atoms_with_charges_and_masses
    ):
        """Test that get_unique_chemical_symbols returns the correct values."""
        result = get_unique_chemical_symbols(mock_atoms_with_charges_and_masses)
        expected = "CH"
        assert result == expected


class TestGetMolecularWeight:
    """Test get_molecular_weight function."""

    def test_get_molecular_weight_returns_float(
        self, mock_atoms_with_charges_and_masses
    ):
        """Test that molecular weight returns a float."""
        result = get_molecular_weight(mock_atoms_with_charges_and_masses)
        assert isinstance(result, float)

    def test_get_molecular_weight_positive(self, mock_atoms_with_charges_and_masses):
        """Test that molecular weight is positive."""
        result = get_molecular_weight(mock_atoms_with_charges_and_masses)
        assert result >= 0

    def test_get_molecular_weight_calculation(
        self,
    ):
        """Test the calculation is correct."""
        atoms = Mock(spec=ase.Atoms)
        atoms.get_masses.return_value = np.array([12.011, 1.008, 1.008, 1.008])
        result = get_molecular_weight(atoms)
        assert round(result, 3) == 15.035


class TestGetMaxForcesNorm:
    """Test get_max_forces_norm function."""

    def test_max_forces_norm_returns_scalar(self, mock_atoms_with_forces):
        """Test that max forces norm returns a scalar."""
        result = get_max_forces_norm(mock_atoms_with_forces)
        assert isinstance(result, (float, np.floating))

    def test_max_forces_norm_positive(self, mock_atoms_with_forces):
        """Test that max forces norm is positive."""
        result = get_max_forces_norm(mock_atoms_with_forces)
        assert result >= 0

    def test_max_forces_norm_calculation(self, mock_atoms_with_forces):
        """Test the calculation is correct."""
        forces = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [-0.2, -0.3, -0.4]])
        expected = np.linalg.norm(forces, axis=1).max()
        result = get_max_forces_norm(mock_atoms_with_forces)
        np.testing.assert_almost_equal(result, expected)


class TestCalculateDipoleMoment:
    """Test calculate_dipole_moment function."""

    def test_dipole_moment_returns_float(self, mock_atoms_with_charges_and_masses):
        """Test that dipole moment returns a float."""
        result = calculate_dipole_moment(mock_atoms_with_charges_and_masses)
        assert isinstance(result, (float, np.floating))

    def test_dipole_moment_positive(self, mock_atoms_with_charges_and_masses):
        """Test that dipole moment is positive (magnitude)."""
        result = calculate_dipole_moment(mock_atoms_with_charges_and_masses)
        assert result >= 0

    def test_dipole_moment_custom_charges_type(
        self, mock_atoms_with_charges_and_masses
    ):
        """Test dipole moment calculation with a custom charges_type."""
        mock_atoms_with_charges_and_masses.info = {"custom_charges": [0.1, -0.1]}
        mock_atoms_with_charges_and_masses.get_center_of_mass.return_value = np.array([
            0.5,
            0.0,
            0.0,
        ])

        result = calculate_dipole_moment(
            mock_atoms_with_charges_and_masses, charges_type="custom_charges"
        )
        assert isinstance(result, (float, np.floating))

        assert result >= 0

    def test_dipole_moment_zero_for_neutral_symmetric(self):
        """Test that symmetric neutral molecule has near-zero dipole."""
        atoms = Mock(spec=ase.Atoms)
        # Checkerboard charges on the corners of a unit square: each charge
        # is exactly canceled by its diagonal opposite, so the dipole is 0.
        atoms.info = {"mulliken_charges": [0.5, -0.5, -0.5, 0.5]}
        atoms.arrays = {}
        atoms.get_positions.return_value = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ])
        atoms.get_center_of_mass.return_value = np.array([0.5, 0.5, 0.0])

        result = calculate_dipole_moment(atoms)
        assert result == pytest.approx(0.0, abs=1e-9)


class TestExtractAseInfo:
    """Test extract_ase_info function."""

    def test_extract_ase_info_returns_dict(self, mock_atoms_with_charges_and_masses):
        """Test that extract_ase_info returns a dictionary."""
        result = process_ase_info(
            mock_atoms_with_charges_and_masses,
            fields_name_mapping=FIELDS["aselmdb_omol"],
        )
        assert isinstance(result, dict)

    def test_extract_ase_info_has_required_keys_omol25(
        self, mock_atoms_with_charges_and_masses
    ):
        """Test that all required keys are present."""
        result = process_ase_info(
            mock_atoms_with_charges_and_masses,
            fields_name_mapping=FIELDS["aselmdb_omol"],
        )
        expected_keys = [
            "subset",
            "net_charge",
            "num_atoms",
            "spin_multiplicity",
            "positions",
            "atomic_numbers",
        ]
        for key in expected_keys:
            assert key in result

    def test_extract_ase_info_positions_is_list(
        self, mock_atoms_with_charges_and_masses
    ):
        """Test that positions is a list."""
        result = process_ase_info(
            mock_atoms_with_charges_and_masses,
            fields_name_mapping=FIELDS["aselmdb_omol"],
        )
        assert isinstance(result["positions"], list)

    def test_extract_ase_info_atom_numbers_types(
        self, mock_atoms_with_charges_and_masses
    ):
        """Test that atom numbers have correct types."""
        result = process_ase_info(
            mock_atoms_with_charges_and_masses,
            fields_name_mapping=FIELDS["aselmdb_omol"],
        )
        assert isinstance(result["atomic_numbers"], list)
        assert all(isinstance(x, (int, np.integer)) for x in result["atomic_numbers"])
        assert isinstance(result["subset"], str)
        assert isinstance(result["num_atoms"], (int, np.integer))
        assert isinstance(result["spin_multiplicity"], (float, np.floating))
        assert isinstance(result["net_charge"], (int, np.integer))

    def test_extract_ase_info_missing_required_omol25_fields_raises_keyerror(self):
        """Test that missing required fields raise KeyError.

        This test ensures that when atoms.info is missing required fields
        (data_id, charge, num_atoms, spin), a KeyError is raised rather than
        returning None, which would cause:
        ValueError: dictionary update sequence element #0 has length 1; 2 is required
        when used with dict.update().
        """
        atoms = Mock(spec=ase.Atoms)
        # Missing all required fields: data_id, charge, num_atoms, spin
        atoms.info = {}
        atoms.arrays = {}
        atoms.positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        atoms.get_atomic_numbers.return_value = np.array([6, 1])

        with pytest.raises(KeyError):
            process_ase_info(atoms, fields_name_mapping=FIELDS["aselmdb_omol"])

    def test_extract_ase_info_missing_charge_field_raises_keyerror(self):
        """Test that missing 'charge' field raises KeyError."""
        atoms = Mock(spec=ase.Atoms)
        # Missing only 'charge' field
        atoms.info = {
            "data_id": "test_set",
            # "charge": 0,  # Missing this one
            "num_atoms": 2,
            "spin": 0.0,
        }
        atoms.arrays = {}
        atoms.positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        atoms.get_atomic_numbers.return_value = np.array([6, 1])

        with pytest.raises(KeyError, match="charge"):
            process_ase_info(atoms, fields_name_mapping=FIELDS["aselmdb_omol"])


def test_get_atoms_with_tags_returns_atoms_with_given_tag():
    """get_atoms_with_tags returns an ASE Atoms object w
    ith only atoms matching the tag(s).
    """
    atoms = ase.Atoms(
        symbols=["C", "H", "O", "N"],
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]],
    )
    # ASE tags: 0=bulk, 1=surface, 2=adsorbate
    atoms.set_tags([0, 0, 2, 2])
    subset = get_atoms_with_tags(atoms, 2)
    assert isinstance(subset, ase.Atoms)
    assert len(subset) == 2
    assert list(subset.get_chemical_symbols()) == ["O", "N"]
    assert list(subset.get_tags()) == [2, 2]
    # single tag as int and list give same result
    subset_list = get_atoms_with_tags(atoms, [2])
    assert len(subset_list) == 2
    assert list(subset_list.get_tags()) == [2, 2]
