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
"""Tests for OpenBabel utilities module."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from ase.io import read

from chemreporter.analysis.obabel_utils import (
    _compute_charge_spin_multiplicity,  # noqa: PLC2701
    _compute_smiles_from_positions,  # noqa: PLC2701
    _ob_conversion,  # noqa: PLC2701
    _ob_mol,  # noqa: PLC2701
    get_charge_spin_multiplicity,
    heuristic_lowest_energy_charge_spin_multiplicity,
    ob_mol_from_positions,
    smiles_from_positions,
)

# Test data directory
TESTS_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@pytest.fixture(autouse=True)
def _clear_openbabel_caches():
    """Drop cached OBMol/OBConversion so mocks cannot poison later tests."""
    _ob_mol.cache_clear()
    _ob_conversion.cache_clear()
    yield
    _ob_mol.cache_clear()
    _ob_conversion.cache_clear()


class TestOpenBabelSingletonReuse:
    """OBMol/OBConversion are reused across calls (memory-leak fix)."""

    def test_ob_mol_returns_same_instance(self):
        """Regression: _ob_mol reuses one OBMol instead of allocating per call."""
        assert _ob_mol() is _ob_mol()

    def test_ob_conversion_returns_same_instance(self):
        """Regression: _ob_conversion reuses one OBConversion across calls."""
        assert _ob_conversion() is _ob_conversion()


class TestSmilesFromCoords:
    """Test smiles_from_coords function."""

    @patch("chemreporter.analysis.obabel_utils.openbabel")
    def test_smiles_from_coords_returns_string(self, mock_openbabel):
        """Test that smiles_from_coords returns a string."""
        # Setup mocks
        mock_conversion = MagicMock()
        mock_mol = MagicMock()
        mock_atom = MagicMock()

        mock_openbabel.OBConversion.return_value = mock_conversion
        mock_openbabel.OBMol.return_value = mock_mol
        mock_mol.NewAtom.return_value = mock_atom
        mock_conversion.WriteString.return_value = "C"
        atomic_numbers = [6, 1, 1, 1, 1]
        positions = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]

        result = _compute_smiles_from_positions(atomic_numbers, positions, 0)
        assert result == "C"
        assert isinstance(result, str)

    @patch("chemreporter.analysis.obabel_utils.openbabel")
    def test_smiles_from_coords_calls_conversion(self, mock_openbabel):
        """Test that conversion methods are called."""
        mock_conversion = MagicMock()
        mock_mol = MagicMock()
        mock_atom = MagicMock()

        mock_openbabel.OBConversion.return_value = mock_conversion
        mock_openbabel.OBMol.return_value = mock_mol
        mock_mol.NewAtom.return_value = mock_atom
        mock_conversion.WriteString.return_value = "CCO"

        atomic_numbers = [6, 6, 8]
        positions = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [2.5, 0.0, 0.0]]

        result = _compute_smiles_from_positions(atomic_numbers, positions, 0)

        # Verify conversion setup
        mock_conversion.SetInAndOutFormats.assert_called_once_with("xyz", "smi")

        # Verify atoms were added
        assert mock_mol.NewAtom.call_count == 3
        assert result == "CCO"
        # Verify bond perception
        mock_mol.ConnectTheDots.assert_called_once()
        mock_mol.PerceiveBondOrders.assert_called_once()

        # Verify write was called
        mock_conversion.WriteString.assert_called_once_with(mock_mol)

    @patch("chemreporter.analysis.obabel_utils.openbabel")
    def test_smiles_from_coords_sets_atomic_numbers(self, mock_openbabel):
        """Test that atomic numbers are set correctly."""
        mock_conversion = MagicMock()
        mock_mol = MagicMock()
        mock_atom = MagicMock()

        mock_openbabel.OBConversion.return_value = mock_conversion
        mock_openbabel.OBMol.return_value = mock_mol
        mock_mol.NewAtom.return_value = mock_atom
        mock_conversion.WriteString.return_value = "N"

        atomic_numbers = [7, 1, 1, 1]  # Nitrogen with hydrogens
        positions = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]

        result = _compute_smiles_from_positions(atomic_numbers, positions, 0)
        assert result == "N"

    @patch("chemreporter.analysis.obabel_utils.openbabel")
    def test_smiles_from_coords_empty_lists(self, mock_openbabel):
        """Test with empty atomic numbers and positions."""
        mock_conversion = MagicMock()
        mock_mol = MagicMock()

        mock_openbabel.OBConversion.return_value = mock_conversion
        mock_openbabel.OBMol.return_value = mock_mol
        mock_conversion.WriteString.return_value = ""

        atomic_numbers = []
        positions = []

        result = _compute_smiles_from_positions(atomic_numbers, positions, 0)

        # No atoms should be added
        mock_mol.NewAtom.assert_not_called()
        assert isinstance(result, str)
        assert len(result) == 0

    @patch("chemreporter.analysis.obabel_utils.openbabel")
    def test_smiles_from_coords_single_atom(self, mock_openbabel):
        """Test with a single atom."""
        mock_conversion = MagicMock()
        mock_mol = MagicMock()
        mock_atom = MagicMock()

        mock_openbabel.OBConversion.return_value = mock_conversion
        mock_openbabel.OBMol.return_value = mock_mol
        mock_mol.NewAtom.return_value = mock_atom
        mock_conversion.WriteString.return_value = "[He]"

        atomic_numbers = [2]  # Helium
        positions = [[0.0, 0.0, 0.0]]

        result = _compute_smiles_from_positions(atomic_numbers, positions, 0)
        assert result == "[He]"
        # One atom should be added
        mock_mol.NewAtom.assert_called_once()
        assert isinstance(result, str)

    @patch("chemreporter.analysis.obabel_utils.openbabel")
    def test_smiles_from_coords_large_molecule(self, mock_openbabel):
        """Test with a larger molecule."""
        mock_conversion = MagicMock()
        mock_mol = MagicMock()
        mock_atom = MagicMock()

        mock_openbabel.OBConversion.return_value = mock_conversion
        mock_openbabel.OBMol.return_value = mock_mol
        mock_mol.NewAtom.return_value = mock_atom
        mock_conversion.WriteString.return_value = "CCCCCCCCCC"  # Decane

        # 10 carbon atoms
        atomic_numbers = [6] * 10
        positions = [[float(i), 0.0, 0.0] for i in range(10)]

        result = _compute_smiles_from_positions(atomic_numbers, positions, 0)
        assert result == "CCCCCCCCCC"
        # 10 atoms should be added
        assert mock_mol.NewAtom.call_count == 10
        assert isinstance(result, str)

    @patch("chemreporter.analysis.obabel_utils.openbabel")
    def test_smiles_from_coords_bad_input(self, mock_openbabel):
        """Test with bad input."""
        mock_conversion = MagicMock()
        mock_mol = MagicMock()
        mock_atom = MagicMock()

        mock_openbabel.OBConversion.return_value = mock_conversion
        mock_openbabel.OBMol.return_value = mock_mol
        mock_mol.NewAtom.return_value = mock_atom
        mock_conversion.WriteString.return_value = ""

        # negative test case
        atomic_numbers = [6, 6, 10]
        positions = [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.5, 0.0, 0.0]]

        result = _compute_smiles_from_positions(atomic_numbers, positions, 0)
        assert len(result) == 0
        assert isinstance(result, str)


def test_get_charge_spin_multiplicity_water():
    """Test get_charge_spin_multiplicity for water (neutral singlet)."""
    # Water: neutral molecule -> charge 0, singlet
    result = get_charge_spin_multiplicity([8, 1, 1], [[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    assert result["net_charge"] == 0
    assert result["spin_multiplicity"] == 1


def test_get_charge_spin_multiplicity_methane():
    """Test get_charge_spin_multiplicity for methane (neutral singlet)."""
    # CH4: neutral molecule -> charge 0, singlet
    result = get_charge_spin_multiplicity(
        [6, 1, 1, 1, 1],
        [
            [0, 0, 0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63],
        ],
    )
    assert result["net_charge"] == 0
    assert result["spin_multiplicity"] == 1


@pytest.mark.parametrize(
    "sample_file, expected_smiles_substring, net_charge",
    [
        (
            "spice2_sample_1.extxyz",
            "O=C(N[C@H](C(=O)NC)CCC(=O)[O-])C.O=C(NO)CCCCCS",
            -1,
        ),
        (
            "spice2_sample_2.extxyz",
            "O=C(N[C@H](C(=O)NC)CCSC)C.[O-]C(=O)[C@@H](N)C[C@H](C(=O)O)C",
            -1,
        ),
    ],
)
def test_smiles_generation_on_spice2_samples(
    sample_file, expected_smiles_substring, net_charge
):
    """Test SMILES generation on real samples extracted from SPICE2."""
    path = TESTS_DATA_DIR / sample_file
    if not path.exists():
        pytest.skip(f"Sample file not found at {path}")

    atoms = read(path)
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    positions = atoms.get_positions().tolist()

    smiles = smiles_from_positions(atomic_numbers, positions, net_charge)
    assert expected_smiles_substring == smiles


@pytest.mark.parametrize(
    "sample_file, expected_spin",
    [
        ("spice2_sample_1.extxyz", 1),
        ("spice2_sample_2.extxyz", 1),
    ],
)
def test_spin_multiplicity_on_spice2_samples(sample_file, expected_spin):
    """Test spin multiplicity heuristic on real samples extracted from SPICE2."""
    path = TESTS_DATA_DIR / sample_file
    if not path.exists():
        pytest.skip(f"Sample file not found at {path}")

    atoms = read(path)
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    positions = atoms.get_positions().tolist()

    result = get_charge_spin_multiplicity(atomic_numbers, positions)

    assert result["spin_multiplicity"] == expected_spin


@pytest.mark.parametrize(
    "sample_file", ["spice2_sample_1.extxyz", "spice2_sample_2.extxyz"]
)
def test_worker_matches_in_process_implementation(sample_file):
    """The pooled wrapper returns exactly what the in-process code returns.

    ``smiles_from_positions`` runs OpenBabel in a recycled worker process to
    bound the ConnectTheDots leak. The mocked tests above cover the in-process
    implementation, since ``unittest.mock`` patches do not cross a process
    boundary; this covers the dispatch itself being behaviour-neutral.
    """
    path = TESTS_DATA_DIR / sample_file
    if not path.exists():
        pytest.skip(f"Sample file not found at {path}")

    atoms = read(path)
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    positions = atoms.get_positions().tolist()
    net_charge = int(atoms.info.get("total_charge", 0))

    expected = _compute_smiles_from_positions(atomic_numbers, positions, net_charge)

    # Repeat so the worker is exercised across successive tasks.
    for _ in range(3):
        assert smiles_from_positions(atomic_numbers, positions, net_charge) == expected


class TestComputeChargeSpinMultiplicityDirect:
    """Direct calls to ``_compute_charge_spin_multiplicity``.

    ``get_charge_spin_multiplicity`` dispatches through a forked worker
    process (see ``_ob_worker_pool``), so coverage.py cannot see line
    execution inside it there. Calling the private function directly,
    in-process, exercises the same logic while remaining visible to
    coverage.
    """

    def test_water(self):
        """Water: neutral molecule -> charge 0, singlet."""
        result = _compute_charge_spin_multiplicity(
            [8, 1, 1], [[0, 0, 0], [0, 1, 0], [1, 0, 0]]
        )
        assert result["net_charge"] == 0
        assert result["spin_multiplicity"] == 1

    def test_h_atom(self):
        """Isolated H atom: heuristic favors closed-shell H- over neutral H."""
        result = _compute_charge_spin_multiplicity([1], [[0, 0, 0]])
        assert result["net_charge"] == -1
        assert result["spin_multiplicity"] == 1

    def test_methane(self):
        """CH4: neutral molecule -> charge 0, singlet."""
        result = _compute_charge_spin_multiplicity(
            [6, 1, 1, 1, 1],
            [
                [0, 0, 0],
                [0.63, 0.63, 0.63],
                [-0.63, -0.63, 0.63],
                [-0.63, 0.63, -0.63],
                [0.63, -0.63, -0.63],
            ],
        )
        assert result["net_charge"] == 0
        assert result["spin_multiplicity"] == 1


class TestHeuristicLowestEnergyChargeSpinMultiplicity:
    """Direct tests of the pure scoring/selection heuristic.

    ``score(q, m) = q**2 + 0.3*q + (m - 1)`` (alpha=1.0, beta=0.3, gamma=2.0,
    s=(m-1)/2), and the state with the lowest score is returned.
    """

    def test_prefers_neutral_over_charged(self):
        """0**2 + 0.3*0 = 0 beats 1**2 + 0.3*1 = 1.3."""
        states = [
            {"net_charge": 0, "spin_multiplicity": 1},
            {"net_charge": 1, "spin_multiplicity": 1},
        ]
        result = heuristic_lowest_energy_charge_spin_multiplicity(states)
        assert result == {"net_charge": 0, "spin_multiplicity": 1}

    def test_prefers_singlet_over_triplet_at_same_charge(self):
        """Singlet score 0 beats triplet score (3-1)=2 at charge 0."""
        states = [
            {"net_charge": 0, "spin_multiplicity": 1},
            {"net_charge": 0, "spin_multiplicity": 3},
        ]
        result = heuristic_lowest_energy_charge_spin_multiplicity(states)
        assert result == {"net_charge": 0, "spin_multiplicity": 1}

    def test_slight_preference_for_anion_over_cation(self):
        """1 - 0.3 = 0.7 (anion) beats 1 + 0.3 = 1.3 (cation)."""
        states = [
            {"net_charge": -1, "spin_multiplicity": 1},
            {"net_charge": 1, "spin_multiplicity": 1},
        ]
        result = heuristic_lowest_energy_charge_spin_multiplicity(states)
        assert result == {"net_charge": -1, "spin_multiplicity": 1}

    def test_spin_penalty_can_outweigh_charge_penalty(self):
        """Neutral triplet (score 2.0) beats a q=+2 singlet (score 4.6)."""
        states = [
            {"net_charge": 2, "spin_multiplicity": 1},
            {"net_charge": 0, "spin_multiplicity": 3},
        ]
        result = heuristic_lowest_energy_charge_spin_multiplicity(states)
        assert result == {"net_charge": 0, "spin_multiplicity": 3}

    def test_single_state_returned_as_is(self):
        """With only one candidate state, it is returned unchanged."""
        states = [{"net_charge": 2, "spin_multiplicity": 4}]
        result = heuristic_lowest_energy_charge_spin_multiplicity(states)
        assert result == states[0]


class TestComputeSmilesFromPositionsNoneMolecule:
    """The ``ob_mol is None`` short-circuit in ``_compute_smiles_from_positions``.

    ``ob_mol_from_positions`` never actually returns ``None`` in the current
    implementation, so this branch is only reachable by mocking it.
    """

    @patch("chemreporter.analysis.obabel_utils.ob_mol_from_positions")
    def test_returns_empty_string_when_mol_is_none(self, mock_ob_mol_from_positions):
        """Guards against a future ob_mol_from_positions returning None."""
        mock_ob_mol_from_positions.return_value = None

        result = _compute_smiles_from_positions([6], [[0.0, 0.0, 0.0]], 0)

        assert not result


class TestFixFormalChargesAndRadicalsBranches:
    """Exercise the valence/radical fixup branches inside ``ob_mol_from_positions``.

    These hand-picked, non-geometry-optimized 3D coordinates are deliberately
    imperfect, so that OpenBabel's initial ``ConnectTheDots``/
    ``PerceiveBondOrders`` pass assigns a few atoms non-ideal valences, which
    ``fix_formal_charges_and_radicals`` then has to correct via
    ``fix_pi_bonds`` and/or ``fix_radical_carbons``. The exact geometries and
    charges were found by trial and error against a coverage report; what
    matters for this test is that the fixup machinery actually runs (and
    successfully reconciles the total formal charge with the requested net
    charge), not the specific chemical identity of the input.
    """

    def test_styrene_like_anion_triggers_pi_bond_and_radical_carbon_fixups(self):
        """A styrene-shaped radical anion exercises fix_pi_bonds's bond-order
        swap and fix_radical_carbons's adjacent-radical-pair branch.
        """
        atomic_numbers = [6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]
        positions = [
            [1.39, 0, 0],
            [0.695, 1.203, 0],
            [-0.695, 1.203, 0],
            [-1.39, 0, 0],
            [-0.695, -1.203, 0],
            [0.695, -1.203, 0],
            [2.47, 0, 0],
            [3.7, 0.5, 0],
            [1.235, 2.137, 0],
            [-1.235, 2.137, 0],
            [-1.235, -2.137, 0],
            [0.695, -2.203, 0],
            [2.5, -0.8, 0],
            [3.8, 1.5, 0],
            [4.5, 0, 0],
            [3.6, -0.3, 0],
        ]

        mol = ob_mol_from_positions(atomic_numbers, positions, -1)

        assert mol is not None
        assert mol.GetTotalCharge() == -1

    def test_glycolaldehyde_like_neutral_triggers_carbon_heteroatom_fixup(self):
        """A glycolaldehyde-shaped neutral molecule exercises
        fix_radical_carbons's carbon-radical-2-bonds-from-heteroatom branch.
        """
        atomic_numbers = [6, 6, 8, 8, 1, 1, 1, 1]
        positions = [
            [0, 0, 0],
            [1.5, 0, 0],
            [2.1, 1.2, 0],
            [-0.7, 1.1, 0],
            [-0.5, -0.9, 0],
            [-0.5, 0, -1.0],
            [1.9, -0.9, 0],
            [1.9, 0.9, 0.8],
        ]

        mol = ob_mol_from_positions(atomic_numbers, positions, 0)

        assert mol is not None
        assert mol.GetTotalCharge() == 0
