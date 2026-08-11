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
import ase
import pytest

from chemreporter.source_database_tools.open_catalyst_20.catalyst_data_parser import (
    check_xyz_adsorbate_is_valid,
    get_slab_adsorbate_z_distance,
)


def _slab_and_adsorbate(cell, slab_positions, adsorbate_positions):
    slab = ase.Atoms(
        "Cu" * len(slab_positions), positions=slab_positions, cell=cell, pbc=True
    )
    adsorbate = ase.Atoms(
        "O" * len(adsorbate_positions),
        positions=adsorbate_positions,
        cell=cell,
        pbc=True,
    )
    return slab, adsorbate


def test_orthogonal_cell_adsorbate_above_is_valid():
    """Adsorbate above the slab in an orthogonal cell is valid."""
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]]
    slab, adsorbate = _slab_and_adsorbate(
        cell,
        [[1.0, 1.0, 5.0], [9.0, 9.0, 5.0]],
        [[5.0, 5.0, 8.0]],
    )
    assert check_xyz_adsorbate_is_valid(slab, adsorbate) is True


def test_adsorbate_below_surface_is_invalid():
    """Adsorbate below the slab surface is invalid."""
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]]
    slab, adsorbate = _slab_and_adsorbate(
        cell,
        [[1.0, 1.0, 5.0], [9.0, 9.0, 5.0]],
        [[5.0, 5.0, 2.0]],
    )
    assert check_xyz_adsorbate_is_valid(slab, adsorbate) is False


def test_tilted_cell_above_surface_normal_but_below_in_global_z():
    """Adsorbate above the tilted surface normal stays valid despite lower z."""
    # Cell whose in-plane vector a has a z-component, so the surface normal is
    # tilted away from global z. Positions are built from fractional coords:
    # cart = [10*fa, 10*fb, 6*fa + 10*fc]. Height along the true normal depends
    # only on fc, while global z also grows with fa.
    cell = [[10.0, 0.0, 6.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    slab_positions = [
        [1.0, 1.0, 3.6],  # fa=0.1, fc=0.3
        [9.0, 1.0, 8.4],  # fa=0.9, fc=0.3 -> highest global z
        [1.0, 9.0, 3.6],
        [9.0, 9.0, 8.4],
    ]
    adsorbate_positions = [[1.0, 5.0, 5.6]]  # fa=0.1, fc=0.5
    slab, adsorbate = _slab_and_adsorbate(cell, slab_positions, adsorbate_positions)
    # Adsorbate is above the surface along the normal (fc=0.5 > 0.3) yet sits
    # below the highest slab atom in global z.
    assert adsorbate_positions[0][2] < max(p[2] for p in slab_positions)
    assert check_xyz_adsorbate_is_valid(slab, adsorbate) is True


def test_no_cell_falls_back_to_global_z():
    """Without a periodic cell the check uses raw global-z positions."""
    slab = ase.Atoms("Cu", positions=[[5.0, 5.0, 5.0]])
    adsorbate_above = ase.Atoms("O", positions=[[5.0, 5.0, 8.0]])
    assert check_xyz_adsorbate_is_valid(slab, adsorbate_above) is True

    adsorbate_below = ase.Atoms("O", positions=[[5.0, 5.0, 2.0]])
    assert check_xyz_adsorbate_is_valid(slab, adsorbate_below) is False


def test_xy_containment_wraps_across_periodic_boundary():
    """An adsorbate near one cell edge is contained by a slab near the other."""
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]]
    # Slab sits right at the upper x edge (frac 0.95-0.99); the adsorbate's
    # raw fractional x is 1.97 (position 19.7, i.e. stored past the cell
    # boundary) which is physically adjacent to the slab across the periodic
    # image. Without minimum-image wrapping this would be rejected as being
    # ~1 full cell-length outside the slab's footprint.
    slab = ase.Atoms(
        "Cu2", positions=[[9.5, 5.0, 5.0], [9.9, 5.0, 5.0]], cell=cell, pbc=True
    )
    adsorbate = ase.Atoms("O", positions=[[19.7, 5.0, 10.0]], cell=cell, pbc=True)
    assert check_xyz_adsorbate_is_valid(slab, adsorbate) is True


def test_xy_containment_still_rejects_atoms_that_are_genuinely_outside():
    """Wrapping does not make an unrelated, distant adsorbate valid."""
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]]
    slab = ase.Atoms(
        "Cu2", positions=[[9.5, 5.0, 5.0], [9.9, 5.0, 5.0]], cell=cell, pbc=True
    )
    adsorbate = ase.Atoms("O", positions=[[5.0, 5.0, 10.0]], cell=cell, pbc=True)
    assert check_xyz_adsorbate_is_valid(slab, adsorbate) is False


def test_z_tolerance_allows_small_vdw_overlap():
    """A slight below-surface offset within vdW radii is still valid."""
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]]
    # Cu vdW radius 1.4 + O vdW radius 1.52 = 2.92 tolerance. An adsorbate
    # 1.0 Angstrom below the slab's surface height sits well within that
    # padding, even though its center of mass is technically lower than the
    # slab's highest atom.
    slab = ase.Atoms("Cu", positions=[[5.0, 5.0, 5.0]], cell=cell, pbc=True)
    adsorbate = ase.Atoms("O", positions=[[5.0, 5.0, 4.0]], cell=cell, pbc=True)
    assert check_xyz_adsorbate_is_valid(slab, adsorbate) is True


def test_z_tolerance_still_rejects_large_overlap():
    """An offset beyond the combined vdW radii is still invalid."""
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]]
    slab = ase.Atoms("Cu", positions=[[5.0, 5.0, 5.0]], cell=cell, pbc=True)
    adsorbate = ase.Atoms("O", positions=[[5.0, 5.0, 1.0]], cell=cell, pbc=True)
    assert check_xyz_adsorbate_is_valid(slab, adsorbate) is False


def test_z_distance_orthogonal_cell():
    """Distance along an orthogonal normal matches plain z-distance."""
    cell = [[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 30.0]]
    slab = ase.Atoms("Cu2", positions=[[1.0, 1.0, 4.0], [9.0, 9.0, 6.0]], cell=cell)
    adsorbate = ase.Atoms("O", positions=[[5.0, 5.0, 9.0]], cell=cell)
    assert get_slab_adsorbate_z_distance(slab, adsorbate) == 4.0


def test_z_distance_tilted_cell_uses_surface_normal():
    """Distance is measured along the true surface normal for a tilted cell."""
    # Same tilted cell as the containment test above: surface normal is not
    # global z, so the raw z-distance would differ from the normal-projected
    # distance whenever fa != 0.
    cell = [[10.0, 0.0, 6.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]
    slab = ase.Atoms(
        "Cu2",
        positions=[[1.0, 1.0, 3.6], [9.0, 1.0, 8.4]],  # fa=0.1/0.9, fc=0.3 both
        cell=cell,
    )
    adsorbate = ase.Atoms("O", positions=[[1.0, 5.0, 5.6]], cell=cell)  # fa=0.1, fc=0.5
    # Both slab atoms share fc=0.3, so the true normal distance is
    # (0.5 - 0.3) * (c @ normal) ~= 1.715, not the raw z difference.
    distance = get_slab_adsorbate_z_distance(slab, adsorbate)
    assert distance == pytest.approx(1.7149858514250877)
    raw_z_distance = 5.6 - (3.6 + 8.4) / 2
    assert distance != pytest.approx(raw_z_distance)


def test_z_distance_no_cell_falls_back_to_global_z():
    """Without a periodic cell the distance uses raw global-z positions."""
    slab = ase.Atoms("Cu", positions=[[5.0, 5.0, 4.0]])
    adsorbate = ase.Atoms("O", positions=[[5.0, 5.0, 9.0]])
    assert get_slab_adsorbate_z_distance(slab, adsorbate) == 5.0
