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
from typing import Any

import ase
import numpy as np
from ase.data import vdw_radii
from ase.neighborlist import natural_cutoffs
from scipy.spatial.distance import cdist

from chemreporter.analysis.ase_utils import get_atoms_with_tags
from chemreporter.analysis.obabel_utils import (
    get_charge_spin_multiplicity,
    smiles_from_positions,
)
from chemreporter.analysis.structure_check import is_molecular_structure_valid

MATERIAL_CLASS_MAPPING = {
    0: "intermetallics",
    1: "metalloids",
    2: "non-metals",
    3: "halides",
}

DEFAULT_VDW_RADIUS = 1.7  # fallback for elements with no tabulated ASE vdW radius (Å)

MATERIAL_ANOMALY_MAPPING = {
    0: "no_anomaly",
    1: "adsorbate_dissociation",
    2: "adsorbate_desorption",
    3: "surface_reconstruction",
    4: "uninteracting_hydrogen",
}


def get_catalyst_data(atoms: ase.Atoms) -> dict[str, Any]:
    """Get catalyst data from atoms object.

    Args:
        atoms: ASE Atoms object

    Returns:
        Dictionary with catalyst data.
    """
    # OC20 tags: 0=bulk, 1=surface, 2=adsorbate
    # adsorbate_atoms = get_atoms_with_tags(atoms, [1])
    # first_layer_slab_atoms = get_atoms_with_tags(atoms, [2])

    adsorbate_atoms = get_atoms_with_tags(atoms, [2])
    first_layer_slab_atoms = get_atoms_with_tags(atoms, [1])
    slab_atoms = get_atoms_with_tags(atoms, [0, 1])

    ads_cutoffs = np.array(natural_cutoffs(adsorbate_atoms, mult=1.2))
    slab_cutoffs = np.array(natural_cutoffs(first_layer_slab_atoms, mult=1.2))
    distances = cdist(first_layer_slab_atoms.positions, adsorbate_atoms.positions)
    cutoffs = slab_cutoffs[:, np.newaxis] + ads_cutoffs[np.newaxis, :]
    nearby_mask = np.any(distances < cutoffs, axis=1)
    nearby_surface = first_layer_slab_atoms[nearby_mask]
    adsorbate_plus_nearby = adsorbate_atoms + nearby_surface
    is_adsorbate_valid = is_molecular_structure_valid(adsorbate_plus_nearby)
    adsorbate_smiles = ""

    charge_spin_multiplicity = get_charge_spin_multiplicity(
        adsorbate_atoms.get_atomic_numbers().tolist(),
        adsorbate_atoms.positions.tolist(),
    )
    if len(adsorbate_atoms) > 0:
        adsorbate_smiles = smiles_from_positions(
            adsorbate_atoms.get_atomic_numbers().tolist(),
            adsorbate_atoms.positions.tolist(),
            charge_spin_multiplicity["net_charge"],
        )

    substrate_height = 0.0
    if len(first_layer_slab_atoms) > 0 and len(adsorbate_atoms) > 0:
        substrate_height = get_slab_adsorbate_z_distance(
            first_layer_slab_atoms, adsorbate_atoms
        )
    # 'bulk_id', 'ads_id', 'bulk_mpid', 'bulk_symbols', 'ads_symbols', 'miller_index',
    # 'shift', 'top', 'adsorption_site', 'class', 'anomaly', 'split', 'ref_energy',
    # 'num_atoms', 'numbers', 'positions', 'tags' ,'frame_number'
    return {
        "catalyst_bulk_id": int(atoms.info.get("bulk_id", None)),
        "catalyst_adsorbate_id": int(atoms.info.get("ads_id", None)),
        "catalyst_bulk_symbols": atoms.info.get("bulk_symbols", None),
        "catalyst_adsorbate_symbols": atoms.info.get("ads_symbols", None),
        "catalyst_adsorbate_smiles": adsorbate_smiles,
        "catalyst_num_adsorbate_atoms": len(adsorbate_atoms),
        "catalyst_num_bulk_atoms": len(slab_atoms),
        "catalyst_reference_energy": float(atoms.info.get("ref_energy", 0.0)),
        "catalyst_class": MATERIAL_CLASS_MAPPING.get(
            atoms.info.get("class", None), None
        ),
        "catalyst_anomaly": MATERIAL_ANOMALY_MAPPING.get(
            atoms.info.get("anomaly", None), None
        ),
        "catalyst_substrate_height": substrate_height,
        "catalyst_miller_index": atoms.info.get("miller_index", None),
        "catalyst_relaxation_frame_idx": int(
            atoms.info.get("frame_number", "0").replace("frame", "")
        ),
        "is_molecular_structure_valid": is_adsorbate_valid,
        "catalyst_xyz_adsorbate_is_valid": check_xyz_adsorbate_is_valid(
            slab_atoms, adsorbate_atoms
        ),
    }


def _surface_normal(cell: ase.cell.Cell) -> np.ndarray | None:
    """Unit surface normal spanned by the two in-plane lattice vectors.

    Args:
        cell: ASE cell of the slab.

    Returns:
        Unit normal oriented toward the out-of-plane lattice vector, or None
        when the cell is singular (no periodic cell defined).
    """
    vecs = np.asarray(cell)
    normal = np.cross(vecs[0], vecs[1])
    norm = float(np.linalg.norm(normal))
    if norm < 1e-8:
        return None
    normal = normal / norm
    if normal @ vecs[2] < 0:
        normal = -normal
    return normal


def _vdw_radius(atomic_number: int) -> float:
    """Van der Waals radius for an atomic number, with a fallback default.

    Args:
        atomic_number: atomic number.

    Returns:
        float: van der Waals radius in Angstrom.
    """
    radius = vdw_radii[atomic_number]
    return DEFAULT_VDW_RADIUS if np.isnan(radius) else float(radius)


def _max_vdw_radius(atoms: ase.Atoms) -> float:
    """Largest van der Waals radius among an Atoms object's elements.

    Args:
        atoms: ASE Atoms object.

    Returns:
        float: largest van der Waals radius in Angstrom.
    """
    return max(_vdw_radius(number) for number in atoms.numbers)


def check_xyz_adsorbate_is_valid(
    slab_atoms: ase.Atoms, adsorbate_atoms: ase.Atoms
) -> bool:
    """Check the adsorbate sits above the slab within its in-plane footprint.

    Heights are measured along the true surface normal derived from the slab
    lattice vectors, so the check stays correct when the surface layer is not
    orthogonal to the z-axis. The in-plane footprint is compared using the
    minimum-image convention, so an adsorbate near a periodic cell edge is not
    incorrectly rejected. Both checks are padded by van der Waals radii so
    atoms that physically overlap the boundary are not rejected for having a
    center of mass just outside it. When the atoms carry no periodic cell the
    check falls back to the global z-axis and raw xy positions.

    Args:
        slab_atoms: ASE Atoms object for the slab.
        adsorbate_atoms: ASE Atoms object for the adsorbate.

    Returns:
        bool: True if the adsorbate is valid, False otherwise.
    """
    normal = _surface_normal(slab_atoms.cell)
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])
        adsorbate_xy = adsorbate_atoms.positions[:, :2]
        slab_xy = slab_atoms.positions[:, :2]
        xy_tolerance = _max_vdw_radius(adsorbate_atoms)
    else:
        slab_xy = slab_atoms.get_scaled_positions(wrap=True)[:, :2]
        adsorbate_xy_raw = adsorbate_atoms.get_scaled_positions(wrap=False)[:, :2]
        slab_centroid = np.mean(slab_xy, axis=0)
        adsorbate_xy = adsorbate_xy_raw - np.round(adsorbate_xy_raw - slab_centroid)
        cell_lengths_ab = slab_atoms.cell.lengths()[:2]
        xy_tolerance = _max_vdw_radius(adsorbate_atoms) / cell_lengths_ab

    slab_height_max = float(np.max(slab_atoms.positions @ normal))
    adsorbate_height_min = float(np.min(adsorbate_atoms.positions @ normal))
    z_tolerance = _max_vdw_radius(slab_atoms) + _max_vdw_radius(adsorbate_atoms)
    z_ok = adsorbate_height_min > slab_height_max - z_tolerance

    ads_xy_min = np.min(adsorbate_xy, axis=0)
    ads_xy_max = np.max(adsorbate_xy, axis=0)
    slab_xy_min = np.min(slab_xy, axis=0)
    slab_xy_max = np.max(slab_xy, axis=0)
    xy_ok = np.all(ads_xy_min >= slab_xy_min - xy_tolerance) and np.all(
        ads_xy_max <= slab_xy_max + xy_tolerance
    )
    return bool(z_ok and xy_ok)


def get_slab_adsorbate_z_distance(
    first_layer_slab_atoms: ase.Atoms,
    adsorbate_atoms: ase.Atoms,
) -> float:
    """Distance along the surface normal between slab surface and adsorbate.

    Heights are measured along the true surface normal derived from the slab
    lattice vectors, so the distance stays correct when the surface layer is
    not orthogonal to the z-axis. When the atoms carry no periodic cell the
    check falls back to the global z-axis.

    Args:
        first_layer_slab_atoms: ASE Atoms object for the top layer of the slab.
        adsorbate_atoms: ASE Atoms object for the adsorbate.

    Returns:
        float: distance from mean slab height to adsorbate center of mass
            height, both measured along the surface normal.
    """
    normal = _surface_normal(first_layer_slab_atoms.cell)
    if normal is None:
        normal = np.array([0.0, 0.0, 1.0])

    slab_height_mean = float(np.mean(first_layer_slab_atoms.positions @ normal))
    adsorbate_height_com = float(adsorbate_atoms.get_center_of_mass() @ normal)

    return adsorbate_height_com - slab_height_mean
