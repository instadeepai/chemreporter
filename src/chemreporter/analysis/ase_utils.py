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
"""Module to calculate properties using ASE objects."""

from typing import Any

import ase
import numpy as np
from ase.units import Angstrom, Debye, eV


def get_molecular_weight(atoms: ase.Atoms) -> float:
    """Get the molecular weight of an ASE atoms object.

    Args:
        atoms: ASE Atoms object

    Returns:
        Molecular weight of the system in atomic mass units
    """
    total_mass = atoms.get_masses().sum()
    return float(total_mass)


def get_forces(atoms: ase.Atoms) -> np.ndarray:
    """Get the forces of an ASE atoms object.

    Args:
        atoms: ASE Atoms object

    Returns:
        Forces of the atoms object
    """
    return np.array(atoms._calc.results["forces"])


def get_net_forces_norm(atoms: ase.Atoms) -> np.ndarray:
    """Get the net force of an ASE atoms object.

    Args:
        atoms: ASE Atoms object

    Returns:
        Sum of the forces of the atoms object
    """
    forces = get_forces(atoms)
    total_force = forces.sum(axis=0)
    return np.linalg.norm(total_force)


def get_max_forces_norm(atoms: ase.Atoms) -> np.ndarray:
    """Get the max of the norm of the forces of an ASE
        atoms object.

    Args:
        atoms: ASE Atoms object

    Returns:
        Max of the norm of the forces of the atoms object
    """
    forces = get_forces(atoms)
    return np.linalg.norm(forces, axis=1).max()


def calculate_dipole_moment(
    atoms: ase.Atoms, charges_type: str = "mulliken_charges"
) -> float | None:
    """Calculate the dipole moment of an ASE atoms object.

    Per default this uses the mulliken charges from the atoms object
    Args:
        atoms: ASE Atoms object
        charges_type: Type of charges to use, default is "mulliken_charges"
              this argument must match a key in the atoms.info dictionary

    Returns:
        Dipole moment magnitude in Debye units

    Raises:
        ValueError: If the charges of type are not found in the atoms object
    """
    dipole = np.zeros(3)
    if charges_type not in atoms.info:
        return None
    charges = atoms.info[charges_type]
    for pos, charge in zip(atoms.get_positions(), charges):
        dipole += charge * pos

    # Compute the net charge
    net_charge = sum(charges)

    # Apply the origin correction
    dipole_corrected = dipole - net_charge * atoms.get_center_of_mass()

    # Convert to Debye units
    dipole_magnitude = np.linalg.norm(dipole_corrected) * (eV * Angstrom) / Debye

    return dipole_magnitude


def process_ase_info(
    atoms: ase.Atoms, fields_name_mapping: dict[str, str]
) -> dict[str, Any]:
    """Extract information and calculated properties from ASE atoms object.

    Args:
        atoms: ASE Atoms object
        fields_name_mapping: Mapping of the fields to extract from the atoms object

    Returns:
        Dictionary with extracted information and calculated properties
        properties:
            positions: Positions of the atoms in Angstrom
            atom_numbers_int: Atomic numbers of the atoms
            data_id: subset of the atoms object
            charge: Charge of the atoms object
            num_atoms: Number of atoms in the atoms object
            dipole_moment_magnitude: Dipole moment magnitude in Debye units
            net_force_norm: Sum of the norm of the forces in eV/Angstrom
            max_force_norm: Max of the norm of the forces in eV/Angstrom

    Raises:
        KeyError: If the fields format is not found or if the field,
        is not found in the atoms object.
        or if the fields format is not correct.
    """
    out: dict[str, Any] = {}
    # Extract readily available information with the name mapping
    for key, value in fields_name_mapping.items():
        if value in atoms.info:
            out[key] = atoms.info[value]
        elif value in atoms.arrays:
            out[key] = atoms.arrays[value].tolist()
        else:
            raise KeyError(
                f"Field {value} not found in atoms object (info or arrays)"
                f"fields_name_mapping should be one of \n"
                f"{list(atoms.info.keys()) + list(atoms.arrays.keys())} \n"
                "hint : check your name mapping is correct "
            )
    # Add additional infromations from the atoms object
    out["energy"] = atoms._calc.results.get("energy", None)
    out["positions"] = np.array(atoms.positions).tolist()
    out["atomic_numbers"] = atoms.get_atomic_numbers().tolist()

    return out


def get_unique_chemical_symbols(atoms: ase.Atoms) -> str:
    """Get the chemical symbols of the atoms in the atoms object.

    Args:
        atoms: ASE Atoms object

    Returns:
        Chemical symbols of the atoms in the atoms object
    """
    return "".join(sorted(set(atoms.get_chemical_symbols())))


def get_atoms_with_tags(atoms: ase.Atoms, tags: int | list[int]) -> ase.Atoms:
    """Get atoms whose tag is in the given tag(s).

    Args:
        atoms: ASE Atoms object
        tags: Single tag or list of tags to select (e. g. 2 for abs in OC20)

    Returns:
        ASE Atoms object with only atoms with the given tag(s).
    """
    if isinstance(tags, int):
        tags = [tags]
    if "tags" not in atoms.arrays:
        return atoms[[]]
    mask = np.isin(atoms.arrays["tags"], np.asarray(tags))
    return atoms[mask]
