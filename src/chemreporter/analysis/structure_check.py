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
import logging

import numpy as np
from ase import Atoms
from ase.neighborlist import natural_cutoffs
from matscipy.neighbours import neighbour_list

logger = logging.getLogger("chemreporter")

DEFAULT_MINIMUM_SUBGRAPH_SIZE = 2
OH_BOND_CUTOFF = 1.2
PADDING_SIZE = 10.0


def _add_padding(atoms: Atoms) -> Atoms:
    """Add padding to the atoms object.

    matscipy neighbour_list needs a cell to be defined

    Args:
        atoms: ASE Atoms object

    Returns:
        ASE Atoms object with padding
    """
    if np.any(atoms.cell == 0) or np.any(atoms.cell is None):
        positions = atoms.positions
        min_pos = np.min(positions, axis=0)
        max_pos = np.max(positions, axis=0)
        cell_size = np.max(max_pos - min_pos) + PADDING_SIZE
        atoms.cell = [cell_size, cell_size, cell_size]
    return atoms


def is_molecular_structure_valid(
    atoms: Atoms,
    min_subgraph_size: int = DEFAULT_MINIMUM_SUBGRAPH_SIZE,
) -> bool:
    """Check if a molecular structure is topologically reasonable using
    ASE's natural cutoffs for bond estimation and matscipy's neighbour_list
    for fast connectivity determination.

    Args:
        atoms (ase.Atoms): Molecular structure to evaluate.
        min_subgraph_size (int): Minimum allowed size for any molecular fragment.

    Returns:
        bool: True if the molecule appears chemically reasonable, False otherwise.
    """
    try:
        n_atoms = len(atoms)
        if n_atoms < min_subgraph_size:
            return False

        # Create a copy to avoid modifying the original
        atoms_copy = atoms.copy()

        atoms_copy = _add_padding(atoms_copy)

        cutoffs = natural_cutoffs(atoms_copy, mult=1.2)

        i_list, j_list = neighbour_list("ij", atoms_copy, cutoffs)

        graph: list[set[int]] = [set() for _ in range(n_atoms)]
        for i, j in zip(i_list, j_list):
            graph[i].add(j)
            graph[j].add(i)

        visited = set()
        components = []
        for start in range(n_atoms):
            if start not in visited:
                stack = [start]
                comp = set()
                while stack:
                    node = stack.pop()
                    if node not in visited:
                        visited.add(node)
                        comp.add(node)
                        stack.extend(graph[node] - visited)
                components.append(comp)

        for comp in components:
            if len(comp) < min_subgraph_size:
                logger.debug("Fragment too small: size=%s", len(comp))
                return False

        for i, atom_number in enumerate(atoms.numbers):
            if atom_number == 1:  # Hydrogen
                n_neighbors = len(graph[i])
                if n_neighbors != 1:
                    logger.debug("Hydrogen %s has %s neighbors", i, n_neighbors)
                    return False

        return True

    except Exception as e:
        # Real-world geometries can be pathological in many ways (degenerate
        # cells, overlapping atoms, ...), so this stays broad by design: any
        # failure to compute connectivity means the structure isn't usable.
        logger.warning("Error during molecule validation: %s", e)
        return False


def count_water_molecules(atoms: Atoms) -> int:
    """Count the number of water molecules in a molecular structure.

    Steps:
    1. Filter to only H and O atoms
    2. Use matscipy's neighbour_list to get the neighbors of the O atoms
    3. Count the number of H atoms around each O atom using a cutoff

    Args:
        atoms: ASE Atoms object

    Returns:
        Number of water molecules
    """
    symbols = np.array(atoms.get_chemical_symbols())

    hydro_oxygen_mask = (symbols == "H") | (symbols == "O")
    hydro_oxygen_indices = np.where(hydro_oxygen_mask)[0]

    if len(hydro_oxygen_indices) == 0:
        return 0

    hydro_oxygen_atoms = atoms[hydro_oxygen_mask].copy()

    hydro_oxygen_atoms = _add_padding(hydro_oxygen_atoms)

    i, j = neighbour_list("ij", hydro_oxygen_atoms, OH_BOND_CUTOFF)

    symbols_hydro_oxygen = np.array(hydro_oxygen_atoms.get_chemical_symbols())

    num_water_molecules = 0
    for o_index in np.where(symbols_hydro_oxygen == "O")[0]:
        neighbors = j[i == o_index]
        num_h = np.sum(symbols_hydro_oxygen[neighbors] == "H")

        if num_h == 2:
            num_water_molecules += 1

    return num_water_molecules
