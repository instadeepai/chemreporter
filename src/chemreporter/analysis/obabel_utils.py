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
import atexit
import functools
import multiprocessing
from collections import deque
from multiprocessing import pool

import numpy as np
from openbabel import openbabel
from rdkit import Chem

# Max and min charge to consider for the charge and spin multiplicity
MAX_CHARGE = 3
MIN_CHARGE = -3


@functools.cache
def _periodic_table():
    return Chem.GetPeriodicTable()


OB_MOLECULES_PER_WORKER = 5000


@functools.lru_cache(maxsize=1)
def _ob_worker_pool() -> pool.Pool:
    """Build the recycled worker pool used for OpenBabel perception calls.

    Uses the "fork" context deliberately: the worker inherits already-imported
    modules, so recycling costs ~3 ms instead of a full interpreter start, and
    callers do not need an ``if __name__ == "__main__"`` guard the way "spawn"
    would require. The worker only calls OpenBabel, so it never touches the
    thread pools that make forking a threaded parent unsafe.

    Returns:
        Single-worker pool whose worker process is replaced every
        ``OB_MOLECULES_PER_WORKER`` molecules, bounding the ConnectTheDots leak.
    """
    worker_pool = multiprocessing.get_context("fork").Pool(
        processes=1, maxtasksperchild=OB_MOLECULES_PER_WORKER or None
    )
    atexit.register(worker_pool.terminate)
    return worker_pool


@functools.lru_cache(maxsize=1)
def _ob_conversion() -> openbabel.OBConversion:
    """Build one reusable OBConversion for this process.

    Shared process-wide singleton: do not retain the returned object across
    calls or mutate its formats, and never use it concurrently from threads.

    Returns:
        OBConversion reused across SMILES conversions.
    """
    openbabel.obErrorLog.SetOutputLevel(0)
    conversion = openbabel.OBConversion()
    conversion.SetInAndOutFormats("xyz", "smi")
    return conversion


@functools.lru_cache(maxsize=1)
def _ob_mol() -> openbabel.OBMol:
    """Build one reusable OBMol for this process.

    Shared process-wide singleton reused to avoid per-molecule allocation
    churn: the value is overwritten on the next call, so callers must Clear()
    and fully consume it before the next _ob_mol() use and never retain it.

    Returns:
        OBMol reused across conversions; callers Clear() before filling.
    """
    return openbabel.OBMol()


def expected_valence(z):
    """Get the expected valence of the atom.

    Transition metals have no single well-defined valence (RDKit's own
    periodic table reports -1, meaning "any valence allowed"), so they
    are skipped rather than assigned a guessed value.

    Args:
        z: atomic number

    Returns:
        expected valence of the atom, or None if unknown or metal
    """
    default_valence = _periodic_table().GetDefaultValence(z)
    return default_valence if default_valence >= 0 else None


def atoms_within_n_bonds(ob_mol, start_idx, n=3):
    """Get the atoms within n bonds of the start atom.

    Args:
        ob_mol: openbabel molecule
        start_idx: index of the start atom
        n: number of bonds

    Returns:
        list of tuples with the index and path of the atoms within n bonds
    """
    visited = {start_idx}
    q = deque([(start_idx, 0, [start_idx])])

    results = []

    while q:
        idx, dist, path = q.popleft()

        atom = ob_mol.GetAtom(idx)

        for nbr in openbabel.OBAtomAtomIter(atom):
            nidx = nbr.GetIdx()

            new_path = path + [nidx]
            new_dist = dist + 1

            visited.add(nidx)

            if new_dist <= n:
                if len(new_path) == n:
                    results.append((nidx, new_path))
                q.append((nidx, new_dist, new_path))

    return results


def diff_expected_valence(atom_nb) -> int:
    """Check if the atom has the expected valence.

    Args:
        atom_nb: openbabel atom

    Returns:
        difference in valence of the atom
    """
    z = atom_nb.GetAtomicNum()
    expected = expected_valence(z)

    if expected is None:
        return 0
    valence = atom_nb.GetTotalValence()
    diff_nb = valence - expected

    return diff_nb


def has_expected_valence(atom_nb) -> bool:
    """Check if the atom has the expected valence.

    Args:
        atom_nb: openbabel atom

    Returns:
        True if the atom has the expected valence, False otherwise
    """
    return diff_expected_valence(atom_nb) == 0


def fix_pi_bonds(ob_mol, atom) -> tuple[openbabel.OBMol, bool]:
    """Fix the pi bonds of the molecule.

    openbabel often assigns C1.-C2=C3-C4. (2 radicals and one double bond)
    instead of the conjugated form : C1=C2-C3=C4

    Args:
        ob_mol: openbabel molecule
        atom: openbabel atom

    Returns:
        openbabel molecule with fixed pi bonds

    """
    nb = atoms_within_n_bonds(ob_mol, atom.GetIdx(), 4)

    for n_idx, path in nb:
        atom_nb = ob_mol.GetAtom(n_idx)
        if has_expected_valence(atom_nb) is False:
            bonds = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            if ob_mol.GetBond(*bonds[1]).GetBondOrder() == 2:
                ob_mol.GetBond(*bonds[1]).SetBondOrder(1)
                ob_mol.GetBond(*bonds[2]).SetBondOrder(2)
                ob_mol.GetBond(*bonds[0]).SetBondOrder(2)
                return ob_mol, True
    return ob_mol, False


def fix_radical_carbons(ob_mol, atom):
    """Fix the radical carbons of the molecule.

    Some outputs have radicals on carv=bon and bonded
    to a more electronegative atom ( N, O, P, S). In this case,
    we need to fix the radical carbons by adding a double bond

    Args:
        ob_mol: openbabel molecule
        atom: openbabel atom

    Returns:
        openbabel molecule with fixed radical carbons
    """
    for bond in openbabel.OBMolBondIter(ob_mol):
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        if a1.GetIdx() == atom.GetIdx() or a2.GetIdx() == atom.GetIdx():
            continue
        if diff_expected_valence(a1) < 0 and diff_expected_valence(a2) < 0:
            if a1.GetAtomicNum() in [6, 7, 8] and a2.GetAtomicNum() in [6, 7, 8]:
                bond.SetBondOrder(bond.GetBondOrder() + 1)
                return ob_mol, True

    z = atom.GetAtomicNum()
    if z == 6:
        nb = atoms_within_n_bonds(ob_mol, atom.GetIdx(), 2)
        for n_idx, path in nb:
            atom_nb = ob_mol.GetAtom(n_idx)
            z = atom_nb.GetAtomicNum()

            if z in [7, 8, 15, 16]:
                bond_order = ob_mol.GetBond(path[0], path[1]).GetBondOrder()
                ob_mol.GetBond(path[0], path[1]).SetBondOrder(bond_order + 1)
                atom_charge = atom_nb.GetFormalCharge()
                atom_nb.SetFormalCharge(atom_charge + 1)
                return ob_mol, True
    return ob_mol, False


def fix_formal_charges_and_radicals(ob_mol, net_charge) -> tuple[openbabel.OBMol]:
    """Fix the formal charges of the molecule.

    Args:
        ob_mol: openbabel molecule
        net_charge: net charge of the molecule

    Returns:
        openbabel molecule with fixed formal charges
        or original molecule if failed to fix charges
    """
    ob_mol_original = ob_mol
    total_charge = 0

    for atom in openbabel.OBMolAtomIter(ob_mol):
        diff = diff_expected_valence(atom)
        if diff in [1, -1]:
            if abs(total_charge - net_charge) > 0:
                atom.SetFormalCharge(atom.GetFormalCharge() + diff)
                total_charge += diff
            else:
                ob_mol, fixed = fix_pi_bonds(ob_mol, atom)
                if fixed:
                    # no charges to fix, don't fix radicals
                    continue
                ob_mol, fixed = fix_radical_carbons(ob_mol, atom)
                if fixed:
                    # need to add charge to the molecule since
                    # the fix_radical_carbons add a formal charge to the atom
                    total_charge += 1

    matched_charges = bool(total_charge == net_charge)
    if matched_charges:
        return ob_mol

    else:
        return ob_mol_original


def ob_mol_from_positions(
    atomic_numbers: list[int], positions: list[list[float]], net_charge: int
) -> openbabel.OBMol | None:
    """Fill the reusable openbabel molecule from positions and atomic numbers.

    Args:
        atomic_numbers: list of atomic numbers
        positions: list of positions
        net_charge: net charge of the molecule

    Returns:
        Reused openbabel molecule, or None if creation fails
    """
    formal_charges = np.zeros(len(atomic_numbers))
    ob_mol = _ob_mol()
    ob_mol.Clear()
    for atomic_number, position, formal_charge in zip(
        atomic_numbers, positions, formal_charges
    ):
        atom = ob_mol.NewAtom()
        atom.SetAtomicNum(int(atomic_number))
        atom.SetVector(*position)
        atom.SetFormalCharge(int(formal_charge))

    ob_mol.SetTotalCharge(net_charge)
    ob_mol.ConnectTheDots()
    ob_mol.PerceiveBondOrders()

    return fix_formal_charges_and_radicals(ob_mol, net_charge)


def _compute_smiles_from_positions(
    atomic_numbers: list[int], positions: list[list[float]], net_charge: int
) -> str:
    """Convert positions and atomic numbers to SMILES with openbabel.

    Runs inside the recycled worker process; call ``smiles_from_positions``
    instead.

    Args:
        atomic_numbers: list of atomic numbers
        positions: list of positions
        net_charge: net charge of the molecule

    Returns:
        SMILES string
        if conversion fails, returns empty string ""
    """
    ob_conversion = _ob_conversion()
    ob_mol = ob_mol_from_positions(atomic_numbers, positions, net_charge)
    if ob_mol is None:
        return ""
    return ob_conversion.WriteString(ob_mol).replace("\t\n", "")


def smiles_from_positions(
    atomic_numbers: list[int], positions: list[list[float]], net_charge: int
) -> str:
    """Convert positions and atomic numbers to SMILES with openbabel.

    Delegates to a recycled worker process to bound an OpenBabel memory leak;
    see ``OB_MOLECULES_PER_WORKER``.

    Args:
        atomic_numbers: list of atomic numbers
        positions: list of positions
        net_charge: net charge of the molecule

    Returns:
        SMILES string
        if conversion fails, returns empty string ""
    """
    return _ob_worker_pool().apply(
        _compute_smiles_from_positions, (atomic_numbers, positions, net_charge)
    )


def _compute_charge_spin_multiplicity(
    atomic_numbers: list[int],
    positions: list[list[float]],
    min_charge: int = MIN_CHARGE,
    max_charge: int = MAX_CHARGE,
) -> dict[str, int]:
    """Get the charge and spin multiplicity of a molecule.

    Runs inside the recycled worker process; call
    ``get_charge_spin_multiplicity`` instead.

    Args:
        atomic_numbers: list of atomic numbers
        positions: list of positions
        min_charge: minimum charge to evaluate
        max_charge: maximum charge to evaluate

    Returns:
        dictionary with the charge and spin multiplicity
    """
    possible_states = []
    for net_charge in range(min_charge, max_charge + 1):
        obmol = ob_mol_from_positions(atomic_numbers, positions, net_charge)
        if obmol is None:
            continue
        if obmol.GetTotalCharge() == net_charge:
            multiplicity = obmol.GetTotalSpinMultiplicity()
            possible_states.append({
                "net_charge": net_charge,
                "spin_multiplicity": multiplicity,
            })
    if not possible_states:
        total_electrons = sum(atomic_numbers)
        return {"net_charge": 0, "spin_multiplicity": 1 + (total_electrons % 2)}

    return heuristic_lowest_energy_charge_spin_multiplicity(possible_states)


def get_charge_spin_multiplicity(
    atomic_numbers: list[int], positions: list[list[float]]
) -> dict[str, int]:
    """Get the charge and spin multiplicity of a molecule.

    Delegates to a recycled worker process to bound an OpenBabel memory leak;
    see ``OB_MOLECULES_PER_WORKER``.

    Args:
        atomic_numbers: list of atomic numbers
        positions: list of positions
        min_charge: minimum charge to evaluate
        max_charge: maximum charge to evaluate

    Returns:
        dictionary with the charge and spin multiplicity
    """
    return _ob_worker_pool().apply(
        _compute_charge_spin_multiplicity,
        (atomic_numbers, positions),
    )


def heuristic_lowest_energy_charge_spin_multiplicity(states):
    """Heuristic to find the lowest energy charge and spin multiplicity.

    This encodes 3 physical assumptions:

    1. Neutrality preference (but not absolute) via q
    2. Slight preference for anions over cations (common in organics)
    via +beta * q
    3. Strong closed-shell preference
    via spin penalty (S) via gamma * S


    Args:
        states: list of states with net charge and spin multiplicity

    Returns:
        dictionary with the lowest energy charge and spin multiplicity
    """
    alpha = 1.0
    beta = 0.3
    gamma = 2.0

    def score(x):
        q = x["net_charge"]
        m = x["spin_multiplicity"]

        s = (m - 1) / 2

        e_charge = alpha * (q**2) + beta * q
        e_spin = gamma * s

        return e_charge + e_spin

    return min(states, key=score)
