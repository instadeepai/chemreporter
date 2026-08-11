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
"""Shared OC20 source-database helpers (xyz and aselmdb implementations)."""

from __future__ import annotations

from typing import Any

from chemreporter.source_database_tools.database_item import DatasetItem
from chemreporter.source_database_tools.open_catalyst_20.catalyst_data_parser import (
    get_catalyst_data,
)


def get_oc20_source_additional_fields(dataset_item: DatasetItem) -> dict[str, Any]:
    """Charge/spin, composition, and catalyst fields for OC20 source DBs.

    Returns:
        Dict merged from charge/spin, composition, and ``get_catalyst_data``.
    """
    atoms = dataset_item.atoms
    data: dict[str, Any] = {}
    # OC20 systems are periodic slab+adsorbate supercells computed under 3D
    # periodic boundary conditions, which requires overall charge neutrality.
    # The molecular charge-search heuristic is meaningless here, so charge is
    # fixed to 0 and spin is taken from the electron-count parity floor (the
    # true magnetic moment is only known to the source DFT calculation).
    total_electrons = int(atoms.get_atomic_numbers().sum())
    data.update({
        "net_charge": 0,
        "spin_multiplicity": 1 + (total_electrons % 2),
    })
    data.update({"composition": atoms.get_chemical_formula()})
    data.update(get_catalyst_data(atoms))
    return data
