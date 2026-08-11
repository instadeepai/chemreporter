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
from dataclasses import dataclass
from typing import Any, Callable

import ase


@dataclass
class DatasetItem:
    """One structure entry read from a source database split."""

    # Name of the database: e.g. "omol25-1"
    database_name: str
    # Name of the split: e.g. "train"
    split_name: str
    # Key in ChemReporter key format: e.g. "omol25_train_data0001_1"
    key: str
    atoms: ase.Atoms
    # Mapping of field names: e.g. {"subset": "data_id", "net_charge": "charge"}
    name_mapping: dict[str, str]
    # List of functions to compute additional fields
    additional_fields: list[Callable[["DatasetItem"], dict[str, Any]]]
