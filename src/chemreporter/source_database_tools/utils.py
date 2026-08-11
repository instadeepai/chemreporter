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

from chemreporter.source_database_tools.database_reader import (
    SourceDatabaseReader,
    parse_key,
)


def retrieve_source_database_data_from_keys(
    source_db_readers: list[SourceDatabaseReader],
    key_entries: list[str],
    data_field: str = "source",
) -> dict[str, Any]:
    """Retrieve raw atom info for key entries from source database readers.

    The ``source`` field is stored in the source database but omitted from the
    query database.

    Args:
        source_db_readers: Source database readers to search.
        key_entries: Keys to retrieve (for example ``omol25_train_MR_693393_...``).
        data_field: Key in ``atoms.info`` to return (default ``"source"``).

    Returns:
        Mapping from each key to the requested field value, for example
        ``{"omol25_train_MR_693393_1_13_0_1_0":
        "omol25/train/MR_693393_1_13_0_1_0.aselmdb"}``.

    Raises:
        RuntimeError: If a key is not found in any source database reader.
    """
    processed_keys = set()
    key_entries_set = set(key_entries)
    output_data = {}
    for key_entry in key_entries_set:
        if key_entry in processed_keys:
            continue
        # only added line compared to write_hdf5_from_indexes
        key_entry_parsed = parse_key(key_entry)

        for source_db_reader in source_db_readers:
            if (
                source_db_reader.database_name == key_entry_parsed.database_name
                and source_db_reader.split_name == key_entry_parsed.split_name
            ):
                systems, extracted_keys = source_db_reader.fetch_atoms(
                    key_from_file=key_entry,
                    key_entries=key_entries_set,
                )
                processed_keys.update(extracted_keys)
                for atoms, system_key in zip(systems, extracted_keys):
                    output_data[system_key] = atoms.info[data_field]
                break  # Stop after first match
        else:
            raise RuntimeError(
                f"Key {key_entry} not found in any source database reader\n"
                f"key_entry: {key_entry_parsed.database_name} -"
                f"{key_entry_parsed.split_name}\n"
                f"source_db_readers: {source_db_readers}\n"
            )

    return output_data
