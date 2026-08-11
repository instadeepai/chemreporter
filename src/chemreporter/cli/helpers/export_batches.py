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
"""Output-file batch planning for cli.export."""

import logging
from typing import Any, List, cast

logger = logging.getLogger("chemreporter")


def _subset_extras(
    extras_fields: dict[str, dict[str, Any]] | None, keys: list[str]
) -> dict[str, dict[str, Any]] | None:
    """Restrict extras_fields to keys.

    Returns:
        A dict restricted to keys, or None if there are no extras.
    """
    if not extras_fields:
        return None
    return {k: extras_fields[k] for k in keys if k in extras_fields}


def build_export_batches(
    worker_kwargs: list[dict],
    num_files_to_export: int,
    is_multiprocess: bool,
    output_path,
    local_output_folder,
    extras_fields: dict[str, dict[str, Any]] | None,
) -> list[tuple[str, list[dict]]]:
    """Split worker_kwargs (one entry per source-file group) across parts.

    Groups in a part are merged into one task for single-process runs, or
    kept separate for multiprocess pool dispatch. Skips parts that already
    exist on disk.

    Returns:
        List of (output_path_name, batch_worker_kwargs) pairs left to process.
    """
    num_groups = len(worker_kwargs)
    num_files = min(num_files_to_export, num_groups)
    batches = []

    for file_index in range(num_files):
        if num_files_to_export > 1:
            start_index = file_index * num_groups // num_files
            end_index = (file_index + 1) * num_groups // num_files
            output_path_name = f"{output_path.stem}_part_{file_index:03d}.hdf5"

            final_dest = output_path.parent / output_path_name
            if final_dest.exists():
                logger.info(
                    f"Skipping part {file_index:03d} - {final_dest} already exists."
                )
                continue

            group_slice = worker_kwargs[start_index:end_index]
            if is_multiprocess:
                batch_worker_kwargs = group_slice
            else:
                merged_keys = [
                    key
                    for group in group_slice
                    for key in cast(List[str], group["keys_list"])
                ]
                batch_worker_kwargs = [
                    {
                        "keys_list": merged_keys,
                        "output_file": str(local_output_folder / output_path_name),
                        "extras_fields": _subset_extras(extras_fields, merged_keys),
                    }
                ]
        else:
            batch_worker_kwargs = worker_kwargs
            output_path_name = output_path.name

        batches.append((output_path_name, batch_worker_kwargs))

    return batches
