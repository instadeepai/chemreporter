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
"""Multiprocessing plumbing for writing HDF5 groups (used by cli.export)."""

import gc
import logging
from typing import Any, List

import h5py

from chemreporter.database_processors.hdf5_writer import write_hdf5
from chemreporter.source_database_tools.database_reader import SourceDatabaseReader

logger = logging.getLogger("chemreporter")


class _HDF5WorkerState:
    source_database_readers: List[SourceDatabaseReader] | None = None


def _initialize_hdf5_worker(
    source_database_readers: List[SourceDatabaseReader],
) -> None:
    """Initialize state reused by every task handled by a worker process."""
    _HDF5WorkerState.source_database_readers = source_database_readers


def _write_hdf5_multiprocess_worker(kwargs: dict) -> None:
    """Write one HDF5 group using state initialized for this worker process.

    Raises:
        RuntimeError: If the worker process was not initialized.
    """
    if _HDF5WorkerState.source_database_readers is None:
        raise RuntimeError("HDF5 worker process was not initialized")

    try:
        write_hdf5_worker(
            kwargs,
            source_database_readers=_HDF5WorkerState.source_database_readers,
            extras_fields=kwargs.get("extras_fields"),
        )
    except Exception as e:
        logger.error("Worker failed on %s: %s", kwargs.get("output_file"), e)
        raise
    finally:
        gc.collect()


def write_hdf5_worker(
    kwargs: dict,
    source_database_readers: List[SourceDatabaseReader],
    extras_fields: dict[str, dict[str, Any]] | None = None,
):
    """Process a group of keys and write them to a HDF5 file.

    Args:
        kwargs: Dictionary with keys:
            - keys_list: List of keys to process
            - output_file: Path for output HDF5 file.
        source_database_readers: List of source database readers.
        extras_fields: Optional dict mapping entry_key to extra fields.
    """
    keys_list, output_file = (
        kwargs["keys_list"],
        kwargs["output_file"],
    )

    write_hdf5(
        key_entries=keys_list,
        source_db_readers=source_database_readers,
        hdf5_path=output_file,
        extras_fields=extras_fields,
    )
    logger.info("Processed %s keys", len(keys_list))


def merge_hdf5_files(local_files, output_path):
    """Merge list of HDF5 files into one and save to local file.

    Args:
        local_files: list of paths to HDF5 files to merge
        output_path: Path to the local file to save the merged HDF5 file
    """
    with h5py.File(output_path, "w", libver="latest", track_order=False) as out_h5:
        for f in local_files:
            with h5py.File(f, "r") as src_h5:
                for key in src_h5.keys():
                    src_h5.copy(key, out_h5)
