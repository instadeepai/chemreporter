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
"""Helpers for the query command's allowlist-based ``restrict_to`` filter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl


def resolve_allowlist_columns(columns: str | list[str]) -> list[str]:
    """Normalize allowlist column config to a non-empty list of column names.

    Args:
        columns: A single column name or a list of column names.

    Returns:
        Column names that define the composite identity.

    Raises:
        ValueError: If no columns are provided.
    """
    if isinstance(columns, str):
        normalized = [columns]
    else:
        normalized = [str(column) for column in columns]
    if not normalized:
        raise ValueError("columns must contain at least one column.")
    return normalized


def load_allowlist_frame(
    path: str | Path,
    columns: str | list[str],
) -> pl.DataFrame:
    """Load a NumPy ``.npz`` or ``.npy`` allowlist as a DataFrame.

    Args:
        path: Path to a ``.npz`` file with one named array per column, or a
            ``.npy`` file holding a single column's values.
        columns: Expected allowlist column name(s).

    Returns:
        DataFrame with exactly ``columns``.

    Raises:
        ValueError: If the file type or contents do not match ``columns``.
        FileNotFoundError: If ``path`` does not exist.
    """
    columns_list = resolve_allowlist_columns(columns)
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Allowlist not found: {path}")

    suffix = file_path.suffix.lower()
    if suffix not in (".npz", ".npy"):
        raise ValueError(
            f"Unsupported allowlist type {file_path.suffix!r} for {path}. "
            "Use .npz or .npy only."
        )

    if suffix == ".npy":
        if len(columns_list) != 1:
            raise ValueError(
                f".npy allowlist supports a single column, got {columns_list}."
            )
        array = np.load(file_path, allow_pickle=True)
        data = {columns_list[0]: np.asarray(array).ravel()}
    else:
        with np.load(file_path, allow_pickle=True) as archive:
            missing = [column for column in columns_list if column not in archive.files]
            if missing:
                raise ValueError(
                    f".npz missing columns {missing}; has {archive.files}."
                )
            data = {
                column: np.asarray(archive[column]).ravel() for column in columns_list
            }

    lengths = {name: len(array) for name, array in data.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"allowlist column lengths are misaligned: {lengths}.")
    return pl.DataFrame(data)
