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
"""Tests for the query command's allowlist helpers."""

from pathlib import Path

import numpy as np
import pytest

from chemreporter.query_database_tools.query_tools import (
    load_allowlist_frame,
    resolve_allowlist_columns,
)


def test_resolve_allowlist_columns_str_and_list():
    """Accept a string or a non-empty list of column names."""
    assert resolve_allowlist_columns("smiles") == ["smiles"]
    assert resolve_allowlist_columns(["smiles", "net_charge"]) == [
        "smiles",
        "net_charge",
    ]


def test_resolve_allowlist_columns_empty_raises():
    """Reject an empty column list."""
    with pytest.raises(ValueError, match="at least one"):
        resolve_allowlist_columns([])


def test_load_rejects_unsupported_type(tmp_path: Path):
    """Reject allowlist files that are neither .npz nor .npy."""
    path = tmp_path / "train.csv"
    path.write_text("smiles\nCCO\n")
    with pytest.raises(ValueError, match="Unsupported allowlist"):
        load_allowlist_frame(path, "smiles")


def test_load_npy_single_column(tmp_path: Path):
    """Load a single-column .npy allowlist."""
    path = tmp_path / "train.npy"
    np.save(path, np.array(["CCO", "C=O"]))
    loaded = load_allowlist_frame(path, "smiles")
    assert loaded["smiles"].to_list() == ["CCO", "C=O"]


def test_load_npy_rejects_multi_column(tmp_path: Path):
    """Reject .npy allowlists when more than one column is expected."""
    path = tmp_path / "train.npy"
    np.save(path, np.array(["CCO"]))
    with pytest.raises(ValueError, match="single column"):
        load_allowlist_frame(path, ["smiles", "net_charge"])
