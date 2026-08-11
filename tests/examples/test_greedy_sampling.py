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
"""Tests for streamed greedy Tanimoto sampling."""

from pathlib import Path

import polars as pl

from chemreporter.query_database_tools.sample_plugins import load_sampler_from_path

REPO_ROOT = Path(__file__).resolve().parents[2]
greedy_sample_keys = load_sampler_from_path(
    REPO_ROOT / "sampling_examples/greedy_sampling.py"
)


def _frame_with_orthogonal_fingerprints(n_bits: int = 8) -> pl.DataFrame:
    """Build a tiny frame with mutually distant one-hot fingerprints."""
    rows: list[dict[str, str | int]] = []
    for i in range(n_bits):
        fp = [0] * n_bits
        fp[i] = 1
        row: dict[str, str | int] = {"entry_key": f"mol_{i}"}
        row.update({f"fingerprint_{j}": fp[j] for j in range(n_bits)})
        rows.append(row)
    # Near-duplicate of mol_0: shares the same bit, so Tanimoto distance is 0.
    near_dup: dict[str, str | int] = {"entry_key": "mol_0_dup", "fingerprint_0": 1}
    near_dup.update({f"fingerprint_{j}": 0 for j in range(1, n_bits)})
    rows.append(near_dup)
    return pl.DataFrame(rows)


def test_greedy_sample_keys_respects_n_samples():
    """Fixed ``n_samples`` returns exactly that many distinct keys."""
    frame = _frame_with_orthogonal_fingerprints()
    keys = greedy_sample_keys(
        frame,
        n_samples=3,
        seed=0,
        chunk_size=4,
        min_distance_threshold=0.5,
    )
    assert len(keys) == 3
    assert len(set(keys)) == 3


def test_n_samples_none_skips_near_duplicates_until_exhausted():
    """With ``n_samples=None``, near-duplicates never enter the pool."""
    frame = _frame_with_orthogonal_fingerprints(n_bits=4)
    keys = greedy_sample_keys(
        frame,
        n_samples=None,
        seed=0,
        start_idx=0,
        chunk_size=10,
        min_distance_threshold=0.5,
    )
    assert "mol_0_dup" not in keys
    assert len(keys) == 4
    assert set(keys) == {f"mol_{i}" for i in range(4)}


def test_upper_bound_with_exhaustion_returns_early():
    """Upper bound plus exhaustion returns fewer than ``n_samples``."""
    frame = _frame_with_orthogonal_fingerprints(n_bits=3)
    # Force a high threshold so only the seed clears it once similar fps remain.
    keys = greedy_sample_keys(
        frame,
        n_samples=10,
        seed=0,
        start_idx=0,
        chunk_size=10,
        min_distance_threshold=0.99,
    )
    assert 1 <= len(keys) <= 3
    assert len(keys) < 10
