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
"""Resolve and extract fingerprint columns from a query result frame.

Shared by custom sampling plugins (see ``sampling_examples/``) that need to
turn ``entry_key`` + fingerprint columns into a float32 matrix.
"""

from __future__ import annotations

import numpy as np
import polars as pl


def prepare_lazy_frame(
    frame: pl.LazyFrame | pl.DataFrame,
) -> tuple[pl.LazyFrame, list[str]]:
    """Project ``entry_key`` and fingerprint bit columns from an input frame.

    Args:
        frame: Input LazyFrame or DataFrame.

    Returns:
        Projected LazyFrame and the sorted fingerprint bit-column names.

    Raises:
        ValueError: If no fingerprint columns are found.
    """
    lazy_df = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    schema_names = lazy_df.collect_schema().names()

    fp_cols = sorted(
        (c for c in schema_names if c.startswith("fingerprint_")),
        key=lambda name: int(name.rsplit("_", 1)[-1]),
    )
    if not fp_cols:
        msg = "LazyFrame must contain fingerprint_* bit columns."
        raise ValueError(msg)

    return lazy_df.select(["entry_key", *fp_cols]), fp_cols


def fingerprints_from_batch(batch: pl.DataFrame, fp_cols: list[str]) -> np.ndarray:
    """Convert a batch of fingerprint bit columns to a float32 matrix.

    Args:
        batch: Batch with fingerprint bit columns.
        fp_cols: Bit column names.

    Returns:
        Fingerprint matrix of shape ``(n_rows, n_bits)``.
    """
    return batch.select(fp_cols).to_numpy().astype(np.float32)
