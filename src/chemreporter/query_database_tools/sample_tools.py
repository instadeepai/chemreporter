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
import logging

import polars as pl

logger = logging.getLogger("chemreporter")


def random_sampler(
    df: pl.DataFrame | pl.LazyFrame,
    n_select: int,
    seed: int | None = None,
) -> pl.DataFrame | pl.LazyFrame:
    """Randomly sample a subset of the database.

    Args:
        df: Polars DataFrame or LazyFrame
        n_select: number of molecules to select
        seed: seed for the random number generator

    Returns:
        keys: Sampled DataFrame or LazyFrame with entry_key column

    Raises:
        ValueError: if n_select is greater than the number of molecules in the database
    """
    if isinstance(df, pl.LazyFrame):
        # For LazyFrame, collect first then sample
        df = df.collect()
    if len(df) < n_select:
        raise ValueError(
            f"n_samples: {n_select} is greater than the number "
            f"of molecules in the database: {len(df)}"
        )
    return df.sample(n_select, seed=seed).select(entry_key=pl.col("entry_key"))
