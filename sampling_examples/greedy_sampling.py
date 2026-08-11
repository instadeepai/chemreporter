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
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import polars as pl

from chemreporter.query_database_tools.fingerprint_columns import (
    fingerprints_from_batch,
    prepare_lazy_frame,
)

logger = logging.getLogger("chemreporter")

DEFAULT_CHUNK_SIZE = 25_000
DEFAULT_POOL_CHUNK_SIZE = 4_096
DEFAULT_MIN_DISTANCE_THRESHOLD = 0.65


def _unselected_mask(keys: np.ndarray, selected_set: set[str]) -> np.ndarray:
    """Boolean mask of ``keys`` not already present in ``selected_set``.

    Returns:
        Unselected-key mask.
    """
    return np.array([key not in selected_set for key in keys], dtype=bool)


def _min_tanimoto_distances_jax(
    batch_fps: jnp.ndarray,
    pool_fps: jnp.ndarray,
) -> jnp.ndarray:
    """Minimum Tanimoto distance from each batch row to the selected pool.

    Args:
        batch_fps: Candidate fingerprint matrix of shape ``(batch_size, n_bits)``.
        pool_fps: Selected pool fingerprint matrix of shape ``(pool_size, n_bits)``.

    Returns:
        Array of shape (batch_size,) with minimum distances to the pool.
    """
    intersection = batch_fps @ pool_fps.T
    batch_counts = jnp.sum(batch_fps * batch_fps, axis=1, keepdims=True)
    pool_counts = jnp.sum(pool_fps * pool_fps, axis=1, keepdims=True)
    union = batch_counts + pool_counts.T - intersection
    distances = 1.0 - (intersection / (union + 1e-8))
    return jnp.min(distances, axis=1)


def _min_tanimoto_distances_to_pool(
    batch_fps: jnp.ndarray,
    pool_fps: jnp.ndarray,
    *,
    jitted_tanimoto_distances: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
    """Minimum Tanimoto distance from each batch row to a large selected pool.

    Processes the pool in chunks of ``DEFAULT_POOL_CHUNK_SIZE`` so peak memory
    stays bounded instead of growing unbounded as the pool fills up under a
    loose ``min_distance_threshold``.

    Args:
        batch_fps: Candidate fingerprint matrix of shape ``(batch_size, n_bits)``.
        pool_fps: Selected pool fingerprint matrix of shape ``(pool_size, n_bits)``.
        jitted_tanimoto_distances: Jitted Tanimoto distance function.

    Returns:
        Array of shape ``(batch_size,)`` with minimum distances to the pool.
    """
    min_dists = np.full(batch_fps.shape[0], np.inf, dtype=np.float32)

    for start in range(0, pool_fps.shape[0], DEFAULT_POOL_CHUNK_SIZE):
        end = min(start + DEFAULT_POOL_CHUNK_SIZE, pool_fps.shape[0])
        chunk_mins = jitted_tanimoto_distances(batch_fps, pool_fps[start:end])
        np.minimum(min_dists, chunk_mins, out=min_dists)
    return min_dists


def _collect_seed_from_stream(
    lazy_df: pl.LazyFrame,
    fp_cols: list[str],
    *,
    start_idx: int | None,
    seed: int | None,
    chunk_size: int,
) -> tuple[str, np.ndarray]:
    """Pick the initial pool structure without scanning the full dataset.

    Args:
        lazy_df: Filtered LazyFrame with fingerprint columns.
        fp_cols: Bit column names.
        start_idx: Optional absolute row index. Expensive on large remote datasets.
        seed: Optional RNG seed used to pick a row from the first batch.
        chunk_size: Batch size used to read the first chunk.

    Returns:
        Seed entry key and fingerprint vector.

    Raises:
        ValueError: If the filtered frame is empty or ``start_idx`` is invalid.
    """
    if start_idx is not None:
        seed_df = lazy_df.slice(start_idx, 1).collect()
        if seed_df.is_empty():
            raise ValueError(f"start_idx {start_idx} is out of range.")
        return (
            str(seed_df["entry_key"][0]),
            fingerprints_from_batch(seed_df, fp_cols)[0],
        )

    batches = lazy_df.collect_batches(chunk_size=chunk_size)
    try:
        first_batch = next(batches)
    except StopIteration:
        first_batch = pl.DataFrame()

    if first_batch.is_empty():
        raise ValueError("No structures matched the filter query.")

    if seed is not None:
        rng = np.random.default_rng(seed)
        row_idx = int(rng.integers(0, len(first_batch)))
    else:
        row_idx = 0

    seed_row = first_batch.slice(row_idx, 1)
    return (
        str(seed_row["entry_key"][0]),
        fingerprints_from_batch(seed_row, fp_cols)[0],
    )


def streaming_greedy_selection_lazy(
    lazy_df: pl.LazyFrame,
    fp_cols: list[str],
    num_selected: int | None = None,
    start_idx: int | None = None,
    seed: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    min_distance_threshold: float = DEFAULT_MIN_DISTANCE_THRESHOLD,
) -> tuple[list[str], np.ndarray]:
    """Select diverse structures with a single-pass streamed sweep.

    Args:
        lazy_df: LazyFrame with ``entry_key`` and fingerprint columns.
        fp_cols: Bit column names.
        num_selected: Optional upper bound on structures to select. When
            ``None``, selection runs until the distance threshold is exhausted.
        start_idx: Optional starting row index. Random when omitted and ``seed``
            is set.
        seed: Optional RNG seed used only when ``start_idx`` is omitted.
        chunk_size: Number of rows processed per streaming batch.
        min_distance_threshold: Minimum Tanimoto distance to the current pool.

    Returns:
        Tuple of selected entry keys and their fingerprint matrix.

    Raises:
        ValueError: If ``num_selected`` is invalid for the input frame.
    """
    jitted_tanimoto_distances = jax.jit(_min_tanimoto_distances_jax)
    if num_selected is not None and num_selected <= 0:
        raise ValueError("num_selected must be positive.")

    logger.info(
        "Starting streaming greedy selection (entry_key + fingerprints only)..."
    )
    seed_key, seed_fp = _collect_seed_from_stream(
        lazy_df,
        fp_cols,
        start_idx=start_idx,
        seed=seed,
        chunk_size=chunk_size,
    )

    selected_keys = [seed_key]
    selected_set = {seed_key}
    pool_fps = seed_fp.reshape(1, -1).astype(np.float32, copy=False)

    for batch_df in lazy_df.collect_batches(chunk_size=chunk_size):
        if num_selected is not None and len(selected_keys) >= num_selected:
            break

        keys = batch_df["entry_key"].to_numpy()
        fps = fingerprints_from_batch(batch_df, fp_cols)
        unselected_mask = _unselected_mask(keys, selected_set)

        if not unselected_mask.any():
            continue

        candidate_keys = keys[unselected_mask]
        candidate_fps = fps[unselected_mask]
        chunk_min_dists = _min_tanimoto_distances_to_pool(
            jnp.array(candidate_fps),
            jnp.array(pool_fps),
            jitted_tanimoto_distances=jitted_tanimoto_distances,
        )
        viable_indices = np.where(chunk_min_dists > min_distance_threshold)[0]

        new_fps = []
        for local_idx in viable_indices:
            if num_selected is not None and len(selected_keys) >= num_selected:
                break

            key = str(candidate_keys[local_idx])
            if key in selected_set:
                continue

            selected_keys.append(key)
            selected_set.add(key)
            new_fps.append(candidate_fps[local_idx].astype(np.float32, copy=False))

        if new_fps:
            pool_fps = np.vstack([pool_fps, np.stack(new_fps)])

    current_pool_size = len(selected_keys)

    if num_selected is None:
        return selected_keys, pool_fps

    if current_pool_size < num_selected:
        logger.warning(
            "Could select only %s/%s structures. Lower min_distance_threshold "
            "or reduce n_samples.",
            current_pool_size,
            num_selected,
        )

    return selected_keys[:num_selected], pool_fps[:num_selected]


def run_greedy_sampler(
    frame: pl.LazyFrame | pl.DataFrame,
    n_samples: int | None = None,
    start_idx: int | None = None,
    seed: int | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    min_distance_threshold: float = DEFAULT_MIN_DISTANCE_THRESHOLD,
) -> list[str]:
    """Select diverse structures using streamed Tanimoto sampling.

    Args:
        frame: LazyFrame or DataFrame with ``entry_key`` and fingerprint columns.
        n_samples: Optional upper bound on structures to select. Omit (``None``)
            to keep adding until the distance threshold is exhausted.
        start_idx: Optional starting row index for selection.
        seed: Optional RNG seed used only when ``start_idx`` is omitted.
        chunk_size: Number of rows processed per streaming batch.
        min_distance_threshold: Minimum Tanimoto distance to the current pool.

    Returns:
        ``entry_key`` values for the selected structures.

    Raises:
        ValueError: If fingerprint columns are missing or selection is invalid.
    """
    lazy_df, fp_cols = prepare_lazy_frame(frame)
    selected_keys, _pool_fps = streaming_greedy_selection_lazy(
        lazy_df,
        fp_cols,
        n_samples,
        start_idx=start_idx,
        seed=seed,
        chunk_size=chunk_size,
        min_distance_threshold=min_distance_threshold,
    )

    logger.info("Selected %s structures", len(selected_keys))
    return selected_keys


# The plugin loader (chemreporter.query_database_tools.sample_plugins) looks up
# custom_sampling_function as the entrypoint, so our custom function above is
# aliased to it
custom_sampling_function = run_greedy_sampler
