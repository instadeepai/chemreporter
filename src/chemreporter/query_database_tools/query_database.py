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
import secrets
from functools import cached_property
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, cast

import numpy as np
import polars as pl
import polars.exceptions

from chemreporter.query_database_tools.sample_tools import random_sampler
from chemreporter.query_database_tools.table_schemas import (
    FINGERPRINTS_SCHEMA,
    FULL_SCHEMA,
    get_schema_dict,
)

logger = logging.getLogger("chemreporter")

MAX_ATTEMPTS_GEN_PARQUET_PATH = 1000
FINGERPRINT_COLUMNS = [field.name for field in FINGERPRINTS_SCHEMA]
FINGERPRINT_BITS_PRESET = "fingerprint_bits"


def _expand_sampling_required_columns(
    sampling_required_columns: list[str] | None,
) -> list[str]:
    """Expand sampling column presets into concrete column names.

    Args:
        sampling_required_columns: Columns required by the custom sampler.

    Returns:
        Concrete column names with ``fingerprint_bits`` expanded.

    Raises:
        ValueError: If no sampling columns are provided.
    """
    if not sampling_required_columns:
        raise ValueError(
            "sampling_required_columns is required when using a custom sampling method."
        )

    expanded: list[str] = []
    for column in sampling_required_columns:
        if column == FINGERPRINT_BITS_PRESET:
            expanded.extend(FINGERPRINT_COLUMNS)
        else:
            expanded.append(column)
    return expanded


class QueryDatabaseInputError(Exception):
    """QueryDatabaseInputError class."""

    pass


def default_save_function(df: pl.DataFrame, path: str) -> None:
    """Save a Polars DataFrame as a Parquet file."""
    df.write_parquet(path)


def return_db_paths(path: Path) -> list[str] | None:
    """Return a list of all parquet files in the directory.

    Returns:
        List of string paths to the parquet files.
        None if the path is not a directory or file.
    """
    if path.is_dir():
        list_paths = list(path.glob("*.parquet"))
        if len(list_paths) == 0:
            return None
        return [str(p) for p in list_paths]
    elif path.is_file():
        if path.exists() is False:
            return None
        return [str(path)]
    else:
        return None


def default_lazy_read_function(
    db_path: Path, schema: dict | None = None
) -> pl.LazyFrame:
    """Read all parquet files in the directory.

    Args:
        db_path: Path to the database directory or file
        schema: Optional schema dictionary to enforce strictly

    Returns:
        A Polars LazyFrame.

    Raises:
        FileNotFoundError: If the database directory or file is not found
    """
    db_paths = return_db_paths(db_path)
    if db_paths is None:
        raise FileNotFoundError(f"No db files found at {db_path}")
    return pl.scan_parquet(db_paths)


def simple_read_function(dir_path: Path, schema: dict | None = None) -> pl.DataFrame:
    """Read all parquet files in the directory.

    Args:
        dir_path: Directory containing parquet files
        schema: Optional schema dictionary to enforce strictly

    Returns:
        A Polars DataFrame.

    Raises:
        FileNotFoundError: If the database directory or file is not found
    """
    paths = return_db_paths(dir_path)
    if paths is None:
        raise FileNotFoundError(f"No db files found at {dir_path}")
    if schema is not None:
        df = pl.read_parquet(paths, schema=schema)
    else:
        df = pl.read_parquet(paths)
    return df


class QueryDatabaseHandler:
    """Parquet storage backend for query database."""

    def __init__(
        self,
        db_path: Path,
        save_func: Callable = default_save_function,
        read_func: Callable = default_lazy_read_function,
    ) -> None:
        """Initialize S3 Parquet backend."""
        self.db_path = db_path
        self.table_name = "query_database"

        self.save_function: Callable = save_func
        self.read_function: Callable = read_func

    def _get_parquet_path(self) -> Path:
        """Get a new unique file path with random hex suffix.

        Returns:
            Path: A unique parquet file path.

        Raises:
            ValueError: If no new path found after MAX_ATTEMPTS_GEN_PARQUET_PATH
            attempts.
        """
        for _attempt in range(MAX_ATTEMPTS_GEN_PARQUET_PATH):
            code = secrets.token_hex(4)
            path = self.db_path / f"{self.table_name}_{code}.parquet"

            if not path.exists():
                return path

        raise ValueError(
            f"No new path found after {MAX_ATTEMPTS_GEN_PARQUET_PATH} \
            attempts"
        )

    def are_keys_already_in_query_database(self, keys: list[str]) -> bool:
        """Check if any of the given keys already exist in the query database.

        Args:
            keys: List of keys to check.

        Returns:
            True if any of the keys already exist in the query database
            False otherwise.
        """
        return any(key in self._existing_keys for key in keys)

    def store(self, df: pl.DataFrame, check_keys: bool = True) -> None:
        """Store DataFrame as a Parquet file in the database.

        Args:
            df: Polars DataFrame
            check_keys: Whether to check if the keys are already in the query database.
                Default is True.

        Raises:
            QueryDatabaseInputError: If (some of the) keys already exist in the
                query database.
        """
        if df.is_empty():
            logger.info("DataFrame is empty. Skipping upload.")
            return
        # check if the keys are not already in the query database
        if check_keys and self.are_keys_already_in_query_database(
            df["entry_key"].to_list()
        ):
            logger.debug("Skipping upload.")
            raise QueryDatabaseInputError(
                "(some of the) keys are already in the query database. Exiting."
            )

        path = self._get_parquet_path()
        logger.debug("writing in %s (new file)", path)
        self.save_function(df, path)

    def get_dataframe(
        self, indices: list[str], columns: list[str] | str
    ) -> pl.DataFrame:
        """Get the dataframe from the indices and columns.

        Args:
            indices: Indices to get.
            columns: Columns to get.

        Returns:
            dataframe with wanted columns and indices
        """
        if isinstance(columns, str):
            columns = [columns]

        # Ensure entry_key is in columns (avoid duplicates)
        if "entry_key" not in columns:
            columns = columns + ["entry_key"]

        filtered_df = self.read_function(
            self.db_path, schema=get_schema_dict(FULL_SCHEMA)
        )

        # Convert indices to list if it's a numpy array
        if hasattr(indices, "tolist"):
            indices = indices.tolist()
        # Flatten if it's a 2D array/list
        if indices and isinstance(indices[0], (list, tuple)):
            indices = [
                item[0] if isinstance(item, (list, tuple)) else item for item in indices
            ]

        # Create DataFrame with explicit str type to match filtered_df schema
        keys_df = pl.DataFrame(
            {"entry_key": indices}, schema={"entry_key": pl.String}
        ).lazy()

        filtered_df = filtered_df.join(keys_df, on="entry_key", how="semi")

        # Select desired columns and collect to DataFrame
        result = filtered_df.select(columns)

        return result

    @cached_property
    def _existing_keys(self) -> set[str]:
        """Cache all keys in the database.

        To be used only for checking if keys are already in the database
        when adding new entries. It won't track if new entries are added.

        Returns:
            Set of all keys in the database.
        """
        db_paths = return_db_paths(self.db_path)
        if db_paths is None:
            # new database: no keys to be found !
            return set()
        df = self.read_function(self.db_path, schema=get_schema_dict(FULL_SCHEMA))
        df_all_keys = df.sql(
            query=f"SELECT entry_key FROM {self.table_name}",
            table_name=self.table_name,
        )
        if isinstance(df_all_keys, pl.LazyFrame):
            df_all_keys = df_all_keys.collect()
        return set(df_all_keys["entry_key"].to_list())

    def query_to_keys(
        self,
        query: str,
        n_samples: int | None = None,
        sampling_method: Literal["random", "all"] | Callable = "random",
        sampling_required_columns: list[str] | None = None,
        sampling_kwargs: dict | None = None,
        seed: int | None = None,
        restrict_to: Mapping[str, Any] | None = None,
    ) -> np.ndarray:
        """Perform a SQL query on the database.

        Note:
        If the full query is provided, the query is executed on the entire database.
        If only the second part of the query is provided,
        the query will return the database indices

        Order when "restrict_to" is set: SQL filter → semi-join on allowlist
        columns → sampling.

        Args:
            query: SQL query to perform.
            n_samples: Number of samples to return.
            sampling_method: Built-in sampler name (``random`` or ``all``), or
                a custom sampler callable ``(frame, n_samples, **kwargs)``.
            sampling_required_columns: Optional column hint for custom samplers.
                Fingerprint columns are projected lazily inside the sampler.
            sampling_kwargs: Kwargs forwarded to ``sampling_method`` when it
                is a custom sampler callable.
            seed: Seed for built-in random sampling (optional).
            restrict_to: Optional allowlist with "columns" and a Polars
                "values" DataFrame; rows must match on all columns.

        Returns:
            NumPy array of entry_key strings.

        Raises:
            QueryDatabaseInputError: If the SQL query is invalid.
            ValueError: If sampling method is invalid or "restrict_to" is
                malformed.
        """
        sampling_kwargs = sampling_kwargs or {}

        if not callable(sampling_method) and sampling_method not in ["random", "all"]:
            raise ValueError(f"Invalid sampling method: {sampling_method}")

        df = self.read_function(self.db_path, schema=get_schema_dict(FULL_SCHEMA))
        if not query:
            raise QueryDatabaseInputError("No query provided.")

        select_columns = {"entry_key"}
        allowlist_frame: pl.DataFrame | None = None
        allowlist_columns: list[str] | None = None
        if restrict_to is not None:
            if "columns" not in restrict_to or "values" not in restrict_to:
                raise ValueError("restrict_to requires 'columns' and 'values' keys.")
            allowlist_columns = cast(list[str], restrict_to["columns"])
            allowlist_frame = restrict_to["values"]
            if not isinstance(allowlist_frame, pl.DataFrame):
                raise ValueError("restrict_to['values'] must be a polars DataFrame.")
            select_columns.update(allowlist_columns)

        if callable(sampling_method):
            select_columns.update(
                _expand_sampling_required_columns(sampling_required_columns)
            )

        columns_sql = ", ".join(select_columns)
        sql_query = f"SELECT {columns_sql} FROM {self.table_name} WHERE {query}"

        logger.debug("Query: %s", sql_query)
        df_filtered = df.sql(
            query=sql_query,
            table_name=self.table_name,
        )

        if allowlist_frame is not None and allowlist_columns is not None:
            # Normalize both sides to LazyFrame (same pattern as get_dataframe).
            allowlist = allowlist_frame.select(allowlist_columns)
            if isinstance(df_filtered, pl.DataFrame):
                df_filtered = df_filtered.lazy()
            left_schema = df_filtered.collect_schema()
            cast_exprs = [
                pl.col(column).cast(left_schema[column])
                for column in allowlist_columns
                if column in left_schema
                and allowlist[column].dtype != left_schema[column]
            ]
            if cast_exprs:
                allowlist = allowlist.with_columns(cast_exprs)
            df_filtered = df_filtered.join(
                allowlist.lazy(),
                on=allowlist_columns,
                how="semi",
            )

        if callable(sampling_method):
            sampled = sampling_method(
                df_filtered,
                n_samples,
                **sampling_kwargs,
            )
            if isinstance(sampled, list):
                return np.array(sampled, dtype=object)
            df_filtered = sampled
        elif n_samples and sampling_method == "random":
            df_filtered = random_sampler(df_filtered, n_samples, seed=seed)

        # Collect LazyFrame and convert to numpy array
        if isinstance(df_filtered, pl.LazyFrame):
            try:
                return df_filtered.select("entry_key").collect()["entry_key"].to_numpy()

            except polars.exceptions.ColumnNotFoundError:
                available_columns = [
                    c
                    for c in df.collect_schema().names()
                    if not c.startswith("fingerprint")
                ]
                if any(
                    c.startswith("fingerprint") for c in df.collect_schema().names()
                ):
                    available_columns.append("fingerprint_[0-1023]")
                available_columns_str = ", ".join(available_columns)
                raise QueryDatabaseInputError(
                    "Column(s) in your query were not found in the database."
                    "\nHint: check your query is correct.\n"
                    f"Available columns: {available_columns_str}"
                )
        else:
            return df_filtered["entry_key"].to_numpy()
