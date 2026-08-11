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
"""This module contains a set of functions to perform actions on the query database."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy.stats import gaussian_kde

from chemreporter.query_database_tools.query_database import QueryDatabaseHandler
from chemreporter.query_database_tools.table_schemas import FULL_SCHEMA, get_schema_dict

logger = logging.getLogger("chemreporter")

# Above this length, a joined column-name string is replaced with a short
# placeholder in generated file names, to avoid unwieldy file names.
MAX_COLUMN_NAME_STR_LENGTH = 100
DEFAULT_PLOT_KWARGS = {
    "color": "blue",
    "linewidth": 2,
    "alpha": 0.5,
}


def check_columns_are_supported(columns: list[str]) -> list[str]:
    """Check if the columns are not strings.
    If not, remove the column from the list.

    Args:
        columns: List of columns to check

    Returns:
        List of columns that are numbers
    """
    supported_columns = []
    schema_dict = get_schema_dict(FULL_SCHEMA)
    for col in columns:
        if col not in schema_dict:
            logger.warning("Column %s not in FULL_SCHEMA", col)
            continue
        if schema_dict[col] not in [pl.Int64, pl.Float64, pl.Boolean]:
            logger.warning(
                "Column %s is not pl.Int64, pl.Float64, pl.Boolean: not supported", col
            )
            continue
        supported_columns.append(col)

    return supported_columns


def make_time_stamp() -> str:
    """Return time stamp."""
    return datetime.now().strftime("%Y%m%d-%H%M")


def make_statistics(
    db_handler: QueryDatabaseHandler,
    keys: list[str],
    output_path: Path,
    column_names: list[str] | str | None = None,
) -> None:
    """Print statistics of the given dataframe.

    Args:
        db_handler: QueryDatabaseHandler instance
        keys: List of entry keys to fetch
        output_path: Path to the output file
        column_names: List of columns to print statistics for
    """
    if column_names is None:
        column_names = []
    elif isinstance(column_names, str):
        column_names = [column_names]

    columns = check_columns_are_supported(column_names)

    if not columns:
        logger.warning("No columns to print statistics for")
        return
    df = db_handler.get_dataframe(indices=keys, columns=columns)
    logger.debug("df columns: %s", df.collect_schema().names())

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    columns_str = "-".join(columns) if columns else "all"
    if len(columns_str) > MAX_COLUMN_NAME_STR_LENGTH:
        columns_str = "many"
    output_file_name = f"{make_time_stamp()}-statistics-{columns_str}.csv"

    if columns:
        stats = df.select(columns).describe()
    else:
        stats = df.describe()

    local_path = output_path / output_file_name

    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        stats.write_csv(f)


def make_plot(
    df_series: pl.Series,
    axe: plt.Axes,
    plot_type: Literal["distribution", "histogram"] = "distribution",
) -> plt.Axes:
    """Plot data from a Polars Series on a given matplotlib Axes.

    Parameters:
        df_series (pl.Series): The data to plot.
        axe (plt.Axes): Matplotlib axes to plot on.
        plot_type (str): Type of plot - 'distribution', 'histogram', or 'kde'.

    Returns:
        plt.Axes: The same axes with the plot added.

    Raises:
        ValueError: If the plot type is not supported.

    """
    data = df_series.to_numpy()

    # Handle boolean data - convert to integers for plotting
    if df_series.dtype == pl.Boolean:
        data = data.astype(np.int32)

    if plot_type == "histogram":
        axe.hist(data, bins=100, **DEFAULT_PLOT_KWARGS)

    elif plot_type == "distribution":
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 500)
        y_vals = kde(x_vals)
        axe.plot(x_vals, y_vals, **DEFAULT_PLOT_KWARGS)
        axe.fill_between(x_vals, y_vals, alpha=0.3)

    return axe


def make_histograms(
    db_handler: QueryDatabaseHandler,
    entry_keys: list[str],
    output_path: Path,
    column_names: list[str] | str,
) -> None:
    """Make grid plots for the given column names of the dataframe.

    very basic plots : to be updated
    Note if there is only one column, the plot will be saved as a single plot

    Args:
        db_handler: QueryDatabaseHandler instance
        entry_keys: List of entry keys to fetch
        output_path: Path to the output file
        column_names: List of column names to plot d
    """
    # Fetch dataframe with required columns
    if isinstance(column_names, str):
        column_names = [column_names]
    column_names = check_columns_are_supported(column_names)
    if not column_names:
        logger.warning("No columns to plot")
        return

    df = db_handler.get_dataframe(indices=entry_keys, columns=column_names)

    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    fig_name = "histograms-" + "-".join(column_names)
    if len(fig_name) > MAX_COLUMN_NAME_STR_LENGTH:
        fig_name = "histograms-many-columns"

    output_file_name = f"{make_time_stamp()}-{fig_name}.png"

    local_path = output_path / output_file_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(column_names), 1)

    # If only one subplot, axes is not an array
    if len(column_names) == 1:
        axes = [axes]

    for i, column_name in enumerate(column_names):
        data_series = df[column_name]
        ax = make_plot(data_series, axes[i], plot_type="histogram")
        # Set axis labels if provided

        ax.set_xlabel(f"{column_name}")
        ax.set_ylabel("Counts")
    fig.savefig(local_path)
    plt.close(fig)


def extract_smiles(
    db_handler: QueryDatabaseHandler,
    keys: list[str],
    output_path: Path,
) -> None:
    """Extract smiles from the given database.

    Args:
        db_handler: QueryDatabaseHandler instance
        keys: List of keys to fetch
        output_path: Path to the output file
        kwargs: Additional keyword arguments
    """
    output_file_name = f"{make_time_stamp()}-smiles.npy"

    local_path = output_path / output_file_name
    local_path.parent.mkdir(parents=True, exist_ok=True)
    df = db_handler.get_dataframe(indices=keys, columns=["smiles"])
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    # Filter out null and empty SMILES
    df = df.filter((pl.col("smiles").is_not_null()))

    smiles = df["smiles"].to_numpy()
    with open(local_path, "wb") as f:
        np.save(f, smiles)
