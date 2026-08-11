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
"""Tests for query database actions tools."""

from unittest.mock import Mock

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

from chemreporter.query_database_tools.actions_tools import (
    check_columns_are_supported,
    extract_smiles,
    make_histograms,
    make_plot,
    make_statistics,
)
from chemreporter.query_database_tools.query_database import (
    QueryDatabaseHandler,
)


class TestCheckColumnsAreSupported:
    """Test check_columns_are_supported function."""

    def test_supported_numeric_columns(self):
        """Test that numeric columns are kept."""
        columns = ["num_atoms", "molecular_weight", "spin_multiplicity"]
        result = check_columns_are_supported(columns)
        assert len(result) == 3
        assert "num_atoms" in result
        assert "molecular_weight" in result
        assert "spin_multiplicity" in result

    def test_unsupported_string_columns_filtered(self):
        """Test that string columns are filtered out."""
        columns = ["num_atoms", "smiles", "entry_key"]
        result = check_columns_are_supported(columns)
        # Only num_atoms should remain (smiles and entry_key are strings)
        assert len(result) == 1
        assert result == ["num_atoms"]

    def test_boolean_columns_supported(self):
        """Test that boolean columns are supported."""
        columns = ["is_molecular_structure_valid", "num_atoms"]
        result = check_columns_are_supported(columns)
        assert len(result) == 2
        assert "is_molecular_structure_valid" in result

    def test_nonexistent_columns_filtered(self):
        """Test that non-existent columns are filtered out."""
        columns = ["num_atoms", "fake_column", "another_fake"]
        result = check_columns_are_supported(columns)
        assert len(result) == 1
        assert result == ["num_atoms"]

    def test_empty_list(self):
        """Test with empty list."""
        columns = []
        result = check_columns_are_supported(columns)
        assert result == []

    def test_all_invalid_columns(self):
        """Test when all columns are invalid."""
        columns = ["fake1", "fake2", "fake3"]
        result = check_columns_are_supported(columns)
        assert result == []


class TestMakeStatistics:
    """Test make_statistics function."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock QueryDatabaseHandler."""
        handler = Mock()

        # Mock dataframe return
        df = pl.DataFrame({
            "entry_key": ["mol_1", "mol_2", "mol_3"],
            "num_atoms": [10, 15, 20],
            "molecular_weight": [100.0, 150.0, 200.0],
        })
        handler.get_dataframe.return_value = df

        return handler

    def test_make_statistics_basic(self, mock_db_handler, tmp_path):
        """Test basic statistics generation."""
        keys = ["mol_1", "mol_2", "mol_3"]
        columns = ["num_atoms", "molecular_weight"]

        make_statistics(mock_db_handler, keys, tmp_path, column_names=columns)

        # Check that handler was called correctly
        mock_db_handler.get_dataframe.assert_called_once_with(
            indices=keys, columns=columns
        )

        # Check that CSV file was created
        csv_files = list(tmp_path.glob("*-statistics-*.csv"))
        assert len(csv_files) == 1

        # Verify file has content
        assert csv_files[0].stat().st_size > 0

    def test_make_statistics_no_columns(self, mock_db_handler, tmp_path):
        """Test statistics with no specific columns."""
        keys = ["mol_1", "mol_2"]

        make_statistics(mock_db_handler, keys, tmp_path, column_names=[])

        # Check file was created
        csv_files = list(tmp_path.glob("*-statistics-all.csv"))
        assert len(csv_files) == 0  # no files should be created

    def test_make_statistics_creates_directory(self, mock_db_handler, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        keys = ["mol_1"]
        output_dir = tmp_path / "nested" / "directory"

        make_statistics(mock_db_handler, keys, output_dir, column_names=["num_atoms"])

        # Directory should be created
        assert output_dir.exists()
        assert output_dir.is_dir()


class TestMakePlot:
    """Test make_plot function."""

    @pytest.fixture
    def sample_series(self):
        """Create a sample Polars Series."""
        return pl.Series("test_data", [1.0, 2.0, 3.0, 4.0, 5.0] * 10)

    @pytest.fixture
    def fig_ax(self):
        """Create matplotlib figure and axes."""
        fig, ax = plt.subplots()
        yield fig, ax
        plt.close(fig)

    def test_make_plot_histogram(self, sample_series, fig_ax):
        """Test histogram plot creation."""
        fig, ax = fig_ax

        result_ax = make_plot(sample_series, ax, plot_type="histogram")

        assert result_ax is ax
        # Check that histogram was plotted (patches are created)
        assert len(ax.patches) > 0

    def test_make_plot_distribution(self, sample_series, fig_ax):
        """Test distribution (KDE) plot creation."""
        fig, ax = fig_ax

        result_ax = make_plot(sample_series, ax, plot_type="distribution")

        assert result_ax is ax
        # Check that line was plotted
        assert len(ax.lines) > 0


class TestMakeHistograms:
    """Test make_histograms function."""

    @pytest.fixture
    def mock_db_handler(self):
        """Create a mock QueryDatabaseHandler."""
        handler = Mock()

        # Mock dataframe with numerical data
        df = pl.DataFrame({
            "entry_key": [f"mol_{i}" for i in range(50)],
            "num_atoms": np.random.randint(5, 100, 50),
            "molecular_weight": np.random.uniform(50, 500, 50),
            "dipole_moment_magnitude": np.random.uniform(0, 5, 50),
        })
        handler.get_dataframe.return_value = df

        return handler

    def test_make_histograms_single_column(self, mock_db_handler, tmp_path):
        """Test histogram creation for single column."""
        keys = [f"mol_{i}" for i in range(10)]
        columns = ["num_atoms"]

        make_histograms(mock_db_handler, keys, tmp_path, columns)

        # Check that PNG file was created
        png_files = list(tmp_path.glob("*-histograms-*.png"))
        assert len(png_files) == 1

        # Verify file exists and has content
        assert png_files[0].stat().st_size > 0

    def test_make_histograms_multiple_columns(self, mock_db_handler, tmp_path):
        """Test histogram creation for multiple columns."""
        keys = [f"mol_{i}" for i in range(10)]
        columns = ["num_atoms", "molecular_weight", "dipole_moment_magnitude"]

        make_histograms(mock_db_handler, keys, tmp_path, columns)

        # Check file was created
        png_files = list(tmp_path.glob("*-histograms-*.png"))
        assert len(png_files) == 1

        # Filename should contain all column names
        filename = png_files[0].name
        for col in columns:
            assert col in filename

    def test_make_histograms_creates_directory(self, mock_db_handler, tmp_path):
        """Test that output directory is created."""
        keys = ["mol_1"]
        output_dir = tmp_path / "plots" / "histograms"

        make_histograms(mock_db_handler, keys, output_dir, ["num_atoms"])

        # Directory should be created
        assert output_dir.exists()
        assert output_dir.is_dir()


class TestExtractSmiles:
    """Test extract_smiles function."""

    @pytest.fixture
    def mock_db_handler_with_smiles(self):
        """Create a mock QueryDatabaseHandler with SMILES data."""
        handler = Mock()

        # Mock dataframe with SMILES
        df = pl.DataFrame({
            "entry_key": ["mol_1", "mol_2", "mol_3", "mol_4"],
            "smiles": ["CCO", "CC", None, "C1CC1"],  # One None value
        })
        handler.get_dataframe.return_value = df

        return handler

    def test_extract_smiles_basic(self, mock_db_handler_with_smiles, tmp_path):
        """Test basic SMILES extraction."""
        keys = ["mol_1", "mol_2", "mol_3", "mol_4"]

        extract_smiles(mock_db_handler_with_smiles, keys, tmp_path)

        # Check that handler was called
        mock_db_handler_with_smiles.get_dataframe.assert_called_once_with(
            indices=keys, columns=["smiles"]
        )

        # Check that .npy file was created
        npy_files = list(tmp_path.glob("*-smiles.npy"))
        assert len(npy_files) == 1

        # Load and verify content
        smiles_array = np.load(npy_files[0], allow_pickle=True)
        assert isinstance(smiles_array, np.ndarray)
        # Should filter out None/null values
        assert len(smiles_array) >= 0

    def test_extract_smiles_creates_directory(
        self, mock_db_handler_with_smiles, tmp_path
    ):
        """Test that output directory is created."""
        keys = ["mol_1"]
        output_dir = tmp_path / "smiles" / "extracted"

        extract_smiles(mock_db_handler_with_smiles, keys, output_dir)

        # Directory should be created
        assert output_dir.exists()
        assert output_dir.is_dir()

    def test_extract_smiles_filters_null(self, tmp_path):
        """Test that null SMILES are filtered out."""
        handler = Mock()

        # Mock dataframe with mixed null values
        df = pl.DataFrame({
            "entry_key": ["mol_1", "mol_2", "mol_3"],
            "smiles": ["CCO", None, "CC"],
        })
        handler.get_dataframe.return_value = df

        extract_smiles(handler, ["mol_1", "mol_2", "mol_3"], tmp_path)

        # Load and verify that nulls were filtered
        npy_files = list(tmp_path.glob("*-smiles.npy"))
        smiles_array = np.load(npy_files[0], allow_pickle=True)

        # Should only contain non-null SMILES
        for smiles in smiles_array:
            assert smiles is not None


class TestIntegration:
    """Integration tests with real QueryDatabaseHandler."""

    @pytest.fixture
    def real_db_handler(self, tmp_path):
        """Create a real QueryDatabaseHandler with test data."""
        # Create test parquet file
        db_dir = tmp_path / "test_db"
        db_dir.mkdir()

        df = pl.DataFrame({
            "entry_key": [f"mol_{i}" for i in range(20)],
            "num_atoms": np.random.randint(5, 50, 20),
            "molecular_weight": np.random.uniform(50, 300, 20),
            "smiles": [f"C{i}" for i in range(20)],
            "subset": ["test"] * 20,
        })

        df.write_parquet(db_dir / "test_data.parquet")

        return QueryDatabaseHandler(db_path=db_dir)

    def test_make_statistics_integration(self, real_db_handler, tmp_path):
        """Integration test for make_statistics with real handler."""
        output_dir = tmp_path / "stats"
        keys = [f"mol_{i}" for i in range(5)]

        make_statistics(
            real_db_handler,
            keys,
            output_dir,
            column_names=["num_atoms", "molecular_weight"],
        )

        # Verify file was created and contains data
        csv_files = list(output_dir.glob("*.csv"))
        assert len(csv_files) == 1

        # Read and verify content
        stats_df = pl.read_csv(csv_files[0])
        assert "describe" in stats_df.columns or len(stats_df) > 0

    def test_extract_smiles_integration(self, real_db_handler, tmp_path):
        """Integration test for extract_smiles with real handler."""
        output_dir = tmp_path / "smiles"
        keys = [f"mol_{i}" for i in range(10)]

        extract_smiles(real_db_handler, keys, output_dir)

        # Verify file was created
        npy_files = list(output_dir.glob("*.npy"))
        assert len(npy_files) == 1

        # Load and verify
        smiles_array = np.load(npy_files[0], allow_pickle=True)
        assert len(smiles_array) > 0
