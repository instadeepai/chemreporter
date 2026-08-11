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
"""Tests for source field parsers."""

from pathlib import Path

from chemreporter.source_database_tools.database_reader import SourceDatabaseReader
from chemreporter.source_database_tools.omol25.reactivity_source_field_parser import (
    parse_ani2x_reactivity_source,
    parse_reactivity_reactivity_source,
    parse_rgd_reactivity_source,
    parse_trans1x_reactivity_source,
)

# Path to sample databases
SAMPLES_DIR = Path(__file__).parent / "data" / "samples"


class TestParseRgdReactivitySource:
    """Test RGD source field parser."""

    def test_parse_valid_rgd_source(self):
        """Test parsing valid RGD source format."""
        source = "rgd_uks/MR_693393_1_13_0_1/orca.tar.zst"
        result = parse_rgd_reactivity_source(source)

        assert result["additional_subset_name"] == "rgd_uks"
        assert result["reaction_id"] == "MR_693393"
        assert result["reaction_pathway_id"] == 1
        assert result["reaction_step_idx"] == 13
        assert result["is_reactant"] == 0
        assert result["is_product"] == 0

    def test_parse_rgd_source_reactant(self):
        """Test parsing RGD source where step is 0 (reactant)."""
        source = "rgd_uks/MR_693393_1_0_0_1/orca.tar.zst"
        result = parse_rgd_reactivity_source(source)

        assert result["reaction_id"] == "MR_693393"
        assert result["reaction_pathway_id"] == 1
        assert result["reaction_step_idx"] == 0
        assert result["is_reactant"] == 1  # Step 0 is reactant
        assert result["is_product"] == 0

    def test_parse_rgd_source_product(self):
        """Test parsing RGD source where step is 18 (product)."""
        source = "rgd_uks/MR_693393_1_18_0_1/orca.tar.zst"
        result = parse_rgd_reactivity_source(source)

        assert result["reaction_id"] == "MR_693393"
        assert result["reaction_pathway_id"] == 1
        assert result["reaction_step_idx"] == 18
        assert result["is_reactant"] == 0
        assert result["is_product"] == 1  # Step 18 is product


class TestParseReactivityReactivitySource:
    """Test reactivity database source field parser."""

    def test_parse_pmechdb_source(self):
        """Test parsing pmechdb source format."""
        source = "pmechdb/r_12345_step3/file.tar"
        result = parse_reactivity_reactivity_source(source)

        assert result["subset"] == "pmechdb"
        assert result["reaction_id"] == "12345"
        assert result["reaction_pathway_id"] == 0
        assert result["reaction_step_idx"] == 3
        assert result["is_reactant"] is False
        assert result["is_product"] is False

    def test_parse_rmechdb_source(self):
        """Test parsing rmechdb source format."""
        source = "rmechdb/r_67890_step10/file.tar"
        result = parse_reactivity_reactivity_source(source)

        assert result["subset"] == "rmechdb"
        assert result["reaction_pathway_id"] == 0
        assert result["reaction_step_idx"] == 10
        assert result["is_reactant"] is False
        assert result["is_product"] is False

    def test_parse_ani1xbb_source(self):
        """Test parsing ani1xbb source format."""
        source = "ani1xbb/r_11111_5/file.tar"
        result = parse_reactivity_reactivity_source(source)

        assert result["subset"] == "ani1xbb"
        assert result["reaction_id"] == "11111"
        assert result["reaction_step_idx"] == 5


class TestParseAni2xReactivitySource:
    """Test ANI2X source field parser."""

    def test_parse_valid_ani2x_source(self):
        """Test parsing valid ANI2X source format."""
        source = "ani2x/r_12345_7/file.tar"
        result = parse_ani2x_reactivity_source(source)

        assert result["reaction_id"] == "12345"
        assert result["reaction_step_idx"] == 7

    def test_parse_ani2x_source_step_zero(self):
        """Test parsing ANI2X source with step 0."""
        source = "ani2x/r_99999_0/file.tar"
        result = parse_ani2x_reactivity_source(source)

        assert result["reaction_id"] == "99999"
        assert result["reaction_step_idx"] == 0


class TestParseTrans1xReactivitySource:
    """Test Trans1x source field parser."""

    def test_parse_trans1x_reactant(self):
        """Test parsing Trans1x source for reactant (step 0)."""
        source = "trans1x/r_12345_2_0/file.tar"
        result = parse_trans1x_reactivity_source(source)

        assert result["reaction_id"] == "12345"
        assert result["reaction_step_idx"] == 0
        assert result["reaction_pathway_id"] == 0
        assert result["is_reactant"] == 1
        assert result["is_product"] is None

    def test_parse_trans1x_product(self):
        """Test parsing Trans1x source for product (step 9)."""
        source = "trans1x/r_12345_2_9/file.tar"
        result = parse_trans1x_reactivity_source(source)

        assert result["reaction_id"] == "12345"
        assert result["reaction_step_idx"] == 10
        assert result["reaction_pathway_id"] == 0
        assert result["is_reactant"] is None
        assert result["is_product"] == 1

    def test_parse_trans1x_intermediate_step_pathway_0(self):
        """Test parsing Trans1x source for intermediate step in pathway 0."""
        source = "trans1x/r_12345_2_5/file.tar"
        result = parse_trans1x_reactivity_source(source)

        assert result["reaction_id"] == "12345"
        assert result["reaction_step_idx"] == 4  # (5 - 2) % 8 + 1 = 4
        assert result["reaction_pathway_id"] == 0
        assert result["is_reactant"] is None
        assert result["is_product"] is None

    def test_parse_trans1x_step_8(self):
        """Test parsing Trans1x source for step 8."""
        source = "trans1x/r_12345_2_8/file.tar"
        result = parse_trans1x_reactivity_source(source)

        assert result["reaction_id"] == "12345"
        assert result["reaction_step_idx"] == 7  # (8 - 2) % 8 + 1 = 7
        assert result["reaction_pathway_id"] == 0

    def test_parse_trans1x_pathway_1(self):
        """Test parsing Trans1x source for pathway 1 (step >= 10)."""
        source = "trans1x/r_12345_2_12/file.tar"
        result = parse_trans1x_reactivity_source(source)

        assert result["reaction_id"] == "12345"
        assert result["reaction_step_idx"] == 3  # (12 - 2) % 8 + 1 = 3
        assert result["reaction_pathway_id"] == 1  # (12 - 2) // 8 = 1
        assert result["is_reactant"] is None
        assert result["is_product"] is None

    def test_parse_trans1x_pathway_2(self):
        """Test parsing Trans1x source for pathway 2 (step >= 18)."""
        source = "trans1x/r_12345_2_20/file.tar"
        result = parse_trans1x_reactivity_source(source)

        assert result["reaction_id"] == "12345"
        assert result["reaction_step_idx"] == 3  # (20 - 2) % 8 + 1 = 3
        assert result["reaction_pathway_id"] == 2  # (20 - 2) // 8 = 2


class TestParseRealDatabaseEntries:
    """Integration tests using real database samples."""

    def test_parse_rgd_database_entries(self):
        """Test parsing real RGD database entries."""
        db_reader = SourceDatabaseReader(
            database_name="omol25",
            split_name="test",
            db_path=SAMPLES_DIR / "rgd_samples.aselmdb",
            database_format="aselmdb_omol",
        )

        entries_checked = 0
        for chunk in db_reader:
            for item in chunk:
                source = item.atoms.info.get("source")
                if source:
                    result = parse_rgd_reactivity_source(source)

                    # Verify all fields are present
                    assert "additional_subset_name" in result
                    assert "reaction_id" in result
                    assert "reaction_pathway_id" in result
                    assert "reaction_step_idx" in result
                    assert "is_reactant" in result
                    assert "is_product" in result

                    # Verify types when values are not None
                    if result["additional_subset_name"] is not None:
                        assert isinstance(result["additional_subset_name"], str)
                    if result["reaction_id"] is not None:
                        assert isinstance(result["reaction_id"], str)
                    if result["reaction_pathway_id"] is not None:
                        assert isinstance(result["reaction_pathway_id"], int)
                    if result["reaction_step_idx"] is not None:
                        assert isinstance(result["reaction_step_idx"], int)
                        # Step should be in valid range
                        assert 0 <= result["reaction_step_idx"] <= 18
                    if result["is_reactant"] is not None:
                        assert result["is_reactant"] in [0, 1]
                    if result["is_product"] is not None:
                        assert result["is_product"] in [0, 1]

                    entries_checked += 1
                    if entries_checked >= 5:
                        return

    def test_parse_reactivity_database_entries(self):
        """Test parsing real reactivity database entries."""
        db_reader = SourceDatabaseReader(
            database_name="omol25",
            split_name="test",
            db_path=SAMPLES_DIR / "reactivity_samples.aselmdb",
            database_format="aselmdb_omol",
        )

        entries_checked = 0
        for chunk in db_reader:
            for item in chunk:
                source = item.atoms.info.get("source")
                if source:
                    result = parse_reactivity_reactivity_source(source)

                    # Verify all fields are present
                    assert "subset" in result
                    assert "reaction_id" in result
                    assert "reaction_pathway_id" in result
                    assert "reaction_step_idx" in result
                    assert "is_reactant" in result
                    assert "is_product" in result

                    # Verify types
                    if result["subset"] is not None:
                        assert isinstance(result["subset"], str)
                    if result["reaction_id"]:
                        assert isinstance(result["reaction_id"], str)
                    assert isinstance(result["reaction_pathway_id"], int)
                    assert isinstance(result["reaction_step_idx"], int)
                    assert isinstance(result["is_reactant"], bool)
                    assert isinstance(result["is_product"], bool)

                    entries_checked += 1
                    if entries_checked >= 5:
                        return

    def test_parse_trans1x_database_entries(self):
        """Test parsing real Trans1x database entries."""
        db_reader = SourceDatabaseReader(
            database_name="omol25",
            split_name="test",
            db_path=SAMPLES_DIR / "trans1x_samples.aselmdb",
            database_format="aselmdb_omol",
        )

        entries_checked = 0
        for chunk in db_reader:
            for item in chunk:
                source = item.atoms.info.get("source")
                if source:
                    result = parse_trans1x_reactivity_source(source)

                    # Verify all fields are present
                    assert "reaction_id" in result
                    assert "reaction_step_idx" in result
                    assert "reaction_pathway_id" in result
                    assert "is_reactant" in result
                    assert "is_product" in result

                    # Verify types when values are not None
                    if result["reaction_id"] is not None:
                        assert isinstance(result["reaction_id"], str)
                    if result["reaction_step_idx"] is not None:
                        assert isinstance(result["reaction_step_idx"], int)
                        # Valid step range for Trans1x
                        assert 0 <= result["reaction_step_idx"] <= 10
                    if result["reaction_pathway_id"] is not None:
                        assert isinstance(result["reaction_pathway_id"], int)
                        assert result["reaction_pathway_id"] >= 0
                    if result["is_reactant"] is not None:
                        assert result["is_reactant"] in [0, 1]
                    if result["is_product"] is not None:
                        assert result["is_product"] in [0, 1]

                    entries_checked += 1
                    if entries_checked >= 5:
                        return

    def test_parse_ani2x_database_entries(self):
        """Test parsing real ANI2X database entries."""
        db_reader = SourceDatabaseReader(
            database_name="omol25",
            split_name="test",
            db_path=SAMPLES_DIR / "ani2x_samples.aselmdb",
            database_format="aselmdb_omol",
        )

        entries_checked = 0
        for chunk in db_reader:
            for item in chunk:
                source = item.atoms.info.get("source")
                if source:
                    result = parse_ani2x_reactivity_source(source)

                    # Verify all fields are present
                    assert "reaction_id" in result
                    assert "reaction_step_idx" in result

                    # Verify types when values are not None
                    if result["reaction_id"] is not None:
                        assert isinstance(result["reaction_id"], str)
                    if result["reaction_step_idx"] is not None:
                        assert isinstance(result["reaction_step_idx"], int)
                        assert result["reaction_step_idx"] >= 0

                    entries_checked += 1
                    if entries_checked >= 5:
                        return
