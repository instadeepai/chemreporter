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
"""Tests for sample_tools module."""

import logging

import polars as pl

from chemreporter.query_database_tools.sample_tools import (
    random_sampler,
)

logger = logging.getLogger("chemreporter")


class TestRamdomSampler:
    """Tests for ramdom_sampler function."""

    def test_ramdom_sampler_basic(self):
        """Test that ramdom_sampler samples the DataFrame."""
        data = {
            "entry_key": [f"mol_{i}" for i in range(20)],
        }
        df = pl.DataFrame(data)
        sampled = random_sampler(df, 10)
        assert len(sampled) == 10
        assert set(sampled.columns) == set(df.columns)

    def test_reproducibility_seed(self):
        """Test that ramdom_sampler is reproducible with a given seed."""
        data = {
            "entry_key": [f"mol_{i}" for i in range(20)],
        }
        df = pl.DataFrame(data)
        sampled = random_sampler(df, 10, seed=42)
        assert len(sampled) == 10
        assert set(sampled.columns) == set(df.columns)
        sampled2 = random_sampler(df, 10, seed=42)
        assert sampled.equals(sampled2)
