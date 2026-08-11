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
"""Pytest configuration and shared fixtures for chemreporter tests."""

from pathlib import Path

import pytest

# Test data directory
TESTS_DIR = Path(__file__).resolve().parent
FIRST_EXTXYZ_PATH = TESTS_DIR / "data" / "0.extxyz"


@pytest.fixture(scope="session")
def first_extxyz_path():
    """Path to the first example structure (0.extxyz)."""
    assert FIRST_EXTXYZ_PATH.exists(), f"Expected {FIRST_EXTXYZ_PATH} to exist"
    return FIRST_EXTXYZ_PATH
