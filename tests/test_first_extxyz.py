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
"""Tests using the first example 0.extxyz."""


def test_first_extxyz_exists(first_extxyz_path):
    """First example 0.extxyz exists and is readable."""
    assert first_extxyz_path.exists()
    text = first_extxyz_path.read_text()
    lines = text.strip().splitlines()
    assert len(lines) >= 2
    n_atoms = int(lines[0])
    assert n_atoms > 0
    # Header line with Lattice= and Properties=
    assert "Lattice=" in lines[1] or "Properties=" in lines[1]
    # Number of data lines should match n_atoms
    assert len(lines) >= 2 + n_atoms


def test_first_extxyz_has_expected_atom_count(first_extxyz_path):
    """First example has 76 atoms."""
    text = first_extxyz_path.read_text()
    n_atoms = int(text.splitlines()[0])
    assert n_atoms == 76
