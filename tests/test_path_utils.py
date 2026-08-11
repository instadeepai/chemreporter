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
"""Tests for the shared path helpers."""

from pathlib import Path

import pytest
from cloudpathlib import AnyPath, AzureBlobPath, GSPath, S3Path

from chemreporter.path_utils import is_local_path


@pytest.mark.parametrize(
    "path",
    [
        Path("/tmp/local/db.aselmdb"),
        Path("relative/local/db.aselmdb"),
        AnyPath("/tmp/local/db.aselmdb"),
        "/tmp/local/db.aselmdb",
        "relative/local/db.aselmdb",
    ],
    ids=[
        "posix-absolute",
        "posix-relative",
        "anypath-local",
        "str-absolute",
        "str-relative",
    ],
)
def test_is_local_path_true_for_local_paths(path):
    """Local pathlib / AnyPath / string values are treated as local."""
    assert is_local_path(path) is True


@pytest.mark.parametrize(
    "path",
    [
        S3Path("s3://bucket/prefix/db.aselmdb"),
        S3Path("s3://bucket/"),
        GSPath("gs://bucket/prefix/db.aselmdb"),
        GSPath("gs://bucket/"),
        AzureBlobPath("az://account/container/prefix/db.aselmdb"),
        AzureBlobPath("az://account/container/"),
        AnyPath("s3://bucket/prefix/db.aselmdb"),
        AnyPath("gs://bucket/prefix/db.aselmdb"),
        AnyPath("az://account/container/prefix/db.aselmdb"),
        "s3://bucket/prefix/db.aselmdb",
        "gs://bucket/prefix/db.aselmdb",
    ],
    ids=[
        "s3-file",
        "s3-prefix",
        "gs-file",
        "gs-prefix",
        "azure-file",
        "azure-prefix",
        "anypath-s3",
        "anypath-gs",
        "anypath-azure",
        "str-s3",
        "str-gs",
    ],
)
def test_is_local_path_false_for_cloud_paths(path):
    """Cloudpathlib S3 / GCS / Azure paths are treated as non-local."""
    assert is_local_path(path) is False
