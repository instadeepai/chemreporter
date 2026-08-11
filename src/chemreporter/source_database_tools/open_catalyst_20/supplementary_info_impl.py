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
import lzma
import pickle
import tempfile
from pathlib import Path
from typing import Any, Callable

import polars as pl

OC20_DATA_MAPPING_FILE = "oc20_data_mapping.pkl"


def sidecar_path(file_path: Path) -> Path:
    """Resolve OC20 sidecar. txt.

    Args:
        file_path: The path to the file.

    Returns:
        The path to the sidecar.
    """
    if file_path.name.endswith(".extxyz.xz"):
        return file_path.with_suffix("").with_suffix(".txt.xz")
    if file_path.name.endswith(".extxyz"):
        return file_path.with_suffix(".txt")
    if file_path.name.endswith(".aselmdb"):
        return file_path.with_suffix(".txt")
    return file_path


def read_supplementary_info_txt(supp_file: Path) -> list[dict[str, Any]]:
    """Parse OC20 sidecar CSV rows (sid, frame_number, ref_energy).

    Returns:
        One dict per frame, in order. Empty list if the file is missing.

    """
    if not supp_file or not supp_file.exists():
        return []

    columns = ["sid", "frame_number", "ref_energy"]
    if str(supp_file).endswith(".xz"):
        with lzma.open(supp_file, mode="rt", encoding="utf-8") as f:
            content = f.read()
    else:
        with open(supp_file, encoding="utf-8") as f:
            content = f.read()
    columns = ["sid", "frame_number", "ref_energy"]
    return pl.read_csv(
        content.encode(),
        separator=",",
        has_header=False,
        new_columns=columns,
    ).to_dicts()


def load_oc20_supplementary_info(
    files_dir: Path,
    files_index: dict[int, Path],
    download_function: Callable[..., Any] | None,
    mapping_filename: str = OC20_DATA_MAPPING_FILE,
) -> dict[str, dict[int, dict[str, Any]]]:
    """Load pickle mapping + per-shard sidecar txts into a nested dict.

    Returns:
        all supplementary info for all entries in all files,
        indexed by filename and entry index
        dict[filename, dict[entry_index, info_dict]]

    Raises:
        ValueError: If a sid from a sidecar row is missing from the pickle map.
    """
    supplementary_info: dict[str, dict[int, dict[str, Any]]] = {}
    mapping_file_path = files_dir / mapping_filename

    with tempfile.TemporaryDirectory() as temp_dir:
        if download_function:
            temp_pickle = Path(temp_dir) / mapping_file_path.name
            download_function(mapping_file_path, temp_pickle)
            local_mapping_file_path = temp_pickle
        else:
            local_mapping_file_path = mapping_file_path
        with open(local_mapping_file_path, "rb") as f:
            pickle_info = pickle.load(f)

        for _, file_path in files_index.items():
            shard_name = file_path.name
            supplementary_info[shard_name] = {}
            txt_file = sidecar_path(file_path)
            if download_function:
                temp_txt = Path(temp_dir) / txt_file.name
                download_function(txt_file, temp_txt)
                local_file_path = temp_txt
            else:
                local_file_path = txt_file
            txt_info = read_supplementary_info_txt(local_file_path)
            for i, entry in enumerate(txt_info):
                sid = entry["sid"]
                row = pickle_info.get(sid, False)
                if not row:
                    raise ValueError(
                        f"Supplementary info not found for sid='{sid}' "
                        f"in file '{shard_name}' frame {i}.\n"
                        f"Total pickle keys: {len(pickle_info)}"
                    )
                supplementary_info[shard_name][i] = dict(row)
                supplementary_info[shard_name][i]["ref_energy"] = entry["ref_energy"]
                supplementary_info[shard_name][i]["frame_number"] = entry[
                    "frame_number"
                ]

    return supplementary_info
