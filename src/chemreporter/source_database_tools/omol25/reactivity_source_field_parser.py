"""Script to extract and analyze meaningful information from source fields (OMOL25).

parses the source field from OMOL25 database entries to extract
reactivity information like reaction IDs, pathway IDs, and reaction steps.

"""

import logging

logger = logging.getLogger("chemreporter")


def parse_rgd_reactivity_source(source: str) -> dict[str, str | int | None]:
    """Parse RGD source field to extract meaningful information.

    Example RGD source format:
        'rgd_uks/MR_693393_1_13_0_1/orca.tar.zst'

    Format breakdown for MR_693393_1_13_0_1:
        - MR_693393_Z: molecule/reaction ID
        - 1: reaction number 1
        - 13: reaction number 2 (step)
        - 0: charge
        - 1: conformer

    Args:
        source: The source field string

    Returns:
        Dictionary with extracted information (all values are HDF5-compatible types):
            - dataset: str or None - Dataset name (e.g., 'rgd_uks')
            - reaction_id: str or None - Molecule identifier (e.g., 'MR_693393')
            - pathway_step: int or None - Reaction step number (e.g., 13)
            - TS: bool or None - Boolean indicating if it's a transition state
    """
    info: dict[str, str | int | None] = {
        "additional_subset_name": None,
        "reaction_id": None,
        "reaction_pathway_id": None,
        "reaction_step_idx": None,
        "is_reactant": None,
    }

    # Split path by '/'
    parts = source.split("/")

    if len(parts) < 2:
        return info

    # Extract dataset (always a string)
    info["additional_subset_name"] = str(parts[0])
    full_id = parts[1]

    # Parse pattern like: MR_693393_1_13_0_1
    id_parts = full_id.split("_")

    # Need at least 5 parts for RGD format
    if len(id_parts) < 5:
        return info

    # Validate that the reaction step part is numeric before converting
    reaction_step_str = id_parts[-3]
    if not reaction_step_str.isdigit():
        return info

    # Extract reaction step (third from last) - must be an int
    reaction_step = int(reaction_step_str)
    info["reaction_step_idx"] = reaction_step

    # Everything else is molecule ID - must be a string
    info["reaction_id"] = str("_".join(id_parts[:-4]))

    # Determine if it's a reactant - stored as int (0 or 1)
    # Steps 0 and 18 are not transition states
    info["is_reactant"] = int(reaction_step == 0)
    info["is_product"] = int(reaction_step == 18)
    info["reaction_pathway_id"] = int(id_parts[-4])

    return info


def parse_reactivity_reactivity_source(
    source: str,
) -> dict[str, str | int | None]:
    """Parse RGD source field to extract meaningful information.

    Example RGD source format:
        'rgd_uks/MR_693393_1_13_0_1/orca.tar.zst'

    Format breakdown for MR_693393_1_13_0_1:
        - MR_693393_Z: molecule/reaction ID
        - 1: reaction number 1
        - 13: reaction number 2 (step)
        - 0: charge
        - 1: conformer

    Args:
        source: The source field string

    Returns:
        Dictionary with extracted information (all values are HDF5-compatible types):
            - dataset: str or None - Dataset name (e.g., 'rgd_uks')
            - reaction_id: str or None - Molecule identifier (e.g., 'MR_693393')
            - pathway_step: int or None - Reaction step number (e.g., 13)
            - TS: bool or None - Boolean indicating if it's a transition state
    """
    info: dict[str, str | int | None] = {
        "subset": None,
        "reaction_id": "",
        "reaction_pathway_id": 0,
        "reaction_step_idx": 0,
        "is_reactant": False,
        "is_product": False,
    }

    # Split path by '/'
    parts = source.split("/")

    if len(parts) < 2:
        return info

    # Extract dataset (always a string)
    info["subset"] = str(parts[0])
    subset = info["subset"]
    full_id = parts[1]

    # Parse pattern like: r_12345_step3 or MR_693393_1_13_0_1
    id_parts = full_id.split("_")

    # Handle subset-specific formats first (they may have fewer parts)
    if subset in ["pmechdb", "rmechdb"]:
        if len(id_parts) >= 3:
            info["reaction_id"] = id_parts[1]
            info["reaction_pathway_id"] = 0
            info["reaction_step_idx"] = int(id_parts[2].strip("step"))
            info["is_reactant"] = False
            info["is_product"] = False
        return info
    elif subset == "ani1xbb":
        if len(id_parts) >= 3:
            info["reaction_id"] = id_parts[1]
            info["reaction_step_idx"] = int(id_parts[2])
        return info

    # For other formats, need at least 5 parts
    if len(id_parts) < 5:
        return info

    return info


def parse_ani2x_reactivity_source(source: str) -> dict[str, str | int | None]:
    """Parse Ani source field to extract info.

    Args:
        source: The source field string.

    Returns:
        Dictionary with extracted information.
    """
    info: dict[str, str | int | None] = {
        "reaction_id": None,
    }

    parts = source.split("/")[1].split("_")

    if len(parts) < 2:
        return info

    info["reaction_id"] = parts[1]
    info["reaction_step_idx"] = int(parts[2])
    return info


def parse_trans1x_reactivity_source(
    source: str,
) -> dict[str, str | int | None]:
    """Parse Trans1x source field to extract info.

    Args:
        source: The source field string.

    Returns:
        Dictionary with extracted information.
    """
    info: dict[str, str | int | None] = {
        "reaction_id": None,
        "reaction_step_idx": None,
        "reaction_pathway_id": None,
        "is_reactant": None,
        "is_product": None,
    }

    # Split path by '/'
    parts = source.split("/")[1].split("_")

    if len(parts) < 2:
        return info

    info["reaction_id"] = parts[1]
    _number_of_pathways = parts[2]
    if int(parts[3]) not in [0, 9]:
        info["reaction_step_idx"] = (int(parts[3]) - 2) % 8 + 1

    elif int(parts[3]) == 9:
        info["reaction_step_idx"] = 10
        info["is_product"] = 1
    elif int(parts[3]) == 0:
        info["reaction_step_idx"] = 0
        info["is_reactant"] = 1

    if int(parts[3]) < 10:
        info["reaction_pathway_id"] = 0
    else:
        info["reaction_pathway_id"] = (int(parts[3]) - 2) // 8

    return info


def parse_metal_complexes_reactivity_source(
    source: str, reference_source: str | None
) -> dict[str, str | int | None]:
    """Parse Metal Complexes source field to extract info.

    Args:
        source: The source field string.
        reference_source: The reference source field string.

    Returns:
        Dictionary with extracted information.
    """
    info: dict[str, str | int | None] = {
        "subset": None,
    }
    try:
        if reference_source is None:
            if "tm_react" in source:
                info = {
                    "subset": "reactivity_metal_complexes",
                    "reaction_step_idx": int(
                        source.split("/")[1].split("step")[1].split("_")[0]
                    ),
                    "reaction_id": source.split("/")[1]
                    .split("step")[0]
                    .replace("_", ""),
                    "source": source,
                }
                return info
            else:
                info = {"subset": "ground_state_metal_complexes"}

        else:
            if "step" in reference_source:
                if "incomplete" in reference_source or "failed" in reference_source:
                    return {"subset": "failed_metal_complexes"}
                info = {
                    "subset": "ground_state_metal_complexes",
                    "source": source + "+" + reference_source,
                }

            else:
                info = {"subset": "ground_state_metal_complexes"}
    except (IndexError, ValueError) as e:
        logger.debug("Failed to parse metal complexes source %r: %s", source, e)
    return info
