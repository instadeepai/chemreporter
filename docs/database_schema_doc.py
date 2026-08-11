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
"""Render database schema documentation from table_schemas field definitions."""

from dataclasses import dataclass

from chemreporter.query_database_tools.table_schemas import (
    AMINO_ACID_CODES,
    AMINO_ACID_RESIDUE_SCHEMA,
    BASE_SCHEMA,
    CALCULATED_SCHEMA,
    CATALYST_SCHEMA,
    FINGERPRINTS_SCHEMA,
    GRAPH_PROP_SCHEMA,
    NUCLEOBASE_CODES,
    NUCLEOBASE_RESIDUE_SCHEMA,
    REACTION_SCHEMA,
    SchemaField,
)


@dataclass(frozen=True)
class SchemaDocSection:
    """A documentation section rendered from a schema field group."""

    title: str
    fields: tuple[SchemaField, ...]
    intro: str = ""


def format_schema_field_markdown(field: SchemaField) -> str:
    """Format one schema field as a Markdown bullet for documentation.

    Args:
        field: Schema field to format.

    Returns:
        Markdown list item describing the field.
    """
    description = " ".join(field.description.split())
    type_label = str(field.polars_type)
    if field.unit:
        type_label = f"{type_label}, {field.unit}"
    return f"- `{field.name}` ({type_label}): {description}"


def _format_motif_field_range(codes: tuple[str, ...]) -> str:
    names = ", ".join(f"`num_{code.lower()}`" for code in codes)
    return f"- {names} (Int64): RDKit substructure counts per motif code."


def render_database_schema_markdown() -> str:
    """Render the query-database schema documentation as Markdown.

    Returns:
        Markdown body with one section per schema group.
    """
    lines: list[str] = []

    for section in SCHEMA_DOC_SECTIONS:
        lines.append(f"## {section.title}")
        lines.append("")
        if section.intro:
            lines.append(section.intro)
            lines.append("")

        if section.title == "Molecular Fingerprints":
            lines.append(
                "- `fingerprint_0` to `fingerprint_1023` (Int64, 0 or 1): "
                "1024-bit Morgan fingerprint for similarity searches."
            )
            lines.append("")
            continue

        if section.title == "Bio motif counts":
            lines.append(_format_motif_field_range(AMINO_ACID_CODES))
            is_protein = next(
                field for field in section.fields if field.name == "is_protein"
            )
            lines.append(format_schema_field_markdown(is_protein))
            lines.append(_format_motif_field_range(NUCLEOBASE_CODES))
            is_nucleobase = next(
                field for field in section.fields if field.name == "is_nucleobase"
            )
            lines.append(format_schema_field_markdown(is_nucleobase))
            lines.append("")
            continue

        for field in section.fields:
            lines.append(format_schema_field_markdown(field))
        lines.append("")

    return "\n".join(lines).rstrip()


SCHEMA_DOC_SECTIONS = (
    SchemaDocSection("Core Fields", tuple(BASE_SCHEMA)),
    SchemaDocSection(
        "Computed Properties",
        tuple(CALCULATED_SCHEMA + GRAPH_PROP_SCHEMA),
    ),
    SchemaDocSection("Molecular Fingerprints", tuple(FINGERPRINTS_SCHEMA)),
    SchemaDocSection(
        "Bio motif counts",
        tuple(AMINO_ACID_RESIDUE_SCHEMA + NUCLEOBASE_RESIDUE_SCHEMA),
    ),
    SchemaDocSection(
        "Reaction Properties",
        tuple(REACTION_SCHEMA),
        intro=(
            "Present when extracted from OMOL25 ``source`` metadata for reactive "
            "subsets."
        ),
    ),
    SchemaDocSection(
        "Catalysis Properties",
        tuple(CATALYST_SCHEMA),
        intro="Present when extracted from OC20 supplementary metadata.",
    ),
)
