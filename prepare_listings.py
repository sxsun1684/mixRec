"""
prepare_listings.py

Stage 1 of the embedding pipeline:
Data exploration, cleaning, and multi-field text construction.

This script processes the Inside Airbnb listings dataset and performs
three core preprocessing steps before vectorization:

1. Data Exploration
   - Inspect dataset size and schema
   - Report missing-value rates for key retrieval fields
   - Validate data quality before indexing

2. Data Cleaning
   - Remove HTML and markdown artifacts from descriptions
   - Parse amenities from JSON arrays
   - Convert price strings into numeric values

3. Embedding Text Construction
   - Combine multiple listing attributes into a single semantic document
   - Generate a versioned `embed_text` field optimized for embedding models

Output:
    listings_prepared.csv

Each row contains:
    - Listing ID
    - Embedding text
    - Embedding version
    - Structured metadata used for filtering, ranking, and geo-search

The resulting `embed_text` field can be directly passed into an
embedding model to build a vector index for semantic retrieval.

Usage:
    python prepare_listings.py listings.csv
"""

import sys
import re
import json
import pandas as pd

# ------------------------------------------------------------------
# Versioning hook.
# Any modification to the text template or cleaning logic should
# trigger a version bump so embeddings and indexes can be rebuilt.
# ------------------------------------------------------------------
EMBED_TEXT_VERSION = "v1"

# Maximum number of amenities included in the embedding text.
# Excessively long amenity lists often dilute semantic relevance.
MAX_AMENITIES = 20


def clean_text(s):
    """
    Normalize free-text content.

    Operations:
        - Remove HTML tags
        - Remove markdown artifacts
        - Collapse repeated whitespace

    Returns:
        Clean plain-text string suitable for embedding.
    """
    if not isinstance(s, str):
        return ""

    s = re.sub(r"<[^>]+>", " ", s)
    s = s.replace("**", " ")
    s = re.sub(r"\s+", " ", s).strip()

    return s


def parse_amenities(s):
    """
    Parse amenities from a JSON string into a Python list.

    Example:
        '["Wifi","Kitchen"]'
            ->
        ["Wifi", "Kitchen"]

    Returns:
        list[str]
    """
    if not isinstance(s, str):
        return []

    try:
        items = json.loads(s)
        return [
            a.strip()
            for a in items
            if isinstance(a, str) and a.strip()
        ]
    except (json.JSONDecodeError, TypeError):
        return []


def parse_price(s):
    """
    Convert a formatted price string into a numeric value.

    Example:
        '$311.00' -> 311.0

    Returns:
        float | None
    """
    if not isinstance(s, str):
        return None

    cleaned = re.sub(r"[^\d.]", "", s)

    try:
        return float(cleaned) if cleaned else None
    except ValueError:
        return None


def build_embed_text(row):
    """
    Construct a semantic document for embedding generation.

    Multiple listing attributes are merged into a structured
    natural-language representation to improve retrieval quality.

    Included fields:
        - Title
        - Property type
        - Room type
        - Capacity
        - Neighborhood
        - Price
        - Amenities
        - Description

    English field labels are intentionally used because most
    embedding models are optimized for English semantic patterns.
    """
    parts = []

    # --------------------------------------------------------------
    # Title
    # --------------------------------------------------------------
    name = clean_text(row.get("name"))
    if name:
        parts.append(f"Title: {name}")

    # --------------------------------------------------------------
    # Property and room information
    # --------------------------------------------------------------
    property_type = row.get("property_type") or ""
    room_type = row.get("room_type") or ""
    accommodates = row.get("accommodates")

    type_parts = [
        str(x)
        for x in [property_type, room_type]
        if x and str(x) != "nan"
    ]

    type_line = " / ".join(type_parts)

    if pd.notna(accommodates):
        type_line += f", sleeps {int(accommodates)}"

    if type_line:
        parts.append(f"Type: {type_line}")

    # --------------------------------------------------------------
    # Neighborhood
    # --------------------------------------------------------------
    neighborhood = row.get("neighbourhood_cleansed")

    if isinstance(neighborhood, str) and neighborhood:
        parts.append(f"Neighborhood: {neighborhood}")

    # --------------------------------------------------------------
    # Price
    # --------------------------------------------------------------
    price = row.get("price_num")

    if price is not None:
        parts.append(f"Price: ${price:.0f} per night")

    # --------------------------------------------------------------
    # Amenities
    # --------------------------------------------------------------
    amenities = row.get("amenities_list") or []

    if amenities:
        parts.append(
            "Amenities: "
            + ", ".join(amenities[:MAX_AMENITIES])
        )

    # --------------------------------------------------------------
    # Description
    # --------------------------------------------------------------
    description = clean_text(row.get("description"))

    if description:
        parts.append(f"Description: {description}")

    return "\n".join(parts)


def main(path):
    """
    Execute the preprocessing pipeline.

    Steps:
        1. Load raw listing data
        2. Explore dataset quality
        3. Clean retrieval-related fields
        4. Generate embedding documents
        5. Export prepared dataset
    """

    # --------------------------------------------------------------
    # Load source dataset
    # --------------------------------------------------------------
    df = pd.read_csv(path)

    # ==============================================================
    # Stage 1: Data Exploration
    # Inspect dataset quality before transformation.
    # ==============================================================

    print(
        f"Total Rows: {len(df)}    "
        f"Total Columns: {len(df.columns)}\n"
    )

    key_fields = [
        "id",
        "name",
        "description",
        "neighbourhood_cleansed",
        "property_type",
        "room_type",
        "accommodates",
        "amenities",
        "price",
        "review_scores_rating",
        "latitude",
        "longitude",
    ]

    print(f'{"Field":<28}{"Missing Rate":>14}')
    print("-" * 42)

    for field in key_fields:
        if field in df.columns:
            missing_rate = 100 * df[field].isna().mean()
            print(
                f"{field:<28}"
                f"{missing_rate:>11.1f}%"
            )
        else:
            print(
                f"{field:<28}"
                f"{'<NOT FOUND>':>14}"
            )

    print()

    # ==============================================================
    # Stage 2: Data Cleaning
    # Normalize fields required by retrieval and ranking.
    # ==============================================================

    df["price_num"] = df["price"].apply(parse_price)

    df["amenities_list"] = (
        df["amenities"]
        .apply(parse_amenities)
    )

    # ==============================================================
    # Stage 3: Embedding Text Generation
    # Build semantic documents for vector indexing.
    # ==============================================================

    df["embed_text"] = df.apply(
        build_embed_text,
        axis=1
    )

    df["embed_version"] = EMBED_TEXT_VERSION

    # --------------------------------------------------------------
    # Fields retained for downstream retrieval.
    #
    # embed_text:
    #     Used for embedding generation and BM25 retrieval.
    #
    # structured metadata:
    #     Used for filtering, ranking, faceted search,
    #     geo constraints, and UI presentation.
    # --------------------------------------------------------------
    keep_columns = [
        "id",
        "embed_text",
        "embed_version",
        "neighbourhood_cleansed",
        "room_type",
        "property_type",
        "price_num",
        "accommodates",
        "bedrooms",
        "review_scores_rating",
        "number_of_reviews",
        "latitude",
        "longitude",
    ]

    keep_columns = [
        col
        for col in keep_columns
        if col in df.columns
    ]

    output_df = df[keep_columns].copy()

    output_path = "listings_prepared.csv"

    output_df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Prepared dataset written to "
        f"{output_path} "
        f"({len(output_df)} rows)\n"
    )

    # --------------------------------------------------------------
    # Preview one generated semantic document
    # --------------------------------------------------------------
    print("=" * 60)
    print("Sample embed_text (first listing)")
    print("=" * 60)

    print(
        df["embed_text"]
        .iloc[0][:600]
    )


if __name__ == "__main__":
    source_file = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "listings.csv"
    )

    main(source_file)