import re
import pandas as pd

from src.config import (
    REED_CLEANED_PATH,
    ESCO_SKILLS_CLEANED_PATH,
    REED_MATCHED_SKILLS_PATH,
)


def load_cleaned_reed() -> pd.DataFrame:
    """
    Loads the cleaned Reed dataset created in preprocessing.
    """
    return pd.read_csv(REED_CLEANED_PATH)


def load_cleaned_esco() -> pd.DataFrame:
    """
    Loads the cleaned ESCO skills dataset.
    """
    return pd.read_csv(ESCO_SKILLS_CLEANED_PATH)


def normalise_text(text: str) -> str:
    """
    Makes text easier to match by:
    - converting to lowercase
    - removing unusual symbols
    - cleaning extra spaces
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9+#/. -]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def split_alt_labels(alt_labels: str) -> list[str]:
    """
    ESCO alternative labels are stored in one field.
    This function splits them into a list.
    """
    if pd.isna(alt_labels) or not str(alt_labels).strip():
        return []

    split_labels = str(alt_labels).split("\n")
    cleaned_labels = [label.strip() for label in split_labels if label.strip()]

    return cleaned_labels


def is_valid_phrase(phrase: str) -> bool:
    """
    Basic filtering to remove very weak labels before matching.

    Rules:
    - ignore blank phrases
    - ignore phrases shorter than 3 characters
    - ignore phrases that are just numbers
    """
    if not phrase:
        return False

    phrase = phrase.strip()

    if len(phrase) < 3:
        return False

    if phrase.isdigit():
        return False

    return True


def build_esco_lookup(esco_df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a lookup table from ESCO using:
    - preferred labels
    - alternative labels

    Each row in the lookup table is one searchable phrase.
    """
    lookup_rows = []

    for _, row in esco_df.iterrows():
        concept_uri = row["concept_uri"]
        preferred_label = row["preferred_label"]
        skill_type = row["skill_type"]

        # Add preferred label
        if pd.notna(preferred_label):
            lookup_rows.append(
                {
                    "concept_uri": concept_uri,
                    "skill_type": skill_type,
                    "matched_phrase": preferred_label,
                    "preferred_label": preferred_label,
                    "label_source": "preferred",
                }
            )

        # Add alternative labels
        alt_labels = split_alt_labels(row["alt_labels"])
        for alt_label in alt_labels:
            lookup_rows.append(
                {
                    "concept_uri": concept_uri,
                    "skill_type": skill_type,
                    "matched_phrase": alt_label,
                    "preferred_label": preferred_label,
                    "label_source": "alt",
                }
            )

    lookup_df = pd.DataFrame(lookup_rows)

    # Normalise the phrases
    lookup_df["matched_phrase_norm"] = lookup_df["matched_phrase"].apply(normalise_text)

    # Remove weak / blank phrases
    lookup_df = lookup_df[lookup_df["matched_phrase_norm"].apply(is_valid_phrase)].copy()

    # Remove duplicates
    lookup_df = lookup_df.drop_duplicates(
        subset=["concept_uri", "matched_phrase_norm"]
    ).reset_index(drop=True)

    return lookup_df


def phrase_matches_text(phrase: str, text: str) -> bool:
    """
    Uses a stricter regex match so phrases are matched more cleanly
    than a simple substring check.
    """
    if not phrase or not text:
        return False

    pattern = rf"(?<!\w){re.escape(phrase)}(?!\w)"
    return re.search(pattern, text) is not None


def extract_skills_from_reed(reed_df: pd.DataFrame, esco_lookup_df: pd.DataFrame) -> pd.DataFrame:
    """
    Baseline rule-based extraction.

    For each Reed job advert:
    - combine title + description
    - normalise the text
    - check ESCO phrases against the advert text

    This version is stricter than the first attempt and removes
    some obvious false positives.
    """
    matched_rows = []

    for _, job_row in reed_df.iterrows():
        job_id = job_row["job_id"]

        job_title = "" if pd.isna(job_row["job_title"]) else str(job_row["job_title"])
        job_description = "" if pd.isna(job_row["job_description"]) else str(job_row["job_description"])

        combined_text = normalise_text(job_title + " " + job_description)

        # Track skills already matched for this job
        matched_skill_keys = set()

        for _, skill_row in esco_lookup_df.iterrows():
            phrase = skill_row["matched_phrase_norm"]

            if phrase_matches_text(phrase, combined_text):
                skill_key = (job_id, skill_row["concept_uri"])

                # Only keep one match per skill per job
                if skill_key not in matched_skill_keys:
                    matched_rows.append(
                        {
                            "job_id": job_id,
                            "job_title": job_row["job_title"],
                            "concept_uri": skill_row["concept_uri"],
                            "preferred_label": skill_row["preferred_label"],
                            "matched_phrase": skill_row["matched_phrase"],
                            "label_source": skill_row["label_source"],
                            "skill_type": skill_row["skill_type"],
                        }
                    )
                    matched_skill_keys.add(skill_key)

    matches_df = pd.DataFrame(matched_rows)

    return matches_df


def save_matches(df: pd.DataFrame, path) -> None:
    """
    Saves matched skills output to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_extraction():
    """
    Main function for the baseline extraction stage.
    """
    print("Starting baseline skill extraction...")

    reed_df = load_cleaned_reed()
    esco_df = load_cleaned_esco()

    print(f"Loaded cleaned Reed rows: {len(reed_df)}")
    print(f"Loaded cleaned ESCO rows: {len(esco_df)}")

    esco_lookup_df = build_esco_lookup(esco_df)
    print(f"Built ESCO lookup rows: {len(esco_lookup_df)}")

    matches_df = extract_skills_from_reed(reed_df, esco_lookup_df)

    print("\n--- MATCHED SKILLS SAMPLE ---")
    print(matches_df.head(10))

    print(f"\nTotal matches found: {len(matches_df)}")

    if not matches_df.empty:
        print("\nTop matched preferred labels:")
        print(matches_df["preferred_label"].value_counts().head(10))

    save_matches(matches_df, REED_MATCHED_SKILLS_PATH)
    print(f"\nSaved matched skills file to: {REED_MATCHED_SKILLS_PATH}")