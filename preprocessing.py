import pandas as pd

from src.config import (
    REED_CLEANED_PATH,
    GOV_CLEANED_PATH,
    ESCO_SKILLS_CLEANED_PATH,
)
from src.loaders import load_reed, load_gov_cleaned, load_esco_skills


def clean_reed(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the key Reed columns and standardise types."""
    columns_to_keep = [
        "uniq_id",
        "job_title",
        "job_description",
        "post_date",
        "category",
        "company_name",
        "city",
        "state",
        "country",
        "job_type",
        "salary_offered",
    ]

    df = df[columns_to_keep].copy()
    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")

    df.columns = [
        "job_id",
        "job_title",
        "job_description",
        "post_date",
        "category",
        "company_name",
        "city",
        "state",
        "country",
        "job_type",
        "salary_offered",
    ]

    return df


def clean_esco_skills(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the core ESCO columns needed for skill matching."""
    columns_to_keep = [
        "conceptUri",
        "skillType",
        "preferredLabel",
        "altLabels",
        "hiddenLabels",
        "description",
    ]

    df = df[columns_to_keep].copy()

    df.columns = [
        "concept_uri",
        "skill_type",
        "preferred_label",
        "alt_labels",
        "hidden_labels",
        "description",
    ]

    return df


def print_missing_summary(name: str, df: pd.DataFrame) -> None:
    print(f"\nMissing values summary for {name}:")
    print(df.isna().sum())


def save_dataframe(df: pd.DataFrame, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_preprocessing():
    print("Starting preprocessing...")

    reed_df = load_reed()
    gov_df = load_gov_cleaned()   # already cleaned/standardised
    esco_df = load_esco_skills()

    reed_cleaned = clean_reed(reed_df)
    gov_cleaned = gov_df.copy()
    esco_cleaned = clean_esco_skills(esco_df)

    print("\n--- CLEANED REED ---")
    print(reed_cleaned.head(3))
    print_missing_summary("Reed", reed_cleaned)

    print("\n--- CLEANED GOV ---")
    print(gov_cleaned.head(3))
    print_missing_summary("GOV", gov_cleaned)

    print("\n--- CLEANED ESCO ---")
    print(esco_cleaned.head(3))
    print_missing_summary("ESCO", esco_cleaned)

    save_dataframe(reed_cleaned, REED_CLEANED_PATH)
    save_dataframe(gov_cleaned, GOV_CLEANED_PATH)
    save_dataframe(esco_cleaned, ESCO_SKILLS_CLEANED_PATH)

    print("\nPreprocessing complete.")
    print(f"Saved Reed cleaned file to: {REED_CLEANED_PATH}")
    print(f"Saved GOV cleaned file to: {GOV_CLEANED_PATH}")
    print(f"Saved ESCO cleaned file to: {ESCO_SKILLS_CLEANED_PATH}")