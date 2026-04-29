import pandas as pd

from src.config import (
    REED_UK_LARGE_PATH,
    REED_UK_LARGE_CLEANED_PATH,
    REED_UK_LARGE_PREVIEW_PATH,
)


def load_reed_large():
    # load the large Reed dataset
    return pd.read_csv(REED_UK_LARGE_PATH)


def fix_bad_text(text):
    # fix common broken encoding issues from scraped text
    if pd.isna(text):
        return text

    text = str(text)

    replacements = {
        "â€™": "'",
        "â€˜": "'",
        "â€œ": '"',
        "â€�": '"',
        "â€“": "-",
        "â€”": "-",
        "Â£": "£",
        "Â ": " ",
        "â€¢": "•",
        "Ã—": "x",
    }

    for bad, good in replacements.items():
        text = text.replace(bad, good)

    # clean extra spaces
    text = " ".join(text.split())
    return text.strip()


def clean_text_columns(df):
    # fix messy encoding in key text columns
    text_columns = [
        "job_title",
        "job_description",
        "job_requirements",
        "company_name",
        "category",
        "city",
        "state",
        "geo",
        "job_type",
        "salary_offered",
    ]

    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].apply(fix_bad_text)

    return df


def standardise_dates(df):
    # convert post_date into proper datetime then back to clean string
    if "post_date" in df.columns:
        parsed = pd.to_datetime(df["post_date"], errors="coerce")
        df["post_date"] = parsed.dt.strftime("%Y-%m-%d")

    return df


def keep_useful_columns(df):
    # keep the columns we actually care about
    useful_columns = [
        "job_title",
        "job_description",
        "job_requirements",
        "post_date",
        "category",
        "company_name",
        "city",
        "state",
        "geo",
        "job_type",
        "salary_offered",
        "job_board",
    ]

    keep_cols = [col for col in useful_columns if col in df.columns]
    return df[keep_cols].copy()


def remove_duplicates(df):
    # remove full duplicates first
    before_full = len(df)
    df = df.drop_duplicates().copy()
    after_full = len(df)

    # then remove duplicates based on main advert text fields
    subset_cols = [col for col in ["job_title", "job_description", "job_requirements"] if col in df.columns]
    before_subset = len(df)
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols).copy()
    after_subset = len(df)

    stats = {
        "before_full": before_full,
        "after_full": after_full,
        "before_subset": before_subset,
        "after_subset": after_subset,
        "full_removed": before_full - after_full,
        "subset_removed": before_subset - after_subset,
    }

    return df, stats


def save_preview(df_before, df_after, stats):
    # save a readable summary of the cleaning step
    REED_UK_LARGE_PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(REED_UK_LARGE_PREVIEW_PATH, "w", encoding="utf-8") as file:
        file.write("REED UK LARGE CLEANING PREVIEW\n")
        file.write("==============================\n\n")

        file.write(f"Rows before cleaning: {len(df_before)}\n")
        file.write(f"Rows after cleaning: {len(df_after)}\n\n")

        file.write("DUPLICATE REMOVAL\n")
        file.write("-----------------\n")
        file.write(f"Full duplicates removed: {stats['full_removed']}\n")
        file.write(f"Subset duplicates removed: {stats['subset_removed']}\n\n")

        file.write("DATE RANGE\n")
        file.write("----------\n")
        if "post_date" in df_after.columns:
            valid_dates = pd.to_datetime(df_after["post_date"], errors="coerce").dropna()
            if not valid_dates.empty:
                file.write(f"Earliest post date: {valid_dates.min()}\n")
                file.write(f"Latest post date: {valid_dates.max()}\n\n")

        file.write("SAMPLE CLEANED ROWS\n")
        file.write("-------------------\n")
        for i in range(min(3, len(df_after))):
            row = df_after.iloc[i]
            file.write(f"\nROW {i + 1}\n")
            for col in ["job_title", "job_description", "job_requirements", "post_date", "category"]:
                if col in df_after.columns:
                    file.write(f"\n{col}:\n")
                    file.write(f"{str(row[col])[:800]}\n")


def run_clean_reed_large():
    print("Cleaning reed_uk_large dataset...")

    df = load_reed_large()
    print(f"Rows loaded: {len(df)}")

    df = keep_useful_columns(df)
    print(f"Columns kept: {len(df.columns)}")

    df = clean_text_columns(df)
    df = standardise_dates(df)

    cleaned_df, stats = remove_duplicates(df)

    print("\n--- CLEANING SUMMARY ---")
    print(f"Rows after cleaning: {len(cleaned_df)}")
    print(f"Full duplicates removed: {stats['full_removed']}")
    print(f"Subset duplicates removed: {stats['subset_removed']}")

    REED_UK_LARGE_CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_csv(REED_UK_LARGE_CLEANED_PATH, index=False)

    save_preview(df, cleaned_df, stats)

    print(f"\nSaved cleaned Reed dataset to: {REED_UK_LARGE_CLEANED_PATH}")
    print(f"Saved cleaning preview to: {REED_UK_LARGE_PREVIEW_PATH}")