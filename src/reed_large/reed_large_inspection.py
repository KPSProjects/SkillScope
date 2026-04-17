import pandas as pd

from src.config import REED_UK_LARGE_PATH


def load_reed_uk_large():
    # load the larger Reed dataset
    return pd.read_csv(REED_UK_LARGE_PATH)


def print_basic_info(df):
    print("REED UK LARGE DATASET INSPECTION")
    print("================================")
    print(f"Row count: {len(df)}")
    print(f"Column count: {len(df.columns)}")
    print("\nColumns:")
    print(list(df.columns))


def print_missing_values(df):
    print("\n--- MISSING VALUES ---")
    missing = df.isna().sum().sort_values(ascending=False)
    print(missing)


def print_duplicate_info(df):
    print("\n--- DUPLICATE INFO ---")
    full_duplicates = df.duplicated().sum()
    print(f"Full duplicate rows: {full_duplicates}")

    useful_subset = [col for col in ["job_title", "job_description", "job_requirements"] if col in df.columns]
    if useful_subset:
        subset_duplicates = df.duplicated(subset=useful_subset).sum()
        print(f"Duplicates based on title/description/requirements: {subset_duplicates}")


def print_text_quality(df):
    print("\n--- TEXT QUALITY ---")

    text_columns = ["job_title", "job_description", "job_requirements"]

    for col in text_columns:
        if col in df.columns:
            series = df[col].dropna().astype(str)

            if len(series) == 0:
                print(f"\n{col}: no usable rows")
                continue

            lengths = series.str.len()

            print(f"\nColumn: {col}")
            print(f"Non-null rows: {len(series)}")
            print(f"Average length: {lengths.mean():.2f}")
            print(f"Median length: {lengths.median():.2f}")
            print(f"Min length: {lengths.min()}")
            print(f"Max length: {lengths.max()}")
            print(f"Rows over 30 chars: {(lengths > 30).sum()}")
            print(f"Rows over 100 chars: {(lengths > 100).sum()}")
            print(f"Rows over 300 chars: {(lengths > 300).sum()}")


def print_date_quality(df):
    print("\n--- DATE QUALITY ---")

    date_columns = ["post_date", "posting_date", "date_posted"]

    for col in date_columns:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors="coerce")
            valid_count = parsed.notna().sum()

            print(f"\nColumn: {col}")
            print(f"Valid dates: {valid_count}")
            print(f"Missing/invalid dates: {parsed.isna().sum()}")

            if valid_count > 0:
                print(f"Earliest date: {parsed.min()}")
                print(f"Latest date: {parsed.max()}")


def print_sample_rows(df, n=3):
    print("\n--- SAMPLE ROWS ---")

    cols_to_show = [col for col in ["job_title", "job_description", "job_requirements", "post_date", "category"] if col in df.columns]

    for i in range(min(n, len(df))):
        print(f"\nROW {i + 1}")
        row = df.iloc[i]

        for col in cols_to_show:
            value = str(row[col])[:800]
            print(f"\n{col}:")
            print(value)


def run_reed_large_inspection():
    print("Loading reed_uk_large dataset...")
    df = load_reed_uk_large()

    print_basic_info(df)
    print_missing_values(df)
    print_duplicate_info(df)
    print_text_quality(df)
    print_date_quality(df)
    print_sample_rows(df, n=3)