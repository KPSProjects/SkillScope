import pandas as pd

from src.config import ONS_PATH, INTERIM_DIR


def clean_ons_standard_table(sheet_name: str) -> pd.DataFrame:
    """
    Cleans ONS Tables 1, 2, and 3.

    These tables have:
    - title / notes rows at the top
    - real column names on row index 5
    - actual data below that
    - one duplicated header row still left inside the data
    """
    # Load the raw sheet with no header yet
    raw_df = pd.read_excel(ONS_PATH, sheet_name=sheet_name, header=None)

    # Row 5 contains the real column names
    new_header = raw_df.iloc[5]

    # Actual data starts below the real header row
    cleaned_df = raw_df.iloc[6:].copy()
    cleaned_df.columns = new_header
    cleaned_df = cleaned_df.reset_index(drop=True)

    # Sometimes the header row appears again inside the data
    # This removes that duplicated header row
    first_column_name = cleaned_df.columns[0]
    cleaned_df = cleaned_df[
        cleaned_df[first_column_name] != first_column_name
    ].copy()

    cleaned_df = cleaned_df.reset_index(drop=True)

    # Clean column names so year columns become 2017 instead of 2017.0
    # This makes later analysis much easier
    cleaned_columns = []
    for col in cleaned_df.columns:
        col_name = str(col).strip()

        if col_name.endswith(".0"):
            col_name = col_name[:-2]

        cleaned_columns.append(col_name)

    cleaned_df.columns = cleaned_columns

    return cleaned_df


def inspect_ons_table4() -> pd.DataFrame:
    """
    Loads Table 4 only for inspection.
    It has a different structure, so it should not be cleaned the same way yet.
    """
    raw_df = pd.read_excel(ONS_PATH, sheet_name="Table 4", header=None)
    return raw_df


def save_ons_table(df: pd.DataFrame, sheet_name: str) -> None:
    """
    Saves cleaned ONS tables to the interim cleaned folder.
    """
    safe_name = sheet_name.lower().replace(" ", "_")
    output_path = INTERIM_DIR / "cleaned" / "ons_skills" / f"{safe_name}_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)

    print(f"Saved cleaned {sheet_name} to: {output_path}")


def run_ons_cleaning():
    """
    Cleans ONS Tables 1–3 and inspects Table 4 separately.
    """
    print("Starting ONS table cleaning...")

    # Clean the standard ONS tables first
    for sheet_name in ["Table 1", "Table 2", "Table 3"]:
        print(f"\n--- CLEANING {sheet_name} ---")

        cleaned_df = clean_ons_standard_table(sheet_name)

        print(f"Shape: {cleaned_df.shape}")
        print("\nColumns:")
        print(list(cleaned_df.columns))
        print("\nFirst 5 rows:")
        print(cleaned_df.head(5))

        save_ons_table(cleaned_df, sheet_name)

    # Table 4 is different, so only inspect it for now
    print("\n--- INSPECTING TABLE 4 SEPARATELY ---")
    table4_df = inspect_ons_table4()

    print(f"Shape: {table4_df.shape}")
    print("\nFirst 8 rows:")
    print(table4_df.head(8))