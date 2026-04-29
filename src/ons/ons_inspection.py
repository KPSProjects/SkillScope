import pandas as pd

from src.config import ONS_PATH


def inspect_ons_sheet(sheet_name: str, n_rows: int = 10) -> None:
    """
    Loads one ONS sheet and prints its basic structure.
    """
    print(f"\n--- Inspecting {sheet_name} ---")

    df = pd.read_excel(ONS_PATH, sheet_name=sheet_name)

    print(f"Shape: {df.shape}")
    print("\nColumns:")
    print(list(df.columns))

    print(f"\nFirst {n_rows} rows:")
    print(df.head(n_rows))


def run_ons_inspection():
    """
    Inspects the main usable ONS sheets.
    """
    for sheet_name in ["Table 1", "Table 2", "Table 3", "Table 4"]:
        inspect_ons_sheet(sheet_name)