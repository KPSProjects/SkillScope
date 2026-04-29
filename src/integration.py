import pandas as pd

from src.config import (
    REED_SKILL_FREQUENCY_PATH,
    ONS_TABLE3_TOP_2025_PATH,
    ONS_TABLE2_TOP_2025_PATH,
    GOV_CATEGORY_COUNTS_PATH,
    COMBINED_SUMMARY_PATH,
)


def load_csv(path) -> pd.DataFrame:
    """
    Loads a CSV file from disk.
    """
    return pd.read_csv(path)


def get_reed_top_skills(limit: int = 10) -> pd.DataFrame:
    """
    Loads the Reed skill frequency summary and returns the top rows.
    """
    df = load_csv(REED_SKILL_FREQUENCY_PATH)
    return df.head(limit).copy()


def get_ons_table3_top(limit: int = 10) -> pd.DataFrame:
    """
    Loads the ONS Table 3 top 2025 summary and returns the top rows.
    """
    df = load_csv(ONS_TABLE3_TOP_2025_PATH)
    return df.head(limit).copy()


def get_ons_table2_top(limit: int = 10) -> pd.DataFrame:
    """
    Loads the ONS Table 2 top 2025 summary and returns the top rows.
    """
    df = load_csv(ONS_TABLE2_TOP_2025_PATH)
    return df.head(limit).copy()


def get_gov_top_categories(limit: int = 10) -> pd.DataFrame:
    """
    Loads the GOV category summary and returns the top rows.
    """
    df = load_csv(GOV_CATEGORY_COUNTS_PATH)
    return df.head(limit).copy()


def dataframe_to_text_block(title: str, df: pd.DataFrame) -> str:
    """
    Converts a dataframe into a readable text block.
    """
    lines = [f"{title}", "-" * len(title)]
    lines.append(df.to_string(index=False))
    lines.append("")
    return "\n".join(lines)


def build_combined_summary() -> str:
    """
    Builds one readable combined summary from the main project outputs.
    """
    reed_top = get_reed_top_skills()
    ons_table3_top = get_ons_table3_top()
    ons_table2_top = get_ons_table2_top()
    gov_top = get_gov_top_categories()

    sections = [
        "CAPSTONE PROJECT COMBINED SUMMARY",
        "=================================",
        "",
        dataframe_to_text_block("Top Reed Extracted Skills", reed_top),
        dataframe_to_text_block("Top ONS Table 3 Groups for 2025", ons_table3_top),
        dataframe_to_text_block("Top ONS Table 2 Groups for 2025", ons_table2_top),
        dataframe_to_text_block("Top GOV Job Categories", gov_top),
    ]

    return "\n".join(sections)


def save_combined_summary(summary_text: str) -> None:
    """
    Saves the combined summary to a text file.
    """
    COMBINED_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COMBINED_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write(summary_text)


def run_integration():
    """
    Main function for the integration layer.
    """
    print("Starting integration / comparison layer...")

    summary_text = build_combined_summary()

    print("\n--- COMBINED SUMMARY PREVIEW ---")
    print(summary_text[:3000])  # preview first part in terminal

    save_combined_summary(summary_text)

    print(f"\nSaved combined summary to: {COMBINED_SUMMARY_PATH}")