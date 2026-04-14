import os
import pandas as pd
import matplotlib.pyplot as plt


def convert_table_2_to_long():
    """
    Convert cleaned ONS Table 2 from wide format to long format.

    Input:
    - table_2_cleaned.csv

    Output:
    - table_2_long.csv
    """
    print("Starting ONS Table 2 analysis...")

    input_file = "data/interim/cleaned/ons_skills/table_2_cleaned.csv"
    output_file = "data/interim/aggregated/ons_skills/table_2_long.csv"

    # Make sure the input file exists before trying to load it
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find file: {input_file}")

    # Create the output folder if it does not already exist
    os.makedirs("data/interim/aggregated/ons_skills", exist_ok=True)

    # Load the cleaned Table 2 CSV
    df = pd.read_csv(input_file)

    print(f"Loaded cleaned Table 2 rows: {len(df)}")
    print("\n--- CLEANED TABLE 2 COLUMNS ---")
    print(df.columns.tolist())

    # Rename long ONS column names to shorter names
    # This makes the later code easier to read
    df = df.rename(columns={
        "SCO least detailed level code": "least_code",
        "SCO least detailed level label": "least_label",
        "SCO middle level code": "middle_code",
        "SCO middle level label": "middle_label"
    })

    # Find all year columns such as 2017, 2018, 2019...
    year_columns = []
    for col in df.columns:
        if str(col).isdigit():
            year_columns.append(col)

    print("\nDetected year columns:")
    print(year_columns)

    # Keep only the columns we actually need
    df = df[["least_code", "least_label", "middle_code", "middle_label"] + year_columns].copy()

    # Convert year values to numeric
    # Any invalid values will become NaN
    for year in year_columns:
        df[year] = pd.to_numeric(df[year], errors="coerce")

    # Convert the table from wide format to long format
    # This makes it easier to analyse and chart later
    long_df = df.melt(
        id_vars=["least_code", "least_label", "middle_code", "middle_label"],
        value_vars=year_columns,
        var_name="year",
        value_name="value"
    )

    # Convert year and value columns to numeric
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    # Remove rows where year or value is missing
    long_df = long_df.dropna(subset=["year", "value"])

    # Make year integers instead of floats
    long_df["year"] = long_df["year"].astype(int)

    # Save the finished long-format version
    long_df.to_csv(output_file, index=False)

    print("\n--- TABLE 2 LONG FORMAT SAMPLE ---")
    print(long_df.head())

    print("\n--- TABLE 2 LONG FORMAT INFO ---")
    print(f"Unique middle groups: {long_df['middle_label'].nunique()}")
    print(f"Unique least-detailed groups: {long_df['least_label'].nunique()}")
    print(f"Years: {sorted(long_df['year'].unique().tolist())}")
    print(f"Rows: {len(long_df)}")

    print(f"\nSaved long format Table 2 to: {output_file}")


def create_table_2_top_2025_summary():
    """
    Create a 2025 summary from Table 2 long-format data.

    Input:
    - table_2_long.csv

    Output:
    - table_2_top_2025.csv
    """
    input_file = "data/interim/aggregated/ons_skills/table_2_long.csv"
    output_file = "data/interim/aggregated/ons_skills/table_2_top_2025.csv"

    # Check the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find file: {input_file}")

    # Load the long-format Table 2 data
    df = pd.read_csv(input_file)

    # Keep only the rows for year 2025
    df_2025 = df[df["year"] == 2025].copy()

    # Sort highest to lowest
    df_2025 = df_2025.sort_values(by="value", ascending=False)

    # Save the 2025 summary
    df_2025.to_csv(output_file, index=False)

    print("\n--- TABLE 2 TOP 2025 SUMMARY ---")
    print(df_2025.head(15))

    print(f"\nSaved top 2025 summary to: {output_file}")


def create_table_2_top_2025_chart():
    """
    Create a bar chart for the top 10 Table 2 groups in 2025.

    Input:
    - table_2_top_2025.csv

    Output:
    - table_2_top_10_2025_chart.png
    """
    input_file = "data/interim/aggregated/ons_skills/table_2_top_2025.csv"
    output_file = "data/processed/ons_skills/table_2_top_10_2025_chart.png"

    # Check the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find file: {input_file}")

    # Create the output folder if needed
    os.makedirs("data/processed/ons_skills", exist_ok=True)

    # Load the 2025 summary
    df = pd.read_csv(input_file)

    # Take only the top 10 rows
    top_10 = df.head(10).copy()

    # Create the chart
    plt.figure(figsize=(12, 6))
    plt.barh(top_10["middle_label"], top_10["value"])
    plt.xlabel("Value")
    plt.ylabel("Middle Level Skill Group")
    plt.title("Top 10 ONS Table 2 Skill Groups in 2025")

    # Invert the y-axis so the highest value is shown at the top
    plt.gca().invert_yaxis()

    # Improve spacing so labels fit better
    plt.tight_layout()

    # Save the chart
    plt.savefig(output_file)
    plt.close()

    print(f"\nSaved Table 2 top 10 chart to: {output_file}")


def convert_table_3_to_long():
    """
    Convert cleaned ONS Table 3 from wide format to long format.

    Input:
    - table_3_cleaned.csv

    Output:
    - table_3_long.csv
    """
    print("Starting ONS Table 3 analysis...")

    input_file = "data/interim/cleaned/ons_skills/table_3_cleaned.csv"
    output_file = "data/interim/aggregated/ons_skills/table_3_long.csv"

    # Make sure the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find file: {input_file}")

    # Load cleaned Table 3 CSV
    df = pd.read_csv(input_file)

    print(f"Loaded cleaned Table 3 rows: {len(df)}")
    print("\n--- CLEANED TABLE 3 COLUMNS ---")
    print(df.columns.tolist())

    # Rename long column names to shorter names
    df = df.rename(columns={
        "SCO least detailed level code": "least_code",
        "SCO least detailed level label": "least_label"
    })

    # Find the year columns
    year_columns = []
    for col in df.columns:
        if str(col).isdigit():
            year_columns.append(col)

    print("\nDetected year columns:")
    print(year_columns)

    # Keep only the useful columns
    df = df[["least_code", "least_label"] + year_columns].copy()

    # Convert year values to numeric
    for year in year_columns:
        df[year] = pd.to_numeric(df[year], errors="coerce")

    # Convert wide format into long format
    long_df = df.melt(
        id_vars=["least_code", "least_label"],
        value_vars=year_columns,
        var_name="year",
        value_name="value"
    )

    # Convert year and value columns to numeric
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")

    # Remove rows where year or value is missing
    long_df = long_df.dropna(subset=["year", "value"])

    # Make year integer
    long_df["year"] = long_df["year"].astype(int)

    # Save the long-format version
    long_df.to_csv(output_file, index=False)

    print("\n--- TABLE 3 LONG FORMAT SAMPLE ---")
    print(long_df.head())

    print("\n--- TABLE 3 LONG FORMAT INFO ---")
    print(f"Unique least-detailed groups: {long_df['least_label'].nunique()}")
    print(f"Years: {sorted(long_df['year'].unique().tolist())}")
    print(f"Rows: {len(long_df)}")

    print(f"\nSaved long format Table 3 to: {output_file}")


def create_table_3_top_2025_summary():
    """
    Create a 2025 summary from Table 3 long-format data.

    Input:
    - table_3_long.csv

    Output:
    - table_3_top_2025.csv
    """
    input_file = "data/interim/aggregated/ons_skills/table_3_long.csv"
    output_file = "data/interim/aggregated/ons_skills/table_3_top_2025.csv"

    # Check the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find file: {input_file}")

    # Load the long-format Table 3 data
    df = pd.read_csv(input_file)

    # Keep only 2025 rows
    df_2025 = df[df["year"] == 2025].copy()

    # Sort highest to lowest
    df_2025 = df_2025.sort_values(by="value", ascending=False)

    # Save the 2025 summary
    df_2025.to_csv(output_file, index=False)

    print("\n--- TABLE 3 TOP 2025 SUMMARY ---")
    print(df_2025.head(15))

    print(f"\nSaved top 2025 summary to: {output_file}")


def create_table_3_top_2025_chart():
    """
    Create a bar chart for the top 10 Table 3 groups in 2025.

    Input:
    - table_3_top_2025.csv

    Output:
    - table_3_top_10_2025_chart.png
    """
    input_file = "data/interim/aggregated/ons_skills/table_3_top_2025.csv"
    output_file = "data/processed/ons_skills/table_3_top_10_2025_chart.png"

    # Check the input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Could not find file: {input_file}")

    # Create the output folder if needed
    os.makedirs("data/processed/ons_skills", exist_ok=True)

    # Load the 2025 summary
    df = pd.read_csv(input_file)

    # Take only the top 10 rows
    top_10 = df.head(10).copy()

    # Create the chart
    plt.figure(figsize=(12, 6))
    plt.barh(top_10["least_label"], top_10["value"])
    plt.xlabel("Value")
    plt.ylabel("Least Detailed Skill Group")
    plt.title("Top 10 ONS Table 3 Skill Groups in 2025")

    # Show highest at the top
    plt.gca().invert_yaxis()

    # Improve spacing
    plt.tight_layout()

    # Save the chart
    plt.savefig(output_file)
    plt.close()

    print(f"\nSaved Table 3 top 10 chart to: {output_file}")


def compare_table_2_and_table_3_2025():
    """
    Print a simple comparison between the top 10 Table 2 and Table 3 groups in 2025.
    This helps with writing the report discussion section.
    """
    table_2_file = "data/interim/aggregated/ons_skills/table_2_top_2025.csv"
    table_3_file = "data/interim/aggregated/ons_skills/table_3_top_2025.csv"

    if not os.path.exists(table_2_file):
        raise FileNotFoundError(f"Could not find file: {table_2_file}")

    if not os.path.exists(table_3_file):
        raise FileNotFoundError(f"Could not find file: {table_3_file}")

    table_2_df = pd.read_csv(table_2_file)
    table_3_df = pd.read_csv(table_3_file)

    print("\n--- TABLE 2 TOP 10 GROUPS IN 2025 ---")
    print(table_2_df[["middle_label", "value"]].head(10))

    print("\n--- TABLE 3 TOP 10 GROUPS IN 2025 ---")
    print(table_3_df[["least_label", "value"]].head(10))

def run_ons_analysis():
    """
    Main ONS analysis function.

    Runs:
    - Table 2 conversion
    - Table 2 2025 summary
    - Table 2 chart
    - Table 3 conversion
    - Table 3 2025 summary
    - Table 3 chart
    - Simple comparison printout
    """
    convert_table_2_to_long()
    create_table_2_top_2025_summary()
    create_table_2_top_2025_chart()

    convert_table_3_to_long()
    create_table_3_top_2025_summary()
    create_table_3_top_2025_chart()

    compare_table_2_and_table_3_2025()