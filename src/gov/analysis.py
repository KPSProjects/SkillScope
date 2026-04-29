import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    GOV_CLEANED_STANDARDISED_PATH,
    GOV_MONTHLY_COUNTS_PATH,
    GOV_MONTHLY_COUNTS_CHART_PATH,
    GOV_CATEGORY_COUNTS_PATH,
    GOV_TOP_CATEGORIES_CHART_PATH,
    GOV_TOP_CATEGORY_MONTHLY_PATH,
    GOV_TOP_CATEGORY_MONTHLY_CHART_PATH,
)


def load_gov_data() -> pd.DataFrame:
    """
    Loads the cleaned GOV dataset.
    """
    return pd.read_csv(GOV_CLEANED_STANDARDISED_PATH)


def clean_gov_basic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies basic cleaning needed for GOV analysis.
    - converts posting_date to datetime
    - cleans category text
    - removes empty / invalid categories
    """
    df = df.copy()

    # convert posting date into real datetime
    df["posting_date"] = pd.to_datetime(df["posting_date"], errors="coerce")

    # clean category text
    df["category"] = df["category"].astype(str).str.strip()

    # remove missing / blank / invalid category values
    df = df[df["category"] != ""].copy()
    df = df[df["category"].str.lower() != "nan"].copy()

    return df


def create_monthly_job_counts():
    """
    Creates a monthly count of GOV job postings.
    """
    df = load_gov_data()
    df = clean_gov_basic(df)

    # remove rows where posting_date could not be read
    df = df.dropna(subset=["posting_date"]).copy()

    # create month period
    df["month"] = df["posting_date"].dt.to_period("M")

    # count jobs in each month
    monthly_counts = (
        df.groupby("month")
        .size()
        .reset_index(name="job_count")
        .sort_values("month")
    )

    # convert month to string for saving / plotting
    monthly_counts["month"] = monthly_counts["month"].astype(str)

    # save CSV
    GOV_MONTHLY_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    monthly_counts.to_csv(GOV_MONTHLY_COUNTS_PATH, index=False)

    # terminal output
    print("\n--- MONTHLY JOB COUNTS ---")
    print(monthly_counts.head(12))

    print("\n--- MONTHLY SUMMARY ---")
    print(f"Rows with unreadable posting_date removed: {load_gov_data()['posting_date'].shape[0] - df['posting_date'].shape[0]}")
    print(f"First month: {monthly_counts['month'].min()}")
    print(f"Last month: {monthly_counts['month'].max()}")
    print(f"Total months: {monthly_counts.shape[0]}")
    print(f"Average monthly jobs: {monthly_counts['job_count'].mean():.2f}")

    highest_month = monthly_counts.loc[monthly_counts["job_count"].idxmax()]
    lowest_month = monthly_counts.loc[monthly_counts["job_count"].idxmin()]

    print(f"Highest month: {highest_month['month']} ({highest_month['job_count']} jobs)")
    print(f"Lowest month: {lowest_month['month']} ({lowest_month['job_count']} jobs)")

    # chart
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_counts["month"], monthly_counts["job_count"], marker="o")
    plt.title("Monthly Job Counts from GOV Dataset")
    plt.xlabel("Month")
    plt.ylabel("Number of Jobs")
    plt.xticks(rotation=45)
    plt.tight_layout()

    GOV_MONTHLY_COUNTS_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(GOV_MONTHLY_COUNTS_CHART_PATH)
    plt.close()

    print(f"\nSaved CSV: {GOV_MONTHLY_COUNTS_PATH}")
    print(f"Saved chart: {GOV_MONTHLY_COUNTS_CHART_PATH}")

    return monthly_counts


def create_category_counts():
    """
    Creates overall counts for GOV job categories.
    """
    df = load_gov_data()
    df = clean_gov_basic(df)

    category_counts = (
        df.groupby("category")
        .size()
        .reset_index(name="job_count")
        .sort_values("job_count", ascending=False)
        .reset_index(drop=True)
    )

    GOV_CATEGORY_COUNTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    category_counts.to_csv(GOV_CATEGORY_COUNTS_PATH, index=False)

    top_categories = category_counts.head(10)

    print("\n--- CATEGORY COUNTS ---")
    print(top_categories)

    plt.figure(figsize=(12, 6))
    plt.bar(top_categories["category"], top_categories["job_count"])
    plt.title("Top 10 Job Categories from GOV Dataset")
    plt.xlabel("Category")
    plt.ylabel("Number of Jobs")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    GOV_TOP_CATEGORIES_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(GOV_TOP_CATEGORIES_CHART_PATH)
    plt.close()

    print(f"\nSaved CSV: {GOV_CATEGORY_COUNTS_PATH}")
    print(f"Saved chart: {GOV_TOP_CATEGORIES_CHART_PATH}")

    return category_counts


def create_top_category_monthly_counts():
    """
    Creates monthly counts for the top 5 GOV categories.
    """
    df = load_gov_data()
    df = clean_gov_basic(df)

    # remove rows where posting_date could not be read
    df = df.dropna(subset=["posting_date"]).copy()

    # keep period first so sorting stays correct
    df["month_period"] = df["posting_date"].dt.to_period("M")

    # find top 5 categories overall
    top_categories = (
        df["category"]
        .value_counts()
        .head(5)
        .index
        .tolist()
    )

    filtered_df = df[df["category"].isin(top_categories)].copy()

    category_monthly_counts = (
        filtered_df.groupby(["month_period", "category"])
        .size()
        .reset_index(name="job_count")
        .sort_values(["month_period", "job_count"], ascending=[True, False])
    )

    # make month string for saving / plotting
    category_monthly_counts["month"] = category_monthly_counts["month_period"].astype(str)

    # final column order
    category_monthly_counts = category_monthly_counts[
        ["month", "category", "job_count"]
    ].copy()

    GOV_TOP_CATEGORY_MONTHLY_PATH.parent.mkdir(parents=True, exist_ok=True)
    category_monthly_counts.to_csv(GOV_TOP_CATEGORY_MONTHLY_PATH, index=False)

    print("\n--- TOP CATEGORY MONTHLY COUNTS ---")
    print(category_monthly_counts.head(20))

    pivot_df = category_monthly_counts.pivot(
        index="month",
        columns="category",
        values="job_count"
    ).fillna(0)

    plt.figure(figsize=(12, 6))
    for category in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[category], marker="o", label=category)

    plt.title("Monthly Job Counts for Top 5 GOV Categories")
    plt.xlabel("Month")
    plt.ylabel("Number of Jobs")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    GOV_TOP_CATEGORY_MONTHLY_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(GOV_TOP_CATEGORY_MONTHLY_CHART_PATH)
    plt.close()

    print(f"\nSaved CSV: {GOV_TOP_CATEGORY_MONTHLY_PATH}")
    print(f"Saved chart: {GOV_TOP_CATEGORY_MONTHLY_CHART_PATH}")

    return category_monthly_counts


def run_gov_analysis():
    """
    Runs all GOV analysis steps.
    """
    print("Starting GOV analysis...")

    monthly_counts = create_monthly_job_counts()
    category_counts = create_category_counts()
    top_category_monthly_counts = create_top_category_monthly_counts()

    print("\nGOV analysis complete.")
    print(f"Monthly rows: {len(monthly_counts)}")
    print(f"Category rows: {len(category_counts)}")
    print(f"Top category monthly rows: {len(top_category_monthly_counts)}")