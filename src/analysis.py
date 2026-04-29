import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    REED_MATCHED_SKILLS_PATH,
    REED_SKILL_FREQUENCY_PATH,
    REED_SKILL_FREQUENCY_FILTERED_PATH,
    REED_TOP_SKILLS_CHART_PATH,
)


def load_reed_matches() -> pd.DataFrame:
    """
    Loads the matched skills file created by the extraction step.
    """
    return pd.read_csv(REED_MATCHED_SKILLS_PATH)


def create_skill_frequency_summary(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts how often each preferred ESCO skill label appears
    across the matched Reed job adverts.
    """
    if matches_df.empty:
        return pd.DataFrame(columns=["preferred_label", "match_count"])

    summary_df = (
        matches_df["preferred_label"]
        .value_counts()
        .reset_index()
    )

    summary_df.columns = ["preferred_label", "match_count"]

    return summary_df


def filter_noisy_skills(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes obviously weak or unhelpful labels from the first baseline summary.
    This is a simple manual clean-up step for analysis/demo purposes.
    """
    excluded_labels = {
        "plan",
        "dies",
    }

    filtered_df = summary_df[~summary_df["preferred_label"].isin(excluded_labels)].copy()
    filtered_df = filtered_df.reset_index(drop=True)

    return filtered_df


def save_summary(df: pd.DataFrame, path) -> None:
    """
    Saves the aggregated summary to CSV.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def create_top_skills_chart(filtered_df: pd.DataFrame) -> None:
    """
    Makes a chart of the top filtered Reed skills.
    """
    top_skills = filtered_df.head(10)

    print("\n--- TOP FILTERED REED SKILLS ---")
    print(top_skills)

    plt.figure(figsize=(12, 6))
    plt.bar(top_skills["preferred_label"], top_skills["match_count"])
    plt.title("Top Extracted Skills from Reed Dataset")
    plt.xlabel("Skill")
    plt.ylabel("Match Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    REED_TOP_SKILLS_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(REED_TOP_SKILLS_CHART_PATH)
    plt.close()

    print(f"\nSaved Reed top skills chart to: {REED_TOP_SKILLS_CHART_PATH}")


def run_analysis():
    """
    Main function for the first aggregation step.
    """
    print("Starting matched skill analysis...")

    matches_df = load_reed_matches()
    print(f"Loaded matched rows: {len(matches_df)}")

    summary_df = create_skill_frequency_summary(matches_df)

    print("\n--- RAW SKILL FREQUENCY SUMMARY ---")
    print(summary_df.head(10))

    filtered_summary_df = filter_noisy_skills(summary_df)

    print("\n--- FILTERED SKILL FREQUENCY SUMMARY ---")
    print(filtered_summary_df.head(10))

    save_summary(summary_df, REED_SKILL_FREQUENCY_PATH)
    save_summary(filtered_summary_df, REED_SKILL_FREQUENCY_FILTERED_PATH)

    print(f"\nSaved raw skill frequency summary to: {REED_SKILL_FREQUENCY_PATH}")
    print(f"Saved filtered skill frequency summary to: {REED_SKILL_FREQUENCY_FILTERED_PATH}")

    create_top_skills_chart(filtered_summary_df)