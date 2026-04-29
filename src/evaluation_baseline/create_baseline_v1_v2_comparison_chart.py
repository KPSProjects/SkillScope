import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.config import (
    REED_LARGE_SKILL_FREQUENCY_FILTERED_PATH,
    REED_LARGE_SKILL_FREQUENCY_FILTERED_V2_PATH,
)


def run_baseline_v1_v2_comparison_chart():
    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "data" / "processed" / "evaluation_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    v1_df = pd.read_csv(REED_LARGE_SKILL_FREQUENCY_FILTERED_PATH)
    v2_df = pd.read_csv(REED_LARGE_SKILL_FREQUENCY_FILTERED_V2_PATH)

    top_v1 = v1_df.head(10).copy()
    top_v2 = v2_df.head(10).copy()

    # chart 1: baseline v1
    plt.figure(figsize=(12, 7))
    plt.barh(top_v1["preferred_label"][::-1], top_v1["match_count"][::-1])
    plt.title("Baseline V1: Top 10 Extracted Skills")
    plt.xlabel("Match Count")
    plt.ylabel("Skill")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_v1_top_skills_chart.png", dpi=300)
    plt.close()

    # chart 2: baseline v2
    plt.figure(figsize=(12, 7))
    plt.barh(top_v2["preferred_label"][::-1], top_v2["match_count"][::-1])
    plt.title("Baseline V2: Top 10 Extracted Skills")
    plt.xlabel("Match Count")
    plt.ylabel("Skill")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_v2_top_skills_chart.png", dpi=300)
    plt.close()

    # save quick text summary too
    summary_path = output_dir / "baseline_v1_v2_comparison_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("BASELINE V1 VS V2 COMPARISON SUMMARY\n")
        file.write("====================================\n\n")

        file.write("V1 Top 10 Skills\n")
        file.write("----------------\n")
        for _, row in top_v1.iterrows():
            file.write(f"{row['preferred_label']}: {row['match_count']}\n")

        file.write("\nV2 Top 10 Skills\n")
        file.write("----------------\n")
        for _, row in top_v2.iterrows():
            file.write(f"{row['preferred_label']}: {row['match_count']}\n")

    print("Saved outputs to:")
    print(output_dir / "baseline_v1_top_skills_chart.png")
    print(output_dir / "baseline_v2_top_skills_chart.png")
    print(summary_path)


if __name__ == "__main__":
    run_baseline_v1_v2_comparison_chart()