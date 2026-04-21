import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.config import REED_LARGE_SKILL_FREQUENCY_FILTERED_V3_PATH


def run_baseline_v3_top_skills_chart():
    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "data" / "processed" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REED_LARGE_SKILL_FREQUENCY_FILTERED_V3_PATH)

    if df.empty:
        print("No baseline v3 filtered skill frequency data found.")
        return

    top_df = df.head(20).copy()

    plt.figure(figsize=(13, 8))
    plt.barh(top_df["preferred_label"][::-1], top_df["match_count"][::-1])
    plt.title("Baseline V3: Top 20 Extracted Skills from Reed Large")
    plt.xlabel("Match Count")
    plt.ylabel("Skill")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_v3_top_skills_chart.png", dpi=300)
    plt.close()

    summary_path = output_dir / "baseline_v3_top_skills_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("BASELINE V3 TOP SKILLS SUMMARY\n")
        file.write("==============================\n\n")

        for _, row in top_df.iterrows():
            file.write(f"{row['preferred_label']}: {row['match_count']}\n")

    print("Saved outputs to:")
    print(output_dir / "baseline_v3_top_skills_chart.png")
    print(summary_path)


if __name__ == "__main__":
    run_baseline_v3_top_skills_chart()