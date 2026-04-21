import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.config import (
    REED_LARGE_SKILL_FREQUENCY_FILTERED_PATH,
    REED_LARGE_SKILL_FREQUENCY_FILTERED_V2_PATH,
)


def build_count_map(df):
    return {
        str(row["preferred_label"]).strip(): int(row["match_count"])
        for _, row in df.iterrows()
    }


def run_baseline_cleanup_changed_only_chart():
    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "data" / "processed" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    v1_df = pd.read_csv(REED_LARGE_SKILL_FREQUENCY_FILTERED_PATH)
    v2_df = pd.read_csv(REED_LARGE_SKILL_FREQUENCY_FILTERED_V2_PATH)

    v1_map = build_count_map(v1_df)
    v2_map = build_count_map(v2_df)

    all_labels = sorted(set(v1_map.keys()) | set(v2_map.keys()))

    changed_rows = []
    for label in all_labels:
        v1_count = v1_map.get(label, 0)
        v2_count = v2_map.get(label, 0)

        if v1_count != v2_count:
            changed_rows.append(
                {
                    "preferred_label": label,
                    "v1_count": v1_count,
                    "v2_count": v2_count,
                    "count_change": v2_count - v1_count,
                    "abs_change": abs(v2_count - v1_count),
                }
            )

    changed_df = pd.DataFrame(changed_rows)

    if changed_df.empty:
        print("No changed labels found between baseline v1 and v2.")
        return

    changed_df = changed_df.sort_values(by="abs_change", ascending=False).reset_index(drop=True)
    top_changed_df = changed_df.head(15).copy()

    labels = top_changed_df["preferred_label"].tolist()
    v1_counts = top_changed_df["v1_count"].tolist()
    v2_counts = top_changed_df["v2_count"].tolist()

    x = range(len(labels))
    width = 0.4

    plt.figure(figsize=(14, 8))
    plt.bar([i - width / 2 for i in x], v1_counts, width=width, label="Baseline V1")
    plt.bar([i + width / 2 for i in x], v2_counts, width=width, label="Baseline V2")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title("Baseline Cleanup Comparison: Labels with Changed Counts")
    plt.xlabel("Label")
    plt.ylabel("Match Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_cleanup_changed_only_chart.png", dpi=300)
    plt.close()

    summary_path = output_dir / "baseline_cleanup_changed_only_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("BASELINE CLEANUP CHANGED-ONLY SUMMARY\n")
        file.write("====================================\n\n")
        file.write(f"Total changed labels: {len(changed_df)}\n\n")
        file.write("Top changed labels\n")
        file.write("------------------\n")

        for _, row in top_changed_df.iterrows():
            file.write(
                f"{row['preferred_label']}: "
                f"V1={row['v1_count']}, "
                f"V2={row['v2_count']}, "
                f"change={row['count_change']}\n"
            )

    print("Saved outputs to:")
    print(output_dir / "baseline_cleanup_changed_only_chart.png")
    print(summary_path)


if __name__ == "__main__":
    run_baseline_cleanup_changed_only_chart()