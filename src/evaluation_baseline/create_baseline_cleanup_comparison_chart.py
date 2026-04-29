import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from src.config import (
    REED_LARGE_SKILL_FREQUENCY_FILTERED_PATH,
    REED_LARGE_SKILL_FREQUENCY_FILTERED_V2_PATH,
)


def get_count_for_label(df, label):
    match = df[df["preferred_label"].str.lower() == label.lower()]
    if match.empty:
        return 0
    return int(match["match_count"].iloc[0])


def run_baseline_cleanup_comparison_chart():
    base_dir = Path(__file__).resolve().parent.parent.parent
    output_dir = base_dir / "data" / "processed" / "evaluation_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    v1_df = pd.read_csv(REED_LARGE_SKILL_FREQUENCY_FILTERED_PATH)
    v2_df = pd.read_csv(REED_LARGE_SKILL_FREQUENCY_FILTERED_V2_PATH)

    noisy_labels = [
        "assume responsibility",
        "cope with stress",
        "show commitment",
        "work independently",
        "maintenance operations",
        "product comprehension",
        "company policies",
        "meet commitments",
        "build trust",
        "demonstrate trustworthiness",
        "demonstrate willingness to learn",
    ]

    v1_counts = [get_count_for_label(v1_df, label) for label in noisy_labels]
    v2_counts = [get_count_for_label(v2_df, label) for label in noisy_labels]

    # chart 1: noisy label cleanup
    x = range(len(noisy_labels))
    width = 0.4

    plt.figure(figsize=(14, 8))
    plt.bar([i - width / 2 for i in x], v1_counts, width=width, label="Baseline V1")
    plt.bar([i + width / 2 for i in x], v2_counts, width=width, label="Baseline V2")
    plt.xticks(x, noisy_labels, rotation=45, ha="right")
    plt.title("Baseline Cleanup Comparison: Noisy Labels in V1 vs V2")
    plt.xlabel("Noisy Label")
    plt.ylabel("Match Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_cleanup_noisy_labels_chart.png", dpi=300)
    plt.close()

    # chart 2: label overlap summary
    v1_labels = set(v1_df["preferred_label"].str.lower())
    v2_labels = set(v2_df["preferred_label"].str.lower())

    v1_only = len(v1_labels - v2_labels)
    shared = len(v1_labels & v2_labels)
    v2_only = len(v2_labels - v1_labels)

    categories = ["V1 only", "Shared", "V2 only"]
    counts = [v1_only, shared, v2_only]

    plt.figure(figsize=(9, 6))
    plt.bar(categories, counts)
    plt.title("Baseline V1 vs V2: Label Overlap Summary")
    plt.xlabel("Category")
    plt.ylabel("Number of Labels")
    plt.tight_layout()
    plt.savefig(output_dir / "baseline_label_overlap_chart.png", dpi=300)
    plt.close()

    # text summary
    summary_path = output_dir / "baseline_cleanup_comparison_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as file:
        file.write("BASELINE CLEANUP COMPARISON SUMMARY\n")
        file.write("===================================\n\n")

        file.write("Noisy label comparison\n")
        file.write("----------------------\n")
        for label, v1_count, v2_count in zip(noisy_labels, v1_counts, v2_counts):
            file.write(f"{label}: V1={v1_count}, V2={v2_count}\n")

        file.write("\nLabel overlap summary\n")
        file.write("---------------------\n")
        file.write(f"V1 only labels: {v1_only}\n")
        file.write(f"Shared labels: {shared}\n")
        file.write(f"V2 only labels: {v2_only}\n")

    print("Saved outputs to:")
    print(output_dir / "baseline_cleanup_noisy_labels_chart.png")
    print(output_dir / "baseline_label_overlap_chart.png")
    print(summary_path)


if __name__ == "__main__":
    run_baseline_cleanup_comparison_chart()