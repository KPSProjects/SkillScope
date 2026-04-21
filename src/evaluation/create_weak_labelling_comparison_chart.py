import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl_rows(file_path: Path):
    rows = []

    if not file_path.exists():
        print(f"Missing file: {file_path}")
        return rows

    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    return rows


def calculate_average_b_tags(rows):
    if not rows:
        return 0

    total_b_tags = 0
    for row in rows:
        tags = row.get("tags_skill", [])
        total_b_tags += sum(1 for tag in tags if tag == "B")

    return total_b_tags / len(rows)


def calculate_average_labelled_tokens(rows):
    if not rows:
        return 0

    total_labelled_tokens = 0
    for row in rows:
        tags = row.get("tags_skill", [])
        total_labelled_tokens += sum(1 for tag in tags if tag in {"B", "I"})

    return total_labelled_tokens / len(rows)


def calculate_average_row_length(rows):
    if not rows:
        return 0

    total_tokens = 0
    for row in rows:
        tokens = row.get("tokens", [])
        total_tokens += len(tokens)

    return total_tokens / len(rows)


def save_summary_csv(output_path: Path, summary_rows):
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "version",
            "rows_attempted",
            "rows_saved",
            "save_rate_percent",
            "avg_b_tags_per_row",
            "avg_labelled_tokens_per_row",
            "avg_tokens_per_row",
        ])

        for row in summary_rows:
            writer.writerow(row)


def save_summary_txt(output_path: Path, summary_rows):
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("WEAK LABELLING COMPARISON SUMMARY\n")
        file.write("=================================\n\n")

        for row in summary_rows:
            version, attempted, saved, save_rate, avg_b, avg_labelled, avg_tokens = row

            file.write(f"Version: {version}\n")
            file.write(f"Rows attempted: {attempted}\n")
            file.write(f"Rows saved: {saved}\n")
            file.write(f"Save rate (%): {save_rate:.2f}\n")
            file.write(f"Average B-tags per row: {avg_b:.2f}\n")
            file.write(f"Average labelled tokens per row: {avg_labelled:.2f}\n")
            file.write(f"Average tokens per row: {avg_tokens:.2f}\n")
            file.write("\n")


def run_weak_labelling_comparison_chart():
    base_dir = Path(__file__).resolve().parent.parent.parent
    processed_dir = base_dir / "data" / "processed" / "distilbert"
    output_dir = base_dir / "data" / "processed" / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "Original": processed_dir / "weak_labelled_reed_large.jsonl",
        "V1": processed_dir / "weak_labelled_reed_large_v1.jsonl",
        "V2": processed_dir / "weak_labelled_reed_large_v2.jsonl",
        "V3": processed_dir / "weak_labelled_reed_large_v3.jsonl",
    }

    rows_attempted = {
        "Original": 3000,
        "V1": 500,
        "V2": 500,
        "V3": 500,
    }

    versions = []
    rows_saved = []
    save_rates = []
    avg_b_tags = []
    avg_labelled_tokens = []
    avg_tokens_per_row = []
    summary_rows = []

    for version, file_path in files.items():
        rows = load_jsonl_rows(file_path)

        saved = len(rows)
        save_rate = (saved / rows_attempted[version]) * 100
        avg_b = calculate_average_b_tags(rows)
        avg_labelled = calculate_average_labelled_tokens(rows)
        avg_tokens = calculate_average_row_length(rows)

        versions.append(version)
        rows_saved.append(saved)
        save_rates.append(save_rate)
        avg_b_tags.append(avg_b)
        avg_labelled_tokens.append(avg_labelled)
        avg_tokens_per_row.append(avg_tokens)

        summary_rows.append([
            version,
            rows_attempted[version],
            saved,
            save_rate,
            avg_b,
            avg_labelled,
            avg_tokens,
        ])

    # save summary files
    save_summary_csv(output_dir / "weak_labelling_comparison_summary.csv", summary_rows)
    save_summary_txt(output_dir / "weak_labelling_comparison_summary.txt", summary_rows)

    # chart 1: grouped comparison chart
    x = np.arange(len(versions))
    width = 0.25

    plt.figure(figsize=(12, 7))
    plt.bar(x - width, save_rates, width, label="Save Rate (%)")
    plt.bar(x, avg_b_tags, width, label="Avg B-Tags")
    plt.bar(x + width, avg_labelled_tokens, width, label="Avg Labelled Tokens")

    plt.xticks(x, versions)
    plt.title("Weak-Labelling Experiment: Quantity vs Quality by Version")
    plt.xlabel("Version")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "weak_labelling_grouped_comparison_chart.png", dpi=300)
    plt.close()

    # chart 2: scatter plot quantity vs quality
    plt.figure(figsize=(10, 6))
    plt.scatter(rows_saved, avg_labelled_tokens, s=140)

    for i, version in enumerate(versions):
        plt.annotate(
            version,
            (rows_saved[i], avg_labelled_tokens[i]),
            textcoords="offset points",
            xytext=(8, 6),
        )

    plt.title("Weak-Labelling Experiment: Rows Saved vs Label Quality")
    plt.xlabel("Rows Saved")
    plt.ylabel("Average Labelled Tokens per Saved Row")
    plt.tight_layout()
    plt.savefig(output_dir / "weak_labelling_quantity_vs_quality_scatter.png", dpi=300)
    plt.close()

    # chart 3: optional average token length
    plt.figure(figsize=(10, 6))
    plt.bar(versions, avg_tokens_per_row)
    plt.title("Weak-Labelling Experiment: Average Tokens per Saved Row")
    plt.xlabel("Version")
    plt.ylabel("Average Tokens per Row")
    plt.tight_layout()
    plt.savefig(output_dir / "weak_labelling_avg_tokens_per_row_chart.png", dpi=300)
    plt.close()

    print("Saved outputs to:")
    print(output_dir / "weak_labelling_comparison_summary.csv")
    print(output_dir / "weak_labelling_comparison_summary.txt")
    print(output_dir / "weak_labelling_grouped_comparison_chart.png")
    print(output_dir / "weak_labelling_quantity_vs_quality_scatter.png")
    print(output_dir / "weak_labelling_avg_tokens_per_row_chart.png")


if __name__ == "__main__":
    run_weak_labelling_comparison_chart()