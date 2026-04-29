from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


OUTPUT_DIR = Path("data/processed/evaluation/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_chart(filename):
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


def add_bar_labels_horizontal(ax):
    for bar in ax.patches:
        width = bar.get_width()
        ax.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}",
            va="center",
            fontsize=9,
        )


def create_artefact_match_scores_chart():
    data = {
        "Job Advert": ["Customs/Admin Role", "AI/Data Role", "Delivery Role"],
        "Match Score (%)": [10.42, 20.00, 72.73],
    }

    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["Job Advert"], df["Match Score (%)"])

    ax.set_title("Artefact Match Scores: Delivery-Focused CV")
    ax.set_xlabel("Match Score (%)")
    ax.set_ylabel("Job Advert")
    ax.set_xlim(0, 100)
    ax.grid(axis="x", alpha=0.3)

    add_bar_labels_horizontal(ax)

    save_chart("01_artefact_match_scores.png")


def create_top_skills_percentage_chart():
    data = {
        "Skill": [
            "customer service",
            "accounting",
            "health and safety",
            "project commissioning",
            "project management",
            "database management systems",
            "statistics",
            "SQL",
            "SAP Data Services",
            "risk management",
            "financial forecasting",
            "JavaScript",
            "quality standards",
            "business processes",
            "data protection",
        ],
        "Match Count": [
            5350,
            1275,
            1175,
            1030,
            1025,
            610,
            600,
            590,
            500,
            425,
            375,
            330,
            310,
            300,
            275,
        ],
    }

    df = pd.DataFrame(data)
    df["Percentage"] = (df["Match Count"] / df["Match Count"].sum()) * 100
    df = df.sort_values("Percentage", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(df["Skill"], df["Percentage"])

    ax.set_title("Baseline V3: Top Extracted Skills by Relative Frequency")
    ax.set_xlabel("Percentage of Top Skill Mentions (%)")
    ax.set_ylabel("Skill")
    ax.grid(axis="x", alpha=0.3)

    for bar in ax.patches:
        width = bar.get_width()
        ax.text(
            width + 0.2,
            bar.get_y() + bar.get_height() / 2,
            f"{width:.1f}%",
            va="center",
            fontsize=9,
        )

    save_chart("02_baseline_v3_top_skills_percentage.png")


def create_distilbert_metrics_chart():
    metrics = ["Precision", "Recall", "F1"]
    dev_scores = [0.4236, 0.4561, 0.4392]
    test_scores = [0.4150, 0.4253, 0.4201]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(metrics, dev_scores, marker="o", label="Dev")
    ax.plot(metrics, test_scores, marker="o", label="Test")

    ax.set_title("DistilBERT SkillSpan Evaluation Results")
    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend()

    for i, score in enumerate(dev_scores):
        ax.text(i, score + 0.025, f"{score:.2f}", ha="center", fontsize=9)

    for i, score in enumerate(test_scores):
        ax.text(i, score - 0.04, f"{score:.2f}", ha="center", fontsize=9)

    save_chart("03_distilbert_skillspan_metrics.png")


def create_weak_labelling_tradeoff_chart():
    versions = [
        "Original\nSmall",
        "Large\nRaw",
        "V1\nStrict",
        "V2\nBalanced",
        "V3\nOptimised",
    ]

    rows = [16, 2740, 842, 368, 143]
    avg_b_tags = [5.06, 4.56, 1.86, 2.29, 2.64]
    avg_labelled_tokens = [6.69, 6.69, 4.21, 5.15, 5.10]

    x = np.arange(len(versions))

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(x, avg_b_tags, marker="o", label="Avg Skill Starts per Row")
    ax1.plot(x, avg_labelled_tokens, marker="o", label="Avg Labelled Tokens per Row")

    ax1.set_xlabel("Weak-Labelling Version")
    ax1.set_ylabel("Skill Density per Row")
    ax1.set_xticks(x)
    ax1.set_xticklabels(versions)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(x, rows, alpha=0.2, label="Rows Saved")
    ax2.set_ylabel("Rows Saved")

    ax1.set_title("Weak-Labelling Trade-off: Quantity vs Skill Density")

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    ax1.annotate(
        "Cleaner but fewer rows",
        xy=(4, avg_labelled_tokens[4]),
        xytext=(2.7, 6.2),
        arrowprops=dict(arrowstyle="->"),
        fontsize=9,
    )

    save_chart("04_weak_labelling_tradeoff.png")


def main():
    create_artefact_match_scores_chart()
    create_top_skills_percentage_chart()
    create_distilbert_metrics_chart()
    create_weak_labelling_tradeoff_chart()

    print(f"Saved report-ready charts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()