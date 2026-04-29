from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_DIR = Path("data/processed/evaluation/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_chart(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def create_weak_labelling_quality_chart():
    data = {
        "Version": ["Original", "V1", "V2", "V3"],
        "Rows Saved": [2750, 115, 43, 15],
        "Avg B-Tags per Row": [4.55, 1.87, 2.28, 2.46],
        "Avg Labelled Tokens per Row": [6.68, 4.13, 5.07, 5.23],
    }

    df = pd.DataFrame(data)

    ax = df.plot(
        x="Version",
        y=["Avg B-Tags per Row", "Avg Labelled Tokens per Row"],
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title("Weak-Labelling Quality by Version")
    ax.set_ylabel("Average per Saved Row")
    ax.set_xlabel("Weak-Labelling Version")
    ax.grid(axis="y", alpha=0.3)

    save_chart(OUTPUT_DIR / "weak_labelling_quality_by_version.png")


def create_weak_labelling_quantity_chart():
    data = {
        "Version": ["Original", "V1", "V2", "V3"],
        "Rows Saved": [2750, 115, 43, 15],
        "Save Rate (%)": [91.5, 22.8, 8.6, 2.8],
    }

    df = pd.DataFrame(data)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(df["Version"], df["Rows Saved"])
    ax1.set_ylabel("Rows Saved")
    ax1.set_xlabel("Weak-Labelling Version")
    ax1.set_title("Weak-Labelling Quantity by Version")
    ax1.grid(axis="y", alpha=0.3)

    save_chart(OUTPUT_DIR / "weak_labelling_quantity_by_version.png")


def create_v3_top_skills_chart():
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

    df = pd.DataFrame(data).sort_values("Match Count", ascending=True)

    plt.figure(figsize=(10, 7))
    plt.barh(df["Skill"], df["Match Count"])
    plt.title("Baseline V3: Top Extracted Skills from Reed Dataset")
    plt.xlabel("Match Count")
    plt.ylabel("Skill")
    plt.grid(axis="x", alpha=0.3)

    save_chart(OUTPUT_DIR / "baseline_v3_top_skills_clean.png")


def create_distilbert_results_chart():
    data = {
        "Metric": ["Precision", "Recall", "F1"],
        "Dev": [0.4236, 0.4561, 0.4392],
        "Test": [0.4150, 0.4253, 0.4201],
    }

    df = pd.DataFrame(data)

    ax = df.plot(
        x="Metric",
        y=["Dev", "Test"],
        kind="bar",
        figsize=(9, 6),
    )

    ax.set_title("DistilBERT SkillSpan Evaluation Results")
    ax.set_ylabel("Score")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    save_chart(OUTPUT_DIR / "distilbert_skillspan_metrics.png")


def create_artefact_match_results_chart():
    data = {
        "Job": ["Delivery Role", "AI/Data Role", "Customs/Admin Role"],
        "Delivery CV Score": [72.73, 20.00, 10.42],
    }

    df = pd.DataFrame(data).sort_values("Delivery CV Score", ascending=True)

    plt.figure(figsize=(9, 6))
    plt.barh(df["Job"], df["Delivery CV Score"])
    plt.title("Artefact Match Scores Using Delivery-Focused CV")
    plt.xlabel("Match Score (%)")
    plt.ylabel("Job Advert")
    plt.xlim(0, 100)
    plt.grid(axis="x", alpha=0.3)

    save_chart(OUTPUT_DIR / "artefact_delivery_cv_match_scores.png")


def main():
    create_weak_labelling_quality_chart()
    create_weak_labelling_quantity_chart()
    create_v3_top_skills_chart()
    create_distilbert_results_chart()
    create_artefact_match_results_chart()

    print(f"Saved evaluation charts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()