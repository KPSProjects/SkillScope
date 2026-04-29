from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    # spaCy results from two evaluation views
    spacy_results = [
        {
            "evaluation_type": "Row-Level\nBinary",
            "precision": 0.7043,
            "recall": 0.5883,
            "f1": 0.6411,
        },
        {
            "evaluation_type": "Exact Span\nMatch",
            "precision": 0.3901,
            "recall": 0.2312,
            "f1": 0.2903,
        },
    ]

    results_df = pd.DataFrame(spacy_results)

    output_dir = Path("data/processed/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "spacy_metrics.csv", index=False)

    x_positions = list(range(len(results_df)))
    bar_width = 0.22

    plt.figure(figsize=(8, 6))

    plt.bar(
        [x - bar_width for x in x_positions],
        results_df["precision"],
        width=bar_width,
        label="Precision",
    )
    plt.bar(
        x_positions,
        results_df["recall"],
        width=bar_width,
        label="Recall",
    )
    plt.bar(
        [x + bar_width for x in x_positions],
        results_df["f1"],
        width=bar_width,
        label="F1",
    )

    plt.title("spaCy Evaluation Results")
    plt.xlabel("Evaluation Type")
    plt.ylabel("Score")
    plt.xticks(x_positions, results_df["evaluation_type"])
    plt.ylim(0, 0.8)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / "spacy_results_graph.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved spaCy graph and csv.")


if __name__ == "__main__":
    main()