from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    # DistilBERT progression through different training setups
    distilbert_results = [
        {
            "setup": "Initial\n3 Epoch",
            "precision": 0.1392,
            "recall": 0.1494,
            "f1": 0.1441,
        },
        {
            "setup": "Row-Level\n3 Epoch",
            "precision": 0.3981,
            "recall": 0.4189,
            "f1": 0.4082,
        },
        {
            "setup": "Row-Level\n5 Epoch",
            "precision": 0.4150,
            "recall": 0.4253,
            "f1": 0.4201,
        },
    ]

    results_df = pd.DataFrame(distilbert_results)

    output_dir = Path("data/processed/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "distilbert_progression_metrics.csv", index=False)

    x_positions = list(range(len(results_df)))
    bar_width = 0.22

    plt.figure(figsize=(9, 6))

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

    plt.title("DistilBERT Improvement Across Training Setups")
    plt.xlabel("Training Setup")
    plt.ylabel("Score")
    plt.xticks(x_positions, results_df["setup"])
    plt.ylim(0, 0.5)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_dir / "distilbert_progression_graph.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("Saved DistilBERT progression graph and csv.")


if __name__ == "__main__":
    main()