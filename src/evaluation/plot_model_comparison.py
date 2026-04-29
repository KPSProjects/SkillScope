from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    # model comparison metrics collected from saved evaluation result files
    model_results = [
        {
            "method": "Baseline\nKeyword",
            "precision": 0.4617,
            "recall": 0.1334,
            "f1": 0.2070,
        },
        {
            "method": "spaCy\nExact Span",
            "precision": 0.3901,
            "recall": 0.2312,
            "f1": 0.2903,
        },
        {
            "method": "DistilBERT\n3 Epoch",
            "precision": 0.3981,
            "recall": 0.4189,
            "f1": 0.4082,
        },
        {
            "method": "DistilBERT\n5 Epoch",
            "precision": 0.4150,
            "recall": 0.4253,
            "f1": 0.4201,
        },
    ]

    # convert results into a dataframe so it is easier to save and plot
    results_df = pd.DataFrame(model_results)

    # output folder
    output_dir = Path("data/processed/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # save a csv copy of the metrics for report use
    metrics_csv_path = output_dir / "model_comparison_metrics.csv"
    results_df.to_csv(metrics_csv_path, index=False)

    # make grouped bar chart
    x_positions = list(range(len(results_df)))
    bar_width = 0.22

    plt.figure(figsize=(10, 6))

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

    plt.title("Skill Extraction Model Comparison")
    plt.xlabel("Method")
    plt.ylabel("Score")
    plt.xticks(x_positions, results_df["method"])
    plt.ylim(0, 0.8)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    # save graph
    graph_output_path = output_dir / "model_comparison_graph.png"
    plt.savefig(graph_output_path, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved graph to: {graph_output_path}")
    print(f"Saved metrics table to: {metrics_csv_path}")


if __name__ == "__main__":
    main()