from pathlib import Path

import pandas as pd


def main():
    summary_rows = [
        {
            "method": "Baseline keyword matcher",
            "evaluation_setup": "SkillSpan token-level",
            "precision": 0.4617,
            "recall": 0.1334,
            "f1": 0.2070,
            "notes": "Transparent baseline, very low recall",
        },
        {
            "method": "spaCy NER",
            "evaluation_setup": "Row-level binary",
            "precision": 0.7043,
            "recall": 0.5883,
            "f1": 0.6411,
            "notes": "Strong row-level detection",
        },
        {
            "method": "spaCy NER",
            "evaluation_setup": "Exact span-level",
            "precision": 0.3901,
            "recall": 0.2312,
            "f1": 0.2903,
            "notes": "Stricter span matching",
        },
        {
            "method": "DistilBERT",
            "evaluation_setup": "Initial token classification, 3 epochs",
            "precision": 0.1392,
            "recall": 0.1494,
            "f1": 0.1441,
            "notes": "Weak early setup",
        },
        {
            "method": "DistilBERT",
            "evaluation_setup": "Row-level token classification, 3 epochs",
            "precision": 0.3981,
            "recall": 0.4189,
            "f1": 0.4082,
            "notes": "Large improvement after row-level redesign",
        },
        {
            "method": "DistilBERT",
            "evaluation_setup": "Row-level token classification, 5 epochs",
            "precision": 0.4150,
            "recall": 0.4253,
            "f1": 0.4201,
            "notes": "Best DistilBERT model",
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    output_dir = Path("data/processed/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "evaluation_summary_table.csv", index=False)

    print("Saved evaluation summary table.")


if __name__ == "__main__":
    main()