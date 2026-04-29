import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


OUTPUT_DIR = Path("data/processed/evaluation/charts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_chart(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def create_weak_labelling_chart():
    # REAL VALUES from your checker (DO NOT CHANGE)
    data = {
        "Version": ["Original", "Large", "V1", "V2", "V3"],
        "Rows": [16, 2740, 842, 368, 143],
        "Avg B-Tags": [5.06, 4.56, 1.86, 2.29, 2.64],
        "Avg Labelled Tokens": [6.69, 6.69, 4.21, 5.15, 5.10],
    }

    df = pd.DataFrame(data)

    # ---------- CHART 1: Quantity ----------
    plt.figure(figsize=(10, 6))
    plt.bar(df["Version"], df["Rows"])
    plt.title("Weak-Labelling: Rows Saved by Version")
    plt.xlabel("Version")
    plt.ylabel("Number of Rows")
    plt.grid(axis="y", alpha=0.3)

    save_chart(OUTPUT_DIR / "weak_labelling_rows.png")

    # ---------- CHART 2: Quality ----------
    plt.figure(figsize=(10, 6))
    plt.plot(df["Version"], df["Avg B-Tags"], marker="o", label="Avg B-Tags")
    plt.plot(df["Version"], df["Avg Labelled Tokens"], marker="o", label="Avg Labelled Tokens")

    plt.title("Weak-Labelling: Skill Density by Version")
    plt.xlabel("Version")
    plt.ylabel("Average per Row")
    plt.legend()
    plt.grid(alpha=0.3)

    save_chart(OUTPUT_DIR / "weak_labelling_quality.png")


def main():
    create_weak_labelling_chart()
    print(f"Saved charts to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()