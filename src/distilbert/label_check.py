import ast
import pandas as pd

from src.config import (
    SKILLSPAN_TRAIN_PATH,
    SKILLSPAN_DEV_PATH,
    SKILLSPAN_TEST_PATH,
)


def load_skillspan_file(path) -> pd.DataFrame:
    """
    Loads one SkillSpan JSON Lines file into a dataframe.
    """
    return pd.read_json(path, lines=True)


def safe_parse_list(value):
    """
    Ensures token/tag fields are treated as Python lists.
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    if isinstance(value, str):
        value = value.strip()

        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            pass

        return [value]

    return [value]


def collect_unique_skill_labels(df: pd.DataFrame) -> set:
    """
    Collects all unique tags from the tags_skill column.
    """
    unique_labels = set()

    for _, row in df.iterrows():
        tags = safe_parse_list(row["tags_skill"])
        unique_labels.update(tags)

    return unique_labels


def run_label_check():
    print("Checking SkillSpan skill labels...")

    train_df = load_skillspan_file(SKILLSPAN_TRAIN_PATH)
    dev_df = load_skillspan_file(SKILLSPAN_DEV_PATH)
    test_df = load_skillspan_file(SKILLSPAN_TEST_PATH)

    all_labels = set()
    all_labels.update(collect_unique_skill_labels(train_df))
    all_labels.update(collect_unique_skill_labels(dev_df))
    all_labels.update(collect_unique_skill_labels(test_df))

    sorted_labels = sorted(all_labels)

    print("\n--- UNIQUE tags_skill LABELS ---")
    print(sorted_labels)
    print(f"\nTotal unique skill labels: {len(sorted_labels)}")