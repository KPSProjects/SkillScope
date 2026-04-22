import pandas as pd

from src.config import (
    REED_PATH,
    GOV_RAW_PATH,
    GOV_SOURCE_CLEANED_PATH,
    ESCO_SKILLS_PATH,
    SKILLSPAN_TRAIN_PATH,
    SKILLSPAN_DEV_PATH,
    SKILLSPAN_TEST_PATH,
    ONS_PATH,
)


def load_reed() -> pd.DataFrame:
    """Load the Reed UK sample dataset stored as line-delimited JSON."""
    return pd.read_json(REED_PATH, lines=True)


def load_gov_raw() -> pd.DataFrame:
    """Load the raw GOV dataset."""
    return pd.read_csv(GOV_RAW_PATH)


def load_gov_cleaned() -> pd.DataFrame:
    """Load the source cleaned GOV dataset."""
    return pd.read_csv(GOV_SOURCE_CLEANED_PATH)


def load_esco_skills() -> pd.DataFrame:
    """Load the main ESCO skills taxonomy file."""
    return pd.read_csv(ESCO_SKILLS_PATH)


def load_skillspan(split: str) -> pd.DataFrame:
    """Load one of the SkillSpan splits: train, dev, or test."""
    split_map = {
        "train": SKILLSPAN_TRAIN_PATH,
        "dev": SKILLSPAN_DEV_PATH,
        "test": SKILLSPAN_TEST_PATH,
    }

    if split not in split_map:
        raise ValueError("split must be one of: train, dev, test")

    return pd.read_json(split_map[split], lines=True)


def load_ons_sheet_names() -> list[str]:
    """Return all sheet names from the ONS workbook."""
    xls = pd.ExcelFile(ONS_PATH)
    return xls.sheet_names