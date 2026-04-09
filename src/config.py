from pathlib import Path

# Project root
BASE_DIR = Path(__file__).resolve().parent.parent

# Data folders
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw dataset paths
REED_PATH = RAW_DIR / "reed_uk" / "reed_uk-jobs__20190101_20191231_sample.ldjson"
GOV_RAW_PATH = RAW_DIR / "gov" / "rawData.csv"

ESCO_SKILLS_PATH = RAW_DIR / "esco" / "skills_en.csv"
ESCO_HIERARCHY_PATH = RAW_DIR / "esco" / "skillsHierarchy_en.csv"
ESCO_GROUPS_PATH = RAW_DIR / "esco" / "skillGroups_en.csv"
ESCO_OCCUPATION_SKILL_REL_PATH = RAW_DIR / "esco" / "occupationSkillRelations_en.csv"
ESCO_OCCUPATIONS_PATH = RAW_DIR / "esco" / "occupations_en.csv"
ESCO_DICTIONARY_PATH = RAW_DIR / "esco" / "dictionary_en.csv"

SKILLSPAN_TRAIN_PATH = RAW_DIR / "skillspan" / "train.json"
SKILLSPAN_DEV_PATH = RAW_DIR / "skillspan" / "dev.json"
SKILLSPAN_TEST_PATH = RAW_DIR / "skillspan" / "test.json"

ONS_PATH = RAW_DIR / "ons_skills" / "skillscompetenciesandotherjobrequirements.xlsx"

# Interim dataset paths
GOV_CLEANED_PATH = INTERIM_DIR / "cleaned" / "gov" / "cleanedData.csv"
REED_CLEANED_PATH = INTERIM_DIR / "cleaned" / "reed_uk" / "reed_cleaned.csv"
ESCO_SKILLS_CLEANED_PATH = INTERIM_DIR / "cleaned" / "esco" / "esco_skills_cleaned.csv"
GOV_SOURCE_CLEANED_PATH = INTERIM_DIR / "cleaned" / "gov" / "cleanedData.csv"
GOV_CLEANED_PATH = INTERIM_DIR / "cleaned" / "gov" / "gov_cleaned_standardised.csv"