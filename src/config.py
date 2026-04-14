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
GOV_CLEANED_STANDARDISED_PATH = INTERIM_DIR / "cleaned" / "gov" / "gov_cleaned_standardised.csv"
REED_MATCHED_SKILLS_PATH = INTERIM_DIR / "matched_skills" / "reed_uk" / "reed_esco_matches.csv"
REED_SKILL_FREQUENCY_PATH = INTERIM_DIR / "aggregated" / "reed_uk" / "reed_skill_frequency.csv"
REED_SKILL_FREQUENCY_FILTERED_PATH = INTERIM_DIR / "aggregated" / "reed_uk" / "reed_skill_frequency_filtered.csv"
ONS_TABLE3_CLEANED_PATH = INTERIM_DIR / "cleaned" / "ons_skills" / "ons_table3_cleaned.csv"
ONS_TABLE3_PATH = INTERIM_DIR / "cleaned" / "ons_skills" / "table_3_cleaned.csv"
ONS_TABLE3_LONG_PATH = INTERIM_DIR / "aggregated" / "ons_skills" / "table_3_long.csv"
ONS_TABLE3_TOP_2025_PATH = INTERIM_DIR / "aggregated" / "ons_skills" / "table_3_top_2025.csv"
REED_TOP_SKILLS_CHART_PATH = PROCESSED_DIR / "reed_uk" / "reed_top_skills_chart.png"
# gov output paths
GOV_CLEANED_STANDARDISED_PATH = INTERIM_DIR / "cleaned" / "gov" / "gov_cleaned_standardised.csv"
GOV_MONTHLY_COUNTS_PATH = INTERIM_DIR / "aggregated" / "gov" / "monthly_job_counts.csv"
GOV_MONTHLY_COUNTS_CHART_PATH = PROCESSED_DIR / "gov" / "monthly_job_counts_chart.png"
GOV_CATEGORY_COUNTS_PATH = INTERIM_DIR / "aggregated" / "gov" / "category_counts.csv"
GOV_TOP_CATEGORIES_CHART_PATH = PROCESSED_DIR / "gov" / "top_categories_chart.png"
GOV_TOP_CATEGORY_MONTHLY_PATH = INTERIM_DIR / "aggregated" / "gov" / "top_category_monthly_counts.csv"
GOV_TOP_CATEGORY_MONTHLY_CHART_PATH = PROCESSED_DIR / "gov" / "top_category_monthly_chart.png"
#txt files
ONS_TABLE3_TOP_2025_PATH = INTERIM_DIR / "aggregated" / "ons_skills" / "table_3_top_2025.csv"
ONS_TABLE2_TOP_2025_PATH = INTERIM_DIR / "aggregated" / "ons_skills" / "table_2_top_2025.csv"
COMBINED_SUMMARY_PATH = PROCESSED_DIR / "combined" / "combined_summary.txt"
SKILLSPAN_TRAIN_PATH = RAW_DIR / "skillspan" / "train.json"
SKILLSPAN_DEV_PATH = RAW_DIR / "skillspan" / "dev.json"
SKILLSPAN_TEST_PATH = RAW_DIR / "skillspan" / "test.json"

SKILLSPAN_PREVIEW_PATH = PROCESSED_DIR / "evaluation" / "skillspan_preview.txt"
BASELINE_EVAL_SUMMARY_PATH = PROCESSED_DIR / "evaluation" / "baseline_eval_summary.txt"
BASELINE_EVAL_EXAMPLES_PATH = PROCESSED_DIR / "evaluation" / "baseline_eval_examples.txt"
DISTILBERT_PREP_PREVIEW_PATH = PROCESSED_DIR / "distilbert" / "distilbert_prep_preview.txt"
DISTILBERT_RESULTS_PATH = PROCESSED_DIR / "distilbert" / "distilbert_results.txt"
DISTILBERT_MODEL_DIR = PROCESSED_DIR / "distilbert" / "model_output"
DISTILBERT_ROW_RESULTS_PATH = PROCESSED_DIR / "distilbert" / "distilbert_row_level_results.txt"
DISTILBERT_ROW_MODEL_DIR = PROCESSED_DIR / "distilbert" / "row_level_model_output"
MODEL_COMPARISON_SUMMARY_PATH = PROCESSED_DIR / "evaluation" / "model_comparison_summary.txt"
DISTILBERT_ESCO_MAPPING_PATH = PROCESSED_DIR / "distilbert" / "distilbert_esco_mapping_preview.txt"
DISTILBERT_PIPELINE_DEMO_PATH = PROCESSED_DIR / "distilbert" / "distilbert_pipeline_demo.txt"
DISTILBERT_DEBUG_PREDICTIONS_PATH = PROCESSED_DIR / "distilbert" / "distilbert_debug_predictions.txt"
DISTILBERT_ROW_5EPOCH_RESULTS_PATH = PROCESSED_DIR / "distilbert" / "distilbert_row_level_5epoch_results.txt"
DISTILBERT_ROW_5EPOCH_MODEL_DIR = PROCESSED_DIR / "distilbert" / "row_level_5epoch_model_output"