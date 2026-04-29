import ast
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from src.config import (
    SKILLSPAN_TRAIN_PATH,
    SKILLSPAN_DEV_PATH,
    SKILLSPAN_TEST_PATH,
    DISTILBERT_PREP_PREVIEW_PATH,
)

MODEL_NAME = "distilbert-base-uncased"

LABEL_LIST = ["O", "B", "I"]
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}


def load_skillspan_file(path) -> pd.DataFrame:
    """
    Loads one SkillSpan JSON Lines file.
    """
    return pd.read_json(path, lines=True)


def safe_parse_list(value):
    """
    Makes sure token/tag fields are treated as lists.
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


def combine_rows_by_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines multiple rows with the same idx into one document-level row.
    """
    combined_rows = []

    for idx_value, group in df.groupby("idx", sort=True):
        all_tokens = []
        all_skill_tags = []

        source_value = group["source"].iloc[0] if "source" in group.columns else "unknown"

        for _, row in group.iterrows():
            tokens = safe_parse_list(row["tokens"])
            skill_tags = safe_parse_list(row["tags_skill"])

            all_tokens.extend(tokens)
            all_skill_tags.extend(skill_tags)

        # keep only rows where tokens and tags line up
        if len(all_tokens) == len(all_skill_tags) and len(all_tokens) > 0:
            combined_rows.append(
                {
                    "idx": idx_value,
                    "source": source_value,
                    "tokens": all_tokens,
                    "tags_skill": all_skill_tags,
                }
            )

    return pd.DataFrame(combined_rows)


def convert_tags_to_ids(tags):
    """
    Converts BIO tags into numeric ids.
    """
    return [LABEL2ID[tag] for tag in tags]


def build_hf_dataset(df: pd.DataFrame) -> Dataset:
    """
    Converts a pandas dataframe into a Hugging Face dataset.
    """
    rows = []

    for _, row in df.iterrows():
        rows.append(
            {
                "idx": int(row["idx"]),
                "source": str(row["source"]),
                "tokens": row["tokens"],
                "ner_tags": convert_tags_to_ids(row["tags_skill"]),
            }
        )

    return Dataset.from_list(rows)


def tokenize_and_align_labels(examples, tokenizer):
    """
    Tokenizes token lists and aligns word-level labels to subword tokens.
    """
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
    )

    aligned_labels = []

    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])
            else:
                # same word split into subwords -> ignore continuation token for loss
                label_ids.append(-100)

            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def build_preview_text(dataset_dict, tokenized_train_sample) -> str:
    """
    Creates a readable preview of the prepared datasets.
    """
    lines = []
    lines.append("DISTILBERT DATA PREPARATION PREVIEW")
    lines.append("==================================")
    lines.append("")
    lines.append(f"Train documents: {len(dataset_dict['train'])}")
    lines.append(f"Dev documents: {len(dataset_dict['dev'])}")
    lines.append(f"Test documents: {len(dataset_dict['test'])}")
    lines.append("")
    lines.append(f"Label list: {LABEL_LIST}")
    lines.append(f"Label2id: {LABEL2ID}")
    lines.append("")
    lines.append("Tokenized train sample keys:")
    lines.append(str(tokenized_train_sample.keys()))
    lines.append("")
    lines.append("First train sample:")
    lines.append(str({k: tokenized_train_sample[k] for k in tokenized_train_sample.keys()}))
    lines.append("")

    return "\n".join(lines)


def run_distilbert_data_prep():
    print("Preparing SkillSpan for DistilBERT...")

    # load raw SkillSpan splits
    train_df = load_skillspan_file(SKILLSPAN_TRAIN_PATH)
    dev_df = load_skillspan_file(SKILLSPAN_DEV_PATH)
    test_df = load_skillspan_file(SKILLSPAN_TEST_PATH)

    # combine rows by document idx
    train_docs = combine_rows_by_idx(train_df)
    dev_docs = combine_rows_by_idx(dev_df)
    test_docs = combine_rows_by_idx(test_df)

    print("\n--- DOCUMENT COUNTS ---")
    print(f"Train docs: {len(train_docs)}")
    print(f"Dev docs: {len(dev_docs)}")
    print(f"Test docs: {len(test_docs)}")

    # convert to HF datasets
    dataset_dict = DatasetDict(
        {
            "train": build_hf_dataset(train_docs),
            "dev": build_hf_dataset(dev_docs),
            "test": build_hf_dataset(test_docs),
        }
    )

    print("\n--- HF DATASET BUILT ---")
    print(dataset_dict)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
    )

    print("\n--- TOKENIZATION COMPLETE ---")
    print(tokenized_datasets)

    preview_text = build_preview_text(
        dataset_dict=dataset_dict,
        tokenized_train_sample=tokenized_datasets["train"][0],
    )

    print("\n--- PREVIEW ---")
    print(preview_text[:4000])

    DISTILBERT_PREP_PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DISTILBERT_PREP_PREVIEW_PATH, "w", encoding="utf-8") as file:
        file.write(preview_text)

    print(f"\nSaved prep preview to: {DISTILBERT_PREP_PREVIEW_PATH}")