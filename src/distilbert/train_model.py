import ast
import time
import numpy as np
import pandas as pd
import evaluate

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)

from src.config import (
    SKILLSPAN_TRAIN_PATH,
    SKILLSPAN_DEV_PATH,
    SKILLSPAN_TEST_PATH,
    DISTILBERT_RESULTS_PATH,
    DISTILBERT_MODEL_DIR,
)

# model name from Hugging Face
MODEL_NAME = "distilbert-base-uncased"

# labels used in SkillSpan
LABEL_LIST = ["O", "B", "I"]

# turn labels into numbers
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# metric used for token classification
seqeval = evaluate.load("seqeval")


def load_skillspan_file(path) -> pd.DataFrame:
    """
    Load one SkillSpan JSONL file.
    """
    return pd.read_json(path, lines=True)


def safe_parse_list(value):
    """
    Make sure tokens/tags are always treated as lists.
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
    Combine all rows with the same idx into one full document.
    """
    combined_rows = []

    for idx_value, group in df.groupby("idx", sort=True):
        all_tokens = []
        all_skill_tags = []

        # keep the source from the first row
        source_value = group["source"].iloc[0] if "source" in group.columns else "unknown"

        for _, row in group.iterrows():
            tokens = safe_parse_list(row["tokens"])
            skill_tags = safe_parse_list(row["tags_skill"])

            all_tokens.extend(tokens)
            all_skill_tags.extend(skill_tags)

        # only keep rows where tokens and labels match properly
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
    Convert BIO tags into numeric ids.
    """
    return [LABEL2ID[tag] for tag in tags]


def build_hf_dataset(df: pd.DataFrame) -> Dataset:
    """
    Turn a pandas dataframe into a Hugging Face dataset.
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
    Tokenize the text and align the labels to the new subword tokens.
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
            # special tokens get ignored in the loss
            if word_idx is None:
                label_ids.append(-100)

            # first token of a word keeps the real label
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])

            # extra subword pieces get ignored
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def compute_metrics(eval_pred):
    """
    Compute token classification metrics.
    """
    predictions, labels = eval_pred

    # choose the label with the highest score
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred_row, label_row in zip(predictions, labels):
        current_preds = []
        current_labels = []

        for pred_id, label_id in zip(pred_row, label_row):
            # ignore masked labels
            if label_id != -100:
                current_preds.append(ID2LABEL[pred_id])
                current_labels.append(ID2LABEL[label_id])

        true_predictions.append(current_preds)
        true_labels.append(current_labels)

    results = seqeval.compute(
        predictions=true_predictions,
        references=true_labels,
    )

    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def prepare_datasets():
    """
    Load SkillSpan and prepare train/dev/test datasets for DistilBERT.
    """
    # load raw files
    train_df = load_skillspan_file(SKILLSPAN_TRAIN_PATH)
    dev_df = load_skillspan_file(SKILLSPAN_DEV_PATH)
    test_df = load_skillspan_file(SKILLSPAN_TEST_PATH)

    # combine rows into full documents
    train_docs = combine_rows_by_idx(train_df)
    dev_docs = combine_rows_by_idx(dev_df)
    test_docs = combine_rows_by_idx(test_df)

    # build Hugging Face dataset dictionary
    dataset_dict = DatasetDict(
        {
            "train": build_hf_dataset(train_docs),
            "dev": build_hf_dataset(dev_docs),
            "test": build_hf_dataset(test_docs),
        }
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # tokenize and align labels
    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
    )

    return tokenized_datasets, tokenizer


def run_distilbert_training():
    """
    Train and evaluate DistilBERT for skill extraction.
    """
    # start full timer
    overall_start = time.time()
    print("Starting DistilBERT training...")

    # prepare datasets and time it
    prep_start = time.time()
    tokenized_datasets, tokenizer = prepare_datasets()
    prep_end = time.time()

    print("\n--- TOKENIZED DATASETS READY ---")
    print(tokenized_datasets)
    print(f"Dataset preparation time: {prep_end - prep_start:.2f} seconds")

    # load DistilBERT model for token classification
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # handle padding correctly during training
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # training settings
    training_args = TrainingArguments(
        output_dir=str(DISTILBERT_MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
    )

    # build trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["dev"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # train model and time it
    train_start = time.time()
    trainer.train()
    train_end = time.time()

    print(f"\nTraining time: {train_end - train_start:.2f} seconds")

    # evaluate on dev set
    print("\n--- DEV EVALUATION ---")
    dev_start = time.time()
    dev_results = trainer.evaluate(tokenized_datasets["dev"])
    dev_end = time.time()
    print(dev_results)
    print(f"DEV evaluation_baseline time: {dev_end - dev_start:.2f} seconds")

    # evaluate on test set
    print("\n--- TEST EVALUATION ---")
    test_start = time.time()
    test_results = trainer.evaluate(tokenized_datasets["test"])
    test_end = time.time()
    print(test_results)
    print(f"TEST evaluation_baseline time: {test_end - test_start:.2f} seconds")

    # save the results to a text file
    DISTILBERT_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DISTILBERT_RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write("DISTILBERT TOKEN CLASSIFICATION RESULTS\n")
        file.write("=====================================\n\n")
        file.write(f"Dataset preparation time: {prep_end - prep_start:.2f} seconds\n")
        file.write(f"Training time: {train_end - train_start:.2f} seconds\n")
        file.write(f"DEV evaluation_baseline time: {dev_end - dev_start:.2f} seconds\n")
        file.write(f"TEST evaluation_baseline time: {test_end - test_start:.2f} seconds\n\n")
        file.write("DEV RESULTS\n")
        file.write(str(dev_results))
        file.write("\n\nTEST RESULTS\n")
        file.write(str(test_results))

    # end full timer
    overall_end = time.time()
    print(f"\nTotal elapsed time: {overall_end - overall_start:.2f} seconds")
    print(f"Saved results to: {DISTILBERT_RESULTS_PATH}")
    print(f"Saved model output to: {DISTILBERT_MODEL_DIR}")