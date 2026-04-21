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
    DISTILBERT_ROW_5EPOCH_RESULTS_PATH,
    DISTILBERT_ROW_5EPOCH_MODEL_DIR,
)

# model name
MODEL_NAME = "distilbert-base-uncased"

# labels used in SkillSpan
LABEL_LIST = ["O", "B", "I"]

# change labels into numbers
LABEL2ID = {label: i for i, label in enumerate(LABEL_LIST)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# metric for evaluation_baseline
seqeval = evaluate.load("seqeval")


def load_skillspan_file(path):
    # load one SkillSpan file
    return pd.read_json(path, lines=True)


def safe_parse_list(value):
    # make sure tokens and tags always stay as lists
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


def convert_tags_to_ids(tags):
    # change B I O into numbers
    return [LABEL2ID[tag] for tag in tags]


def build_hf_dataset_from_rows(df):
    # build a Hugging Face dataset using each row as one sample
    rows = []

    for _, row in df.iterrows():
        tokens = safe_parse_list(row["tokens"])
        tags = safe_parse_list(row["tags_skill"])

        # only keep rows where token count matches tag count
        if len(tokens) == len(tags) and len(tokens) > 0:
            rows.append(
                {
                    "idx": int(row["idx"]),
                    "source": str(row["source"]),
                    "tokens": tokens,
                    "ner_tags": convert_tags_to_ids(tags),
                }
            )

    return Dataset.from_list(rows)


def tokenize_and_align_labels(examples, tokenizer):
    # tokenize the words
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
            # special tokens are ignored
            if word_idx is None:
                label_ids.append(-100)

            # first token of a word keeps the correct label
            elif word_idx != previous_word_idx:
                label_ids.append(labels[word_idx])

            # extra split pieces are ignored
            else:
                label_ids.append(-100)

            previous_word_idx = word_idx

        aligned_labels.append(label_ids)

    tokenized_inputs["labels"] = aligned_labels
    return tokenized_inputs


def compute_metrics(eval_pred):
    # calculate precision recall f1 and accuracy
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for pred_row, label_row in zip(predictions, labels):
        current_preds = []
        current_labels = []

        for pred_id, label_id in zip(pred_row, label_row):
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


def prepare_row_level_datasets():
    # load SkillSpan files
    train_df = load_skillspan_file(SKILLSPAN_TRAIN_PATH)
    dev_df = load_skillspan_file(SKILLSPAN_DEV_PATH)
    test_df = load_skillspan_file(SKILLSPAN_TEST_PATH)

    # build dataset dictionary
    dataset_dict = DatasetDict(
        {
            "train": build_hf_dataset_from_rows(train_df),
            "dev": build_hf_dataset_from_rows(dev_df),
            "test": build_hf_dataset_from_rows(test_df),
        }
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # tokenize datasets
    tokenized_datasets = dataset_dict.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
    )

    return tokenized_datasets, tokenizer


def run_distilbert_row_level_training():
    # start full timer
    overall_start = time.time()
    print("Starting row-level DistilBERT training for 5 epochs...")

    # prepare the data
    prep_start = time.time()
    tokenized_datasets, tokenizer = prepare_row_level_datasets()
    prep_end = time.time()

    print("\n--- DATASETS READY ---")
    print(tokenized_datasets)
    print(f"Dataset preparation time: {prep_end - prep_start:.2f} seconds")

    # load model
    model_load_start = time.time()
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model_load_end = time.time()

    print(f"Model loading time: {model_load_end - model_load_start:.2f} seconds")

    # helps with padding during training
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    # rough training estimate based on previous run
    print("\n--- ROUGH TRAINING ESTIMATE ---")
    print("Previous 3 epoch row-level run took about 30 minutes.")
    print("This 5 epoch run may take around 45 to 55 minutes depending on your machine.")

    # training settings
    training_args = TrainingArguments(
        output_dir=str(DISTILBERT_ROW_5EPOCH_MODEL_DIR),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
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

    # train model
    train_start = time.time()
    trainer.train()
    train_end = time.time()

    print(f"\nTraining time: {train_end - train_start:.2f} seconds")

    # evaluate on dev
    print("\n--- DEV RESULTS ---")
    dev_start = time.time()
    dev_results = trainer.evaluate(tokenized_datasets["dev"])
    dev_end = time.time()
    print(dev_results)
    print(f"DEV evaluation time: {dev_end - dev_start:.2f} seconds")

    # evaluate on test
    print("\n--- TEST RESULTS ---")
    test_start = time.time()
    test_results = trainer.evaluate(tokenized_datasets["test"])
    test_end = time.time()
    print(test_results)
    print(f"TEST evaluation time: {test_end - test_start:.2f} seconds")

    # save results
    DISTILBERT_ROW_5EPOCH_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DISTILBERT_ROW_5EPOCH_RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write("ROW-LEVEL DISTILBERT RESULTS 5 EPOCHS\n")
        file.write("====================================\n\n")
        file.write(f"Dataset preparation time: {prep_end - prep_start:.2f} seconds\n")
        file.write(f"Model loading time: {model_load_end - model_load_start:.2f} seconds\n")
        file.write(f"Training time: {train_end - train_start:.2f} seconds\n")
        file.write(f"DEV evaluation time: {dev_end - dev_start:.2f} seconds\n")
        file.write(f"TEST evaluation time: {test_end - test_start:.2f} seconds\n\n")
        file.write("DEV RESULTS\n")
        file.write(str(dev_results))
        file.write("\n\nTEST RESULTS\n")
        file.write(str(test_results))

    # end full timer
    overall_end = time.time()
    print(f"\nTotal elapsed time: {overall_end - overall_start:.2f} seconds")
    print(f"Saved results to: {DISTILBERT_ROW_5EPOCH_RESULTS_PATH}")
    print(f"Saved model output to: {DISTILBERT_ROW_5EPOCH_MODEL_DIR}")