import ast
from pathlib import Path
import pandas as pd
import torch

from transformers import AutoTokenizer, AutoModelForTokenClassification

from src.config import (
    SKILLSPAN_TEST_PATH,
    DISTILBERT_ROW_MODEL_DIR,
    DISTILBERT_DEBUG_PREDICTIONS_PATH,
)

# original tokenizer name
MODEL_NAME = "distilbert-base-uncased"

# label mapping
LABEL_LIST = ["O", "B", "I"]
ID2LABEL = {0: "O", 1: "B", 2: "I"}


def load_skillspan_file(path):
    # load SkillSpan jsonl file
    return pd.read_json(path, lines=True)


def safe_parse_list(value):
    # make sure tokens/tags are always lists
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


def extract_spans(tokens, tags):
    # turn BIO tags into phrases
    spans = []
    current_tokens = []

    for token, tag in zip(tokens, tags):
        if tag == "B":
            if current_tokens:
                spans.append(" ".join(current_tokens))
            current_tokens = [str(token)]

        elif tag == "I":
            if current_tokens:
                current_tokens.append(str(token))
            else:
                current_tokens = [str(token)]

        else:
            if current_tokens:
                spans.append(" ".join(current_tokens))
                current_tokens = []

    if current_tokens:
        spans.append(" ".join(current_tokens))

    return spans


def load_model_and_tokenizer():
    # load tokenizer from original base model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # find latest checkpoint
    model_dir = Path(DISTILBERT_ROW_MODEL_DIR)
    checkpoint_dirs = sorted(
        [p for p in model_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1])
    )

    if not checkpoint_dirs:
        raise FileNotFoundError("No checkpoint folders were found.")

    latest_checkpoint = checkpoint_dirs[-1]
    print(f"Loading model from checkpoint: {latest_checkpoint}")

    # load trained model
    model = AutoModelForTokenClassification.from_pretrained(latest_checkpoint)
    model.eval()

    return model, tokenizer


def predict_labels(tokens, model, tokenizer):
    # tokenize using the same split-token setup as training
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**encoded)

    pred_ids = torch.argmax(outputs.logits, dim=2)[0].tolist()
    word_ids = encoded.word_ids(batch_index=0)

    predicted_labels = []
    seen_word_ids = set()

    for pred_id, word_id in zip(pred_ids, word_ids):
        # skip special tokens
        if word_id is None:
            continue

        # only keep first subword label
        if word_id in seen_word_ids:
            continue

        seen_word_ids.add(word_id)
        predicted_labels.append(ID2LABEL[pred_id])

    return predicted_labels


def build_debug_text(rows_to_show):
    lines = []
    lines.append("DISTILBERT DEBUG PREDICTIONS")
    lines.append("============================")
    lines.append("")

    for row_info in rows_to_show:
        lines.append(f"Row index: {row_info['row_index']}")
        lines.append(f"idx: {row_info['idx']}")
        lines.append(f"source: {row_info['source']}")
        lines.append(f"tokens: {row_info['tokens']}")
        lines.append(f"gold labels: {row_info['gold_labels']}")
        lines.append(f"predicted labels: {row_info['predicted_labels']}")
        lines.append(f"gold spans: {row_info['gold_spans']}")
        lines.append(f"predicted spans: {row_info['predicted_spans']}")
        lines.append("")

    return "\n".join(lines)


def run_debug_predictions():
    print("Starting DistilBERT debug predictions...")

    # load test data
    test_df = load_skillspan_file(SKILLSPAN_TEST_PATH)

    # load model + tokenizer
    model, tokenizer = load_model_and_tokenizer()

    rows_to_show = []

    # only show rows that actually contain gold skill spans
    shown_count = 0
    row_index = 0

    while row_index < len(test_df) and shown_count < 10:
        row = test_df.iloc[row_index]

        tokens = safe_parse_list(row["tokens"])
        gold_labels = safe_parse_list(row["tags_skill"])

        # skip bad rows
        if not tokens or len(tokens) != len(gold_labels):
            row_index += 1
            continue

        gold_spans = extract_spans(tokens, gold_labels)

        # skip rows with no gold skill spans
        if not gold_spans:
            row_index += 1
            continue

        predicted_labels = predict_labels(tokens, model, tokenizer)

        # trim in case token lengths do not line up exactly
        min_len = min(len(tokens), len(gold_labels), len(predicted_labels))
        tokens = tokens[:min_len]
        gold_labels = gold_labels[:min_len]
        predicted_labels = predicted_labels[:min_len]

        gold_spans = extract_spans(tokens, gold_labels)
        predicted_spans = extract_spans(tokens, predicted_labels)

        row_info = {
            "row_index": row_index,
            "idx": row["idx"],
            "source": row["source"],
            "tokens": tokens,
            "gold_labels": gold_labels,
            "predicted_labels": predicted_labels,
            "gold_spans": gold_spans,
            "predicted_spans": predicted_spans,
        }

        rows_to_show.append(row_info)

        shown_count += 1
        row_index += 1

    debug_text = build_debug_text(rows_to_show)

    print("\n--- DEBUG PREDICTIONS PREVIEW ---")
    print(debug_text[:5000])

    DISTILBERT_DEBUG_PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DISTILBERT_DEBUG_PREDICTIONS_PATH, "w", encoding="utf-8") as file:
        file.write(debug_text)

    print(f"\nSaved debug predictions to: {DISTILBERT_DEBUG_PREDICTIONS_PATH}")