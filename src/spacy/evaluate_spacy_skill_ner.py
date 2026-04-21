import json
from pathlib import Path

import spacy
from sklearn.metrics import precision_score, recall_score, f1_score

from src.config import (
    SPACY_TEST_DATA_PATH,
    SPACY_MODEL_DIR,
    SPACY_EVALUATION_RESULTS_PATH,
    SPACY_ROW_LEVEL_RESULTS_PATH,
)


def load_spacy_json(path: Path):
    # load the prepared spaCy json file
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def spans_to_set(entities):
    # turn entity list into a set so we can compare exact matches
    return {(start, end, label) for start, end, label in entities}


def run_evaluate_spacy_skill_ner():
    print("Starting spaCy skill NER evaluation...")

    test_data = load_spacy_json(SPACY_TEST_DATA_PATH)
    print(f"Loaded test rows: {len(test_data)}")

    nlp = spacy.load(SPACY_MODEL_DIR)
    print(f"Loaded model from: {SPACY_MODEL_DIR}")

    y_true = []
    y_pred = []

    total_gold_entities = 0
    total_predicted_entities = 0
    total_exact_matches = 0

    row_results = []

    for i, (text, annotations) in enumerate(test_data, start=1):
        gold_entities = annotations.get("entities", [])
        gold_set = spans_to_set(gold_entities)

        doc = nlp(text)
        predicted_entities = [(ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
        pred_set = spans_to_set(predicted_entities)

        total_gold_entities += len(gold_set)
        total_predicted_entities += len(pred_set)
        total_exact_matches += len(gold_set & pred_set)

        # binary row-level view: did the row contain at least one skill span?
        gold_binary = 1 if len(gold_set) > 0 else 0
        pred_binary = 1 if len(pred_set) > 0 else 0

        y_true.append(gold_binary)
        y_pred.append(pred_binary)

        row_results.append(
            {
                "row_number": i,
                "text_preview": text[:200],
                "gold_count": len(gold_set),
                "predicted_count": len(pred_set),
                "exact_matches": len(gold_set & pred_set),
                "gold_entities": list(gold_set),
                "predicted_entities": list(pred_set),
            }
        )

        if i % 500 == 0 or i == len(test_data):
            print(f"Processed {i}/{len(test_data)} rows")

    # row-level binary scores
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # span-level exact-match scores
    span_precision = (
        total_exact_matches / total_predicted_entities if total_predicted_entities > 0 else 0
    )
    span_recall = (
        total_exact_matches / total_gold_entities if total_gold_entities > 0 else 0
    )
    span_f1 = (
        2 * span_precision * span_recall / (span_precision + span_recall)
        if (span_precision + span_recall) > 0
        else 0
    )

    SPACY_EVALUATION_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SPACY_ROW_LEVEL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(SPACY_EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write("SPACY SKILL NER EVALUATION RESULTS\n")
        file.write("=================================\n\n")

        file.write(f"Test rows: {len(test_data)}\n")
        file.write(f"Total gold skill spans: {total_gold_entities}\n")
        file.write(f"Total predicted skill spans: {total_predicted_entities}\n")
        file.write(f"Total exact span matches: {total_exact_matches}\n\n")

        file.write("Row-level binary metrics\n")
        file.write("------------------------\n")
        file.write(f"Precision: {precision:.4f}\n")
        file.write(f"Recall: {recall:.4f}\n")
        file.write(f"F1: {f1:.4f}\n\n")

        file.write("Span-level exact-match metrics\n")
        file.write("------------------------------\n")
        file.write(f"Precision: {span_precision:.4f}\n")
        file.write(f"Recall: {span_recall:.4f}\n")
        file.write(f"F1: {span_f1:.4f}\n")

    with open(SPACY_ROW_LEVEL_RESULTS_PATH, "w", encoding="utf-8") as file:
        file.write("SPACY ROW LEVEL RESULTS\n")
        file.write("=======================\n\n")

        for row in row_results[:200]:
            file.write(f"Row: {row['row_number']}\n")
            file.write(f"Text preview: {row['text_preview']}\n")
            file.write(f"Gold count: {row['gold_count']}\n")
            file.write(f"Predicted count: {row['predicted_count']}\n")
            file.write(f"Exact matches: {row['exact_matches']}\n")
            file.write(f"Gold entities: {row['gold_entities']}\n")
            file.write(f"Predicted entities: {row['predicted_entities']}\n")
            file.write("\n---\n\n")

    print("Saved evaluation results to:")
    print(SPACY_EVALUATION_RESULTS_PATH)
    print(SPACY_ROW_LEVEL_RESULTS_PATH)


if __name__ == "__main__":
    run_evaluate_spacy_skill_ner()