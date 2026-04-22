from src.config import MODEL_COMPARISON_SUMMARY_PATH


def build_model_comparison_text():
    # baseline results
    baseline_precision = 0.0316
    baseline_recall = 0.0294

    # first DistilBERT run using combined documents
    combined_precision = 0.1392
    combined_recall = 0.1494
    combined_f1 = 0.1441
    combined_accuracy = 0.8919

    # second DistilBERT run using row-level samples
    row_precision = 0.3981
    row_recall = 0.4189
    row_f1 = 0.4082
    row_accuracy = 0.9393

    lines = []
    lines.append("MODEL COMPARISON SUMMARY")
    lines.append("========================")
    lines.append("")

    lines.append("1. BASELINE RULE-BASED METHOD")
    lines.append("-----------------------------")
    lines.append(f"Precision: {baseline_precision:.4f}")
    lines.append(f"Recall: {baseline_recall:.4f}")
    lines.append("This baseline used simple ESCO phrase matching.")
    lines.append("It was useful as a starting point, but the results were very weak.")
    lines.append("")

    lines.append("2. DISTILBERT USING COMBINED DOCUMENTS")
    lines.append("--------------------------------------")
    lines.append(f"Precision: {combined_precision:.4f}")
    lines.append(f"Recall: {combined_recall:.4f}")
    lines.append(f"F1: {combined_f1:.4f}")
    lines.append(f"Accuracy: {combined_accuracy:.4f}")
    lines.append("This was better than the baseline.")
    lines.append("However, the number of combined training documents was small.")
    lines.append("This likely made learning harder for the model.")
    lines.append("")

    lines.append("3. DISTILBERT USING ROW-LEVEL SAMPLES")
    lines.append("-------------------------------------")
    lines.append(f"Precision: {row_precision:.4f}")
    lines.append(f"Recall: {row_recall:.4f}")
    lines.append(f"F1: {row_f1:.4f}")
    lines.append(f"Accuracy: {row_accuracy:.4f}")
    lines.append("This was the best result.")
    lines.append("Using the original row-level SkillSpan data gave the model more training examples.")
    lines.append("This improved the extraction performance a lot.")
    lines.append("")

    lines.append("4. FINAL RESULT")
    lines.append("---------------")
    lines.append("The baseline method performed badly and should only be treated as a simple starting point.")
    lines.append("Both DistilBERT runs improved on the baseline.")
    lines.append("The row-level DistilBERT model performed best overall.")
    lines.append("This means the row-level DistilBERT model should be used as the main extraction method.")
    lines.append("")
    lines.append("Best model chosen: DistilBERT trained on row-level SkillSpan samples.")
    lines.append("")
    lines.append("Next step: map extracted skill phrases to ESCO preferred labels and concept URIs.")

    return "\n".join(lines)


def save_model_comparison(text):
    # make the folder if it does not exist
    MODEL_COMPARISON_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)

    # save the summary
    with open(MODEL_COMPARISON_SUMMARY_PATH, "w", encoding="utf-8") as file:
        file.write(text)


def run_model_comparison():
    # start the comparison step
    print("Starting model comparison summary...")

    comparison_text = build_model_comparison_text()

    print("\n--- MODEL COMPARISON SUMMARY PREVIEW ---")
    print(comparison_text)

    save_model_comparison(comparison_text)

    print(f"\nSaved model comparison summary to: {MODEL_COMPARISON_SUMMARY_PATH}")